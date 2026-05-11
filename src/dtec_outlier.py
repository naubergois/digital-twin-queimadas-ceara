"""
Camada de **outliers sobre o estado do gêmeo digital** (DTEC §3 estendido).

Mantém o gêmeo digital no centro: o detector de outliers opera **sobre as
features produzidas pelo `GOESFireDigitalTwin`** (`twin_risk`, `bt13_max`,
`bt7_max`, `btd_median`, `bt13_anom_21`, `persist_h`). A ideia é simples:

1. O gêmeo digital fornece um campo de risco contínuo (assinatura térmica
   reforçada por persistência).
2. A cabeça supervisionada produz `P(fogo|features)`.
3. Antes do NMS, filtramos as células previstas que **não são outliers**
   no espaço de features — só ficam candidatas com perfil estatisticamente
   anómalo dentro da própria cena. Isto sobe a precisão sem perder o conceito
   de gêmeo digital (é o twin que define o que é "normal").

Algoritmos suportados (não supervisionados, sklearn):

- ``isolation_forest`` — bom para alta dimensionalidade, pouco hipotético.
- ``local_outlier_factor`` — densidade local; útil quando o "normal" varia
  por região da cena.
- ``elliptic_envelope`` — assume Gaussiana multivariada robusta.

O modo ``ensemble`` exige que ≥ 2 dos 3 considerem a célula outlier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


OutlierKind = Literal[
    "isolation_forest",
    "local_outlier_factor",
    "elliptic_envelope",
    "ensemble",
]


@dataclass
class OutlierConfig:
    method: OutlierKind = "isolation_forest"
    contamination: float = 0.02
    """Fração esperada de outliers (cells "anómalas") na cena."""
    n_neighbors_lof: int = 24
    n_estimators_if: int = 200
    random_state: int = 42
    ensemble_min_agree: int = 2
    """Mínimo de algoritmos que devem concordar (modo `ensemble`)."""
    feature_names: Tuple[str, ...] = field(
        default_factory=lambda: (
            "twin_risk",
            "bt13_max",
            "bt7_max",
            "btd_median",
            "bt13_anom_21",
            "persist_h",
        )
    )


def _stack_features_at(
    feats: Dict[str, np.ndarray],
    mask: np.ndarray,
    names: Tuple[str, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Empilha features ``names`` nas células válidas — devolve (N, F) e índices."""
    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return np.empty((0, len(names)), dtype=np.float64), idx
    cols = []
    for n in names:
        if n not in feats:
            raise KeyError(f"feature {n!r} não encontrada (chaves: {sorted(feats)})")
        cols.append(feats[n].ravel()[idx])
    X = np.column_stack(cols).astype(np.float64)
    return X, idx


def _fit_and_predict_outliers(
    X: np.ndarray,
    cfg: OutlierConfig,
) -> np.ndarray:
    """Devolve booleano (N,): True = outlier (candidato a fogo)."""
    if X.shape[0] < 50:
        return np.zeros(X.shape[0], dtype=bool)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    cont = float(np.clip(cfg.contamination, 0.001, 0.5))

    def _iforest() -> np.ndarray:
        clf = IsolationForest(
            contamination=cont,
            n_estimators=cfg.n_estimators_if,
            random_state=cfg.random_state,
        )
        labels = clf.fit_predict(Xs)
        return labels == -1

    def _lof() -> np.ndarray:
        n_neigh = max(5, min(cfg.n_neighbors_lof, Xs.shape[0] - 1))
        clf = LocalOutlierFactor(
            n_neighbors=n_neigh,
            contamination=cont,
            novelty=False,
        )
        labels = clf.fit_predict(Xs)
        return labels == -1

    def _envelope() -> np.ndarray:
        try:
            clf = EllipticEnvelope(
                contamination=cont,
                random_state=cfg.random_state,
                support_fraction=None,
            )
            labels = clf.fit_predict(Xs)
            return labels == -1
        except Exception:
            return np.zeros(Xs.shape[0], dtype=bool)

    if cfg.method == "isolation_forest":
        return _iforest()
    if cfg.method == "local_outlier_factor":
        return _lof()
    if cfg.method == "elliptic_envelope":
        return _envelope()
    if cfg.method == "ensemble":
        votes = _iforest().astype(int) + _lof().astype(int) + _envelope().astype(int)
        return votes >= int(cfg.ensemble_min_agree)
    raise ValueError(f"método de outlier desconhecido: {cfg.method!r}")


def outlier_mask_from_twin_features(
    feats: Dict[str, np.ndarray],
    valid_bins: np.ndarray,
    *,
    cfg: Optional[OutlierConfig] = None,
) -> np.ndarray:
    """
    Devolve máscara H×W onde True = célula classificada como outlier sobre
    o espaço de features do gêmeo digital. Não usa rótulos.
    """
    c = cfg or OutlierConfig()
    X, idx = _stack_features_at(feats, valid_bins, c.feature_names)
    out = _fit_and_predict_outliers(X, c)
    grid = np.zeros(valid_bins.shape, dtype=bool)
    grid.ravel()[idx] = out
    return grid


def filter_predictions_by_outlier(
    candidate_mask: np.ndarray,
    feats: Dict[str, np.ndarray],
    valid_bins: np.ndarray,
    *,
    cfg: Optional[OutlierConfig] = None,
) -> np.ndarray:
    """
    Mantém só as células do ``candidate_mask`` que também são outliers
    no espaço de features do gêmeo (AND). Sobe precisão sem mexer no
    classificador a montante.
    """
    out_mask = outlier_mask_from_twin_features(feats, valid_bins, cfg=cfg)
    return candidate_mask & out_mask & valid_bins
