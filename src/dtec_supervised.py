"""
Cabeça supervisionada leve do DTEC (ver ``docs/METODOLOGIA_DTEC_F1_080.md`` §3).

Aprende ``P(fogo | features)`` por **regressão logística** sobre features
absolutas (BT13, BT7) e contrastes moderados (BTD, BT13−BT14), validada
com **bloqueio espacial** dentro do bbox do Ceará para evitar fuga
trivial. Mantém o gêmeo digital como peça central: as features incluem
o campo de risco contínuo de ``GOESFireDigitalTwin``.

Features (por célula, após fusão multi-hora):

- ``bt13_max``      — max de BT13 nas horas (assinatura térmica)
- ``bt7_max``       — max de BT7 (3,9 μm; sensível a fogo)
- ``btd_median``    — mediana de BT7−BT14 (rejeita extremos de glint solar)
- ``twin_risk``     — risco do gêmeo digital com persistência
- ``bt13_anom_21``  — resíduo BT13 contra mediana 21×21 (~150 km)
- ``persist_h``     — nº horas com BT13 acima do p70 da hora

Modelo: regressão logística regularizada (``LogisticRegression(C=1.0)``).
Saída: máscara binária H×W por limiar de probabilidade calibrado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, median_filter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:  # pragma: no cover - opcional
    from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
except Exception:  # pragma: no cover
    HistGradientBoostingClassifier = None  # type: ignore

from src.goes_fire_digital_twin import GOESFireDigitalTwin, GOESFireDigitalTwinConfig


FEATURE_NAMES: Tuple[str, ...] = (
    "bt13_max",
    "bt7_max",
    "btd_median",
    "twin_risk",
    "bt13_anom_21",
    "persist_h",
)


def _fill(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x), x, np.nanmedian(x))


def build_features(
    hourly: List[Dict[int, np.ndarray]],
    valid_bins: np.ndarray,
    *,
    twin_cfg: Optional[GOESFireDigitalTwinConfig] = None,
) -> Dict[str, np.ndarray]:
    """Constrói features H×W a partir do cubo horário GOES."""
    if not hourly:
        raise ValueError("hourly vazio")

    bt13_stack = np.stack([slot[13] for slot in hourly], axis=0)
    bt7_stack = np.stack([slot[7] for slot in hourly], axis=0)
    bt14_stack = np.stack([slot[14] for slot in hourly], axis=0)
    btd_stack = bt7_stack - bt14_stack

    bt13_max = np.nanmax(bt13_stack, axis=0)
    bt7_max = np.nanmax(bt7_stack, axis=0)
    btd_median = np.nanmedian(btd_stack, axis=0)

    f13 = _fill(bt13_max)
    bg21 = median_filter(f13, size=21, mode="nearest")
    bt13_anom_21 = f13 - bg21

    twin = GOESFireDigitalTwin(bt13_max.shape, twin_cfg or GOESFireDigitalTwinConfig(lof_neighbors=0))
    twin.ingest_series(hourly)
    risk = twin.smoothed_risk()

    persist_h = np.zeros_like(f13)
    for h_idx in range(bt13_stack.shape[0]):
        b = bt13_stack[h_idx]
        v = np.isfinite(b) & valid_bins
        if v.any():
            thr = float(np.percentile(b[v], 70.0))
            persist_h += (b >= thr).astype(np.float64)

    return {
        "bt13_max": _fill(bt13_max),
        "bt7_max": _fill(bt7_max),
        "btd_median": _fill(btd_median),
        "twin_risk": _fill(risk),
        "bt13_anom_21": _fill(bt13_anom_21),
        "persist_h": persist_h,
    }


def stack_features(feats: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Empilha por canal — devolve matriz (N, F) e shape original."""
    arrays = [feats[name] for name in FEATURE_NAMES]
    h, w = arrays[0].shape
    X = np.stack([a.ravel() for a in arrays], axis=1)
    return X, (h, w)


ClassifierKind = Literal["logreg", "hgb"]


@dataclass
class DTECSupervisedConfig:
    C: float = 0.5
    class_weight: str = "balanced"
    threshold: float = 0.50
    max_iter: int = 400
    twin_cfg: Optional[GOESFireDigitalTwinConfig] = None
    feature_names: Tuple[str, ...] = field(default_factory=lambda: FEATURE_NAMES)
    classifier: ClassifierKind = "logreg"
    hgb_max_depth: int = 4
    hgb_learning_rate: float = 0.07
    hgb_max_iter: int = 200
    smooth_sigma: float = 1.0
    """Suavização Gaussiana do campo de probabilidade antes do NMS."""
    nms_radius: int = 2
    """Raio do filtro `maximum` (em células) para Non-Maximum Suppression."""
    nms_min_prob: float = 0.5
    """Probabilidade mínima para um pico contar como deteção."""


@dataclass
class DTECSupervisedModel:
    scaler: StandardScaler
    clf: object
    feature_names: Tuple[str, ...]
    threshold: float
    smooth_sigma: float = 1.0
    nms_radius: int = 2
    nms_min_prob: float = 0.5

    def predict_proba_grid(
        self,
        feats: Dict[str, np.ndarray],
        valid_bins: np.ndarray,
    ) -> np.ndarray:
        X, shape = stack_features({n: feats[n] for n in self.feature_names})
        Xs = self.scaler.transform(X)
        p = self.clf.predict_proba(Xs)[:, 1].reshape(shape)
        p = np.where(valid_bins, p, 0.0)
        return p

    def predict_mask(
        self,
        feats: Dict[str, np.ndarray],
        valid_bins: np.ndarray,
        *,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        thr = self.threshold if threshold is None else float(threshold)
        p = self.predict_proba_grid(feats, valid_bins)
        return (p >= thr) & valid_bins

    def predict_nms(
        self,
        feats: Dict[str, np.ndarray],
        valid_bins: np.ndarray,
        *,
        threshold: Optional[float] = None,
        nms_radius: Optional[int] = None,
        smooth_sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Devolve máscara só com picos locais do campo de probabilidade.
        Cada pico = uma deteção (não toda a "região quente"). Reduz FP
        sem sacrificar recall quando focos estão espacialmente espaçados.
        """
        thr = self.nms_min_prob if threshold is None else float(threshold)
        radius = self.nms_radius if nms_radius is None else int(nms_radius)
        sigma = self.smooth_sigma if smooth_sigma is None else float(smooth_sigma)
        p = self.predict_proba_grid(feats, valid_bins)
        if sigma > 1e-6:
            ps = gaussian_filter(p, sigma=sigma, mode="nearest")
        else:
            ps = p
        size = max(3, 2 * int(radius) + 1)
        local_max = maximum_filter(ps, size=size, mode="nearest")
        peaks = (ps == local_max) & (ps >= thr) & valid_bins
        return peaks


def train_dtec_supervised(
    feats: Dict[str, np.ndarray],
    truth_cells: np.ndarray,
    valid_bins: np.ndarray,
    *,
    cfg: Optional[DTECSupervisedConfig] = None,
    train_mask: Optional[np.ndarray] = None,
) -> DTECSupervisedModel:
    """
    Treina o classificador logístico. ``train_mask`` opcional restringe ao
    bloco espacial de treino (DTEC §6); se omitido, usa toda a cena válida.
    """
    c = cfg or DTECSupervisedConfig()
    X_full, shape = stack_features({n: feats[n] for n in c.feature_names})
    y_full = truth_cells.ravel().astype(np.int8)
    valid_flat = valid_bins.ravel()
    if train_mask is not None:
        valid_flat = valid_flat & train_mask.ravel()
    X = X_full[valid_flat]
    y = y_full[valid_flat]

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    if c.classifier == "hgb" and HistGradientBoostingClassifier is not None:
        clf: object = HistGradientBoostingClassifier(
            max_depth=c.hgb_max_depth,
            learning_rate=c.hgb_learning_rate,
            max_iter=c.hgb_max_iter,
            class_weight=c.class_weight if c.class_weight != "balanced" else "balanced",
            random_state=42,
        )
        clf.fit(Xs, y)
    else:
        clf = LogisticRegression(
            C=c.C,
            class_weight=c.class_weight,
            max_iter=c.max_iter,
            solver="liblinear",
        )
        clf.fit(Xs, y)
    return DTECSupervisedModel(
        scaler=scaler,
        clf=clf,
        feature_names=c.feature_names,
        threshold=c.threshold,
        smooth_sigma=c.smooth_sigma,
        nms_radius=c.nms_radius,
        nms_min_prob=c.nms_min_prob,
    )


def blockwise_spatial_folds(
    shape: Tuple[int, int],
    *,
    n_blocks_lat: int = 3,
    n_blocks_lon: int = 3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Devolve lista de pares ``(train_mask, test_mask)`` cobrindo a grade.
    Cada fold deixa **um** bloco fora para teste.
    """
    h, w = shape
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    lat_splits = np.linspace(0, h, n_blocks_lat + 1, dtype=int)
    lon_splits = np.linspace(0, w, n_blocks_lon + 1, dtype=int)
    for i in range(n_blocks_lat):
        for j in range(n_blocks_lon):
            test = np.zeros(shape, dtype=bool)
            test[lat_splits[i]:lat_splits[i + 1], lon_splits[j]:lon_splits[j + 1]] = True
            train = ~test
            folds.append((train, test))
    return folds
