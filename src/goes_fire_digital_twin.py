"""
Gêmeo digital espacial para **risco de queimada** a partir só de observações GOES.

Segue o ciclo **observação → assimilação → estado atualizado** (cf. ISO/IEC 23247).
Técnicas **não supervisionadas** integradas:

- **Score multi-escala**: vários filtros medianos em T_B13; média das normalizações
  robustas (capta focos de tamanhos diferentes vs. fundo local).
- **Contraste espectral** BT7−BT14 com peso configurável.
- **Fusão temporal**: ``max_persist`` (legado) ou ``prob_or`` — combinação tipo
  probabilística ``1 − (1−r′)(1−s)`` com decaimento ``r′ = r × persistence``.
- **Limiar**: percentil global ou **mistura gaussiana 2 componentes (GMM)** nos
  valores de risco (separa cauda “quente” sem rótulos).
- **LOF espacial** opcional (``LocalOutlierFactor``): remove alarmes **fracos** que são
  inliers em ``[T_risco, média local]`` e ficam abaixo do percentil 72 do risco na cena.

Referências software noutros domínios: Twin4Build
(https://github.com/JBjoernskov/Twin4Build).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, uniform_filter
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor


def _robust_unit(x: np.ndarray, valid: np.ndarray, q_lo: float = 3.0, q_hi: float = 97.0) -> np.ndarray:
    xv = x[valid]
    if xv.size == 0:
        return np.zeros_like(x)
    lo = np.percentile(xv, q_lo)
    hi = np.percentile(xv, q_hi)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def hourly_anomaly_score(
    slot: Dict[int, np.ndarray],
    *,
    median_sizes: Tuple[int, ...] = (5, 9, 15),
    dbt_weight: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score contínuo [0,1] por célula — só GOES.
    ``median_sizes``: ímpares ≥3; média das pontuações multi-escala em T_B13.
    """
    bt13 = slot.get(13)
    if bt13 is None:
        raise ValueError("slot deve incluir canal 13")
    valid = np.isfinite(bt13) & (bt13 > 0)
    bt7 = slot.get(7)
    bt14 = slot.get(14)

    f13 = bt13.copy()
    med13 = np.nanmedian(f13)
    f13 = np.nan_to_num(f13, nan=med13)

    scale_scores: List[np.ndarray] = []
    for sz in median_sizes:
        k = int(sz) if int(sz) % 2 == 1 else int(sz) + 1
        k = max(3, k)
        bg = median_filter(f13, size=k, mode="nearest")
        res = np.maximum(0.0, f13 - bg)
        scale_scores.append(_robust_unit(res, valid))

    score = np.mean(np.stack(scale_scores, axis=0), axis=0)

    if (
        bt7 is not None
        and bt14 is not None
        and np.any(np.isfinite(bt7))
        and np.any(np.isfinite(bt14))
    ):
        valid_btd = valid & np.isfinite(bt7) & np.isfinite(bt14)
        dbt = bt7 - bt14
        base = np.percentile(dbt[valid_btd], 55) if valid_btd.any() else 0.0
        dbt_pos = np.maximum(0.0, dbt - base)
        score_btd = _robust_unit(dbt_pos, valid_btd)
        score = np.where(valid_btd, (score + dbt_weight * score_btd) / (1.0 + dbt_weight), score)
        valid = valid_btd

    score = np.clip(score, 0.0, 1.0)
    return score.astype(np.float64), valid


def fuse_risk_and_score(
    risk: np.ndarray,
    score: np.ndarray,
    valid: np.ndarray,
    persistence: float,
    mode: Literal["max_persist", "prob_or"],
) -> np.ndarray:
    """Atualiza campo de risco dados instantâneo ``score``."""
    p = float(np.clip(persistence, 0.0, 1.0))
    r = np.clip(np.nan_to_num(risk), 0.0, 1.0)
    s = np.clip(np.nan_to_num(score), 0.0, 1.0)
    if mode == "max_persist":
        return np.where(valid, np.maximum(r * p, s), r)
    r_decay = r * p
    combined = 1.0 - (1.0 - r_decay) * (1.0 - s)
    return np.where(valid, combined, r)


@dataclass
class GOESFireDigitalTwinConfig:
    persistence: float = 0.5
    median_size: int = 9
    multiscale_median_sizes: Tuple[int, ...] = (5, 9, 15)
    gaussian_sigma: float = 1.2
    dbt_weight: float = 0.55
    fusion: Literal["max_persist", "prob_or"] = "prob_or"
    threshold_mode: Literal["percentile", "gmm2"] = "percentile"
    """``gmm2``: limiar por mistura de 2 Gaussianas nos valores de risco (não supervisionado)."""
    lof_neighbors: int = 24
    """0 desliga o refinamento Local Outlier Factor."""


class GOESFireDigitalTwin:
    """Estado: campo ``risk`` na grade; assimila observações horárias."""

    def __init__(self, shape: Tuple[int, int], cfg: Optional[GOESFireDigitalTwinConfig] = None):
        self._shape = shape
        self.cfg = cfg or GOESFireDigitalTwinConfig()
        self.risk = np.zeros(shape, dtype=np.float64)

    def _median_sizes(self) -> Tuple[int, ...]:
        ms = self.cfg.multiscale_median_sizes
        if ms and len(ms) > 0:
            return tuple(int(x) for x in ms)
        return (max(3, int(self.cfg.median_size) | 1),)

    def reset(self) -> None:
        self.risk.fill(0.0)

    def ingest_hour_slot(self, slot: Dict[int, np.ndarray]) -> None:
        score, valid = hourly_anomaly_score(
            slot,
            median_sizes=self._median_sizes(),
            dbt_weight=self.cfg.dbt_weight,
        )
        self.risk = fuse_risk_and_score(
            self.risk,
            score,
            valid,
            self.cfg.persistence,
            self.cfg.fusion,
        )

    def ingest_series(self, hourly_slots: List[Dict[int, np.ndarray]]) -> None:
        self.reset()
        for slot in hourly_slots:
            self.ingest_hour_slot(slot)

    def smoothed_risk(self) -> np.ndarray:
        r = np.nan_to_num(self.risk)
        if self.cfg.gaussian_sigma > 1e-6:
            r = gaussian_filter(r, sigma=self.cfg.gaussian_sigma, mode="nearest")
        return r

    def _percentile_mask(self, rs: np.ndarray, vb: np.ndarray, contamination: float) -> np.ndarray:
        thr = np.percentile(rs[vb], 100.0 * (1.0 - min(max(contamination, 0.001), 0.5)))
        return (rs >= thr) & vb

    def _gmm2_mask(self, rs: np.ndarray, vb: np.ndarray, contamination: float) -> np.ndarray:
        """GMM-2 nos valores de risco; limiar na **probabilidade posterior** do modo mais quente."""
        X = rs[vb].reshape(-1, 1)
        if X.shape[0] < 80:
            return self._percentile_mask(rs, vb, contamination)
        gmm = GaussianMixture(
            n_components=2,
            covariance_type="diag",
            random_state=42,
            reg_covar=1e-4,
            max_iter=200,
        )
        gmm.fit(X)
        hot_k = int(np.argmax(gmm.means_.ravel()))
        p_hot = gmm.predict_proba(X)[:, hot_k]
        thr_p = np.percentile(
            p_hot,
            100.0 * (1.0 - min(max(contamination, 0.001), 0.5)),
        )
        pred = np.zeros(rs.shape, dtype=bool)
        pred[vb] = p_hot >= thr_p
        return pred & vb

    def _lof_refine(self, rs: np.ndarray, vb: np.ndarray, base_pred: np.ndarray) -> np.ndarray:
        """
        Remove apenas candidatos **fracos** que são inliers LOF (contexto normal)
        e risco abaixo de um percentil global — reduz FP sem exigir outlier em todo TP.
        """
        nn = int(self.cfg.lof_neighbors)
        if nn <= 0:
            return base_pred
        loc = uniform_filter(rs, size=5, mode="nearest")
        idx = np.flatnonzero(vb.ravel())
        if idx.size < nn + 15:
            return base_pred
        X = np.column_stack([rs.ravel()[idx], loc.ravel()[idx]])
        n_neigh = max(5, min(nn, X.shape[0] - 1))
        try:
            lof = LocalOutlierFactor(n_neighbors=n_neigh, novelty=False, contamination="auto")
        except TypeError:
            lof = LocalOutlierFactor(n_neighbors=n_neigh, novelty=False, contamination=0.08)
        y = lof.fit_predict(X)
        normal_ctx = np.zeros(self._shape, dtype=bool)
        normal_ctx.ravel()[idx] = y == 1
        soft_cut = np.percentile(rs[vb], 72.0)
        strip = base_pred & normal_ctx & (rs < soft_cut)
        return base_pred & ~strip

    def predict_mask(self, contamination: float, valid_bins: np.ndarray) -> np.ndarray:
        rs = self.smoothed_risk()
        vb = valid_bins & np.isfinite(rs)
        if not vb.any():
            return np.zeros(self._shape, dtype=bool)
        if self.cfg.threshold_mode == "percentile":
            pred = self._percentile_mask(rs, vb, contamination)
        else:
            pred = self._gmm2_mask(rs, vb, contamination)
        pred = self._lof_refine(rs, vb, pred)
        return pred & vb
