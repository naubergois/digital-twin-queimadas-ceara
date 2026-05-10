"""
Metodologia **combined_persistence** (Linha B — ver ``docs/METODOLOGIA_NOVA_PROPOSTA.md``).

Fusão de pico temporal, média e persistência ponderada, com extensões opcionais:

- **Percentil por hora** fixo, lista por hora, ou **adaptativo** (sobe o percentil em cenas
  com pouco contraste em ``s_t``, para reduzir ruído uniforme).
- **Portão duplo de persistência**: fração mínima ``p`` e/ou número mínimo de horas ativas.
- **Opening fraco condicional**: só depois do limiar global, se a máscara já tiver densidade
  suficiente — estrutura em cruz (4-viz.) em vez de bloco 3×3.
- **Remoção de componentes pequenas** (sal-pimenta espacial sem esvaziar picos isolados de fogo).

Ver ``CombinedPersistenceConfig`` para todos os parâmetros.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_opening, generate_binary_structure, label

from src.goes_fire_digital_twin import hourly_anomaly_score


def _robust_unit_grid(
    x: np.ndarray,
    valid: np.ndarray,
    q_lo: float = 5.0,
    q_hi: float = 95.0,
) -> np.ndarray:
    xv = x[valid]
    if xv.size == 0:
        return np.zeros_like(x)
    lo = np.percentile(xv, q_lo)
    hi = np.percentile(xv, q_hi)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _percentile_for_hour_layer(
    s_h: np.ndarray,
    vb: np.ndarray,
    base_pct: float,
    *,
    adaptive: bool,
    adaptive_span: float = 7.0,
    adaptive_clip: Tuple[float, float] = (78.0, 92.0),
) -> float:
    """
    Percentil usado para marcar células “ativas” na hora ``h``.

    Modo adaptativo: se o spread robusto (P90−P10) de ``s_h`` na máscara válida for baixo,
    sobe-se o percentil (menos células ativas → menos FP em cena plana).
    """
    xs = s_h[vb]
    if adaptive and xs.size >= 64:
        spread = float(np.percentile(xs, 90) - np.percentile(xs, 10))
        ref = 0.12
        bump = adaptive_span * max(0.0, (ref - spread) / (ref + 1e-6))
        pct = float(np.clip(base_pct + bump, adaptive_clip[0], adaptive_clip[1]))
    else:
        pct = float(base_pct)
    return float(np.percentile(xs, pct))


def _resolve_min_active_hours(nh: int, cfg_val: int) -> Optional[int]:
    """cfg_val: ``0`` desliga; ``<0`` automático; ``>0`` fixo."""
    if nh < 2:
        return None
    if cfg_val == 0:
        return None
    if cfg_val < 0:
        return 2 if nh >= 3 else 1
    return min(cfg_val, nh)


def _remove_small_components(pred: np.ndarray, min_cells: int) -> np.ndarray:
    if min_cells <= 1 or not pred.any():
        return pred
    lab, n = label(pred)
    if n == 0:
        return pred
    out = np.zeros_like(pred, dtype=bool)
    for k in range(1, n + 1):
        m = lab == k
        if int(np.sum(m)) >= min_cells:
            out |= m
    return out


@dataclass
class CombinedPersistenceConfig:
    median_sizes: Tuple[int, ...] = (5, 9, 15)
    dbt_weight: float = 0.55
    hour_active_percentile: float = 88.0
    """Percentil global por hora (se ``hour_active_percentiles`` for None)."""
    hour_active_percentiles: Optional[Tuple[float, ...]] = None
    """Um percentil por snapshot; deve ter o mesmo comprimento que horas."""
    adaptive_hour_percentile: bool = True
    adaptive_hour_span: float = 7.5
    adaptive_hour_clip: Tuple[float, float] = (79.0, 93.0)
    weights_peak_mean_persist: Tuple[float, float, float] = (0.42, 0.21, 0.37)
    min_persist_frac: Optional[float] = None
    """Se None: ``max(persist_floor, persist_scale / n_eff_mediano)``."""
    persist_scale: float = 0.56
    persist_floor: float = 0.16
    min_active_hours: int = -1
    """Exige pelo menos este número de horas binárias ativas; ``0`` desliga; ``-1`` automático."""
    robust_norm_quantiles: Tuple[float, float] = (6.0, 94.0)
    morph_open_iterations: int = 0
    """Opening bloco 3×3 (agressivo); preferir ``weak_open_*``."""
    weak_open_after_pred_cells: int = 260
    """Só aplica opening fraco se ``sum(pred) >=`` este valor; ``0`` desliga."""
    weak_open_iterations: int = 1
    weak_open_connectivity: Literal["cross", "box"] = "cross"
    min_component_cells: int = 3
    """Remove CC com menos que este número de células (``<=1`` desliga)."""


def combined_persistence_precision_preset() -> CombinedPersistenceConfig:
    """Limiatres mais conservadores + mais peso na persistência (menos FP). Usar com ``--precision-focus``."""
    return CombinedPersistenceConfig(
        hour_active_percentile=91.0,
        adaptive_hour_clip=(83.0, 94.0),
        adaptive_hour_span=8.5,
        weights_peak_mean_persist=(0.39, 0.17, 0.44),
        persist_floor=0.26,
        persist_scale=0.68,
        weak_open_after_pred_cells=85,
        weak_open_iterations=2,
        weak_open_connectivity="cross",
        min_component_cells=5,
        robust_norm_quantiles=(7.0, 93.0),
    )


def detect_combined_persistence(
    hourly_grids: List[Dict[int, np.ndarray]],
    valid_bins: np.ndarray,
    contamination: float,
    *,
    cfg: Optional[CombinedPersistenceConfig] = None,
    # Compatibilidade com chamadas antigas (kwargs sobrepõem campos do cfg base)
    median_sizes: Optional[Tuple[int, ...]] = None,
    dbt_weight: Optional[float] = None,
    hour_active_percentile: Optional[float] = None,
    weights_peak_mean_persist: Optional[Tuple[float, float, float]] = None,
    min_persist_frac: Optional[float] = None,
    morph_open_iterations: Optional[int] = None,
) -> np.ndarray:
    """
    Deteção não supervisionada com reforço temporal.

    Passar só ``cfg`` ou usar kwargs legados; kwargs não-nulos substituem o campo homónimo em ``cfg``.
    """
    base = cfg or CombinedPersistenceConfig()
    ov: Dict[str, object] = {}
    if median_sizes is not None:
        ov["median_sizes"] = median_sizes
    if dbt_weight is not None:
        ov["dbt_weight"] = dbt_weight
    if hour_active_percentile is not None:
        ov["hour_active_percentile"] = hour_active_percentile
    if weights_peak_mean_persist is not None:
        ov["weights_peak_mean_persist"] = weights_peak_mean_persist
    if min_persist_frac is not None:
        ov["min_persist_frac"] = min_persist_frac
    if morph_open_iterations is not None:
        ov["morph_open_iterations"] = morph_open_iterations
    c = replace(base, **ov) if ov else base

    if not hourly_grids:
        return np.zeros_like(valid_bins, dtype=bool)

    nh = len(hourly_grids)
    pct_per_hour: Optional[Sequence[float]] = c.hour_active_percentiles
    if pct_per_hour is not None and len(pct_per_hour) != nh:
        raise ValueError(
            f"hour_active_percentiles tem len {len(pct_per_hour)} mas há {nh} snapshots horários."
        )

    scores: List[np.ndarray] = []
    valids: List[np.ndarray] = []
    for slot in hourly_grids:
        s, v = hourly_anomaly_score(
            slot,
            median_sizes=c.median_sizes,
            dbt_weight=c.dbt_weight,
        )
        scores.append(s)
        valids.append(v)

    S = np.stack(scores, axis=0)

    persist_num = np.zeros(S.shape[1:], dtype=np.float64)
    persist_den = np.zeros(S.shape[1:], dtype=np.float64)
    active = np.zeros_like(S, dtype=bool)

    q_lo, q_hi = c.robust_norm_quantiles

    for h in range(nh):
        vb = valids[h] & valid_bins
        if not vb.any():
            continue
        base_pct = float(pct_per_hour[h]) if pct_per_hour is not None else c.hour_active_percentile
        thr_h = _percentile_for_hour_layer(
            S[h],
            vb,
            base_pct,
            adaptive=c.adaptive_hour_percentile,
            adaptive_span=c.adaptive_hour_span,
            adaptive_clip=c.adaptive_hour_clip,
        )
        active[h] = (S[h] >= thr_h) & vb
        persist_num += active[h].astype(np.float64)
        persist_den += vb.astype(np.float64)

    persist = np.divide(persist_num, np.maximum(persist_den, 1e-9))
    peak = np.nanmax(S, axis=0)
    mean_s = np.nanmean(S, axis=0)

    vb_eval = valid_bins & np.isfinite(peak) & np.isfinite(mean_s) & (persist_den >= 1)
    if not vb_eval.any():
        return np.zeros_like(valid_bins, dtype=bool)

    wp, wm, wpr = c.weights_peak_mean_persist
    wsum = wp + wm + wpr
    wp, wm, wpr = wp / wsum, wm / wsum, wpr / wsum

    comb = (
        wp * _robust_unit_grid(peak, vb_eval, q_lo=q_lo, q_hi=q_hi)
        + wm * _robust_unit_grid(mean_s, vb_eval, q_lo=q_lo, q_hi=q_hi)
        + wpr * persist
    )

    ct = min(max(contamination, 0.001), 0.5)
    thr_f = np.percentile(comb[vb_eval], 100.0 * (1.0 - ct))
    pred = (comb >= thr_f) & vb_eval

    n_eff = float(np.median(persist_den[valid_bins & (persist_den >= 1)]))
    n_eff = max(1.0, n_eff)
    if n_eff >= 2:
        mp = (
            c.min_persist_frac
            if c.min_persist_frac is not None
            else max(c.persist_floor, c.persist_scale / n_eff)
        )
        pred &= persist >= mp

    k_hours = _resolve_min_active_hours(int(nh), c.min_active_hours)
    if k_hours is not None:
        pred &= persist_num >= float(k_hours)

    if c.weak_open_after_pred_cells > 0 and int(np.sum(pred)) >= c.weak_open_after_pred_cells:
        struct = (
            generate_binary_structure(2, 1)
            if c.weak_open_connectivity == "cross"
            else np.ones((3, 3), dtype=bool)
        )
        for _ in range(max(0, c.weak_open_iterations)):
            pred = binary_opening(pred, structure=struct)

    if c.min_component_cells > 1:
        pred = _remove_small_components(pred, c.min_component_cells)

    if c.morph_open_iterations > 0:
        struct = np.ones((3, 3), dtype=bool)
        for _ in range(c.morph_open_iterations):
            pred = binary_opening(pred, structure=struct)

    return pred & valid_bins


def fuse_masks_intersection(*masks: np.ndarray) -> np.ndarray:
    """Fusão **não supervisionada** tipo AND: só mantém acordo entre vários detectores (ganha precisão)."""
    if not masks:
        raise ValueError("fuse_masks_intersection requer pelo menos uma máscara")
    out = masks[0].astype(bool).copy()
    for m in masks[1:]:
        out &= m.astype(bool)
    return out


def fuse_masks_union(*masks: np.ndarray) -> np.ndarray:
    """Fusão OR — recupera cobertura (recall) quando detectores são complementares."""
    if not masks:
        raise ValueError("fuse_masks_union requer pelo menos uma máscara")
    out = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        out |= m.astype(bool)
    return out
