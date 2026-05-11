"""
DTEC — Dual-Twin Event-Centric detector (ver ``docs/METODOLOGIA_DTEC_F1_080.md``).

Detector reformulado a partir das descobertas no diagnóstico (Ceará, 2024-10-31):

1. Focos INPE não estão nos pixels mais quentes da cena: têm
   ``T_B13 ≈ 306 K``, percentil ~75 da cena, enquanto a cena tem picos a
   314 K (afloramentos/cidades, não fogo).
2. Mediana de ``BT7 − BT14`` nos focos é **menor** que a mediana da cena:
   pixels com BTD muito alto durante o dia tendem a ser **reflexão solar** em
   areia/rocha clara do sertão, não fogo. A heurística clássica
   "BTD alto = fogo" falha de dia em paisagens semi-áridas.

Implicações para o detector:

- **Anomalia em escala fina** (5–7 células ≈ 30–50 km, não 15 ≈ 90 km).
- **Janela BTD**: aceitar só ``BTD ∈ [bt_lo, bt_hi]``; rejeitar a cauda
  alta como reflexão solar. Limiares por percentis da cena (auto-adaptativos).
- **Persistência relativa**: o sinal é a anomalia ``ΔT13`` ao longo das
  horas, não o pico absoluto.
- **Filtro de cintilação solar** opcional: rejeita pixels onde ``T7`` está
  num percentil extremo da cena (probabilidade de sun glint).

O detector continua **não supervisionado**: nada dos focos INPE entra no
modelo; só estatísticas espaciais e temporais da própria cena.

Saída idêntica aos outros detectores: máscara binária H×W para fundir com
o twin existente ou avaliar com ``evaluate_event_centric``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_opening,
    generate_binary_structure,
    label,
    median_filter,
    uniform_filter,
)


@dataclass
class DTECConfig:
    # Escalas de anomalia local — propositadamente pequenas
    fine_median_sizes: Tuple[int, ...] = (3, 5, 7)

    # Janela BTD (BT7 - BT14): percentis aceitáveis da cena
    btd_low_percentile: float = 35.0
    btd_high_percentile: float = 90.0
    """Acima do high → suspeito de reflexão solar; rejeitar."""

    # Filtro de cintilação solar via BT7
    bt7_glint_percentile: float = 99.2
    """Acima → sun glint; rejeitar."""

    # Persistência: número mínimo de horas com anomalia positiva
    min_active_hours: int = 2

    # Sensibilidade do limiar global no risco combinado
    risk_top_percentile: float = 98.0
    """Top X% das células válidas passa pelo limiar global; ajustável."""

    # Pós-processamento morfológico
    weak_open_iterations: int = 0
    min_component_cells: int = 1
    max_component_cells: int = 200
    """Componentes maiores que isto são `manchas frias` (cobertura nublada,
    fronteira de massa de ar) — provavelmente não-fogo. 0 desliga."""

    # Robust normalização para o score combinado
    robust_norm_quantiles: Tuple[float, float] = (5.0, 95.0)

    # Pesos dos sinais
    weight_anomaly: float = 0.55
    weight_btd_band: float = 0.20
    weight_persistence: float = 0.25


def _robust_unit(x: np.ndarray, valid: np.ndarray, q_lo: float, q_hi: float) -> np.ndarray:
    xv = x[valid]
    if xv.size == 0:
        return np.zeros_like(x)
    lo = np.percentile(xv, q_lo)
    hi = np.percentile(xv, q_hi)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def _fine_anomaly(
    bt13: np.ndarray,
    valid: np.ndarray,
    sizes: Tuple[int, ...],
) -> np.ndarray:
    """Média dos residuais positivos contra mediana local em escalas pequenas."""
    med = np.nanmedian(bt13[valid]) if valid.any() else 0.0
    filled = np.where(valid, bt13, med)
    scores: List[np.ndarray] = []
    for sz in sizes:
        k = int(sz) if int(sz) % 2 == 1 else int(sz) + 1
        k = max(3, k)
        bg = median_filter(filled, size=k, mode="nearest")
        res = np.maximum(0.0, filled - bg)
        scores.append(_robust_unit(res, valid, 4.0, 96.0))
    return np.mean(np.stack(scores, axis=0), axis=0)


def _btd_band_score(
    bt7: np.ndarray,
    bt14: np.ndarray,
    valid: np.ndarray,
    *,
    low_pct: float,
    high_pct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score 0–1 alto quando ``BTD`` está dentro de [low, high] (banda esperada
    para focos pequenos diurnos). Acima/abaixo → 0.
    Devolve também a máscara `in_band`.
    """
    btd = bt7 - bt14
    finite = np.isfinite(btd) & valid
    if not finite.any():
        return np.zeros_like(btd, dtype=np.float64), finite
    lo = float(np.percentile(btd[finite], low_pct))
    hi = float(np.percentile(btd[finite], high_pct))
    width = max(hi - lo, 1e-6)
    centre = 0.5 * (lo + hi)
    in_band = (btd >= lo) & (btd <= hi) & finite
    # Score: triangular centrado, máximo no centro da banda
    s = np.where(in_band, 1.0 - 2.0 * np.abs(btd - centre) / width, 0.0)
    return np.clip(s, 0.0, 1.0), in_band


def _glint_mask(
    bt7: np.ndarray,
    valid: np.ndarray,
    *,
    pct: float,
) -> np.ndarray:
    """Pixels acima do percentil ``pct`` em BT7 sobre a cena → provável glint."""
    finite = np.isfinite(bt7) & valid
    if not finite.any():
        return np.zeros_like(bt7, dtype=bool)
    thr = float(np.percentile(bt7[finite], pct))
    return finite & (bt7 >= thr)


def detect_dtec(
    hourly: List[Dict[int, np.ndarray]],
    valid_bins: np.ndarray,
    *,
    cfg: Optional[DTECConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Devolve ``(pred_mask, risk_field)``.

    ``hourly``: lista por hora, cada item com canais ``{7, 13, 14}`` (H, W).
    ``valid_bins``: máscara de células com observação coerente em todas as horas.
    """
    c = cfg or DTECConfig()
    if not hourly:
        return np.zeros_like(valid_bins, dtype=bool), np.zeros_like(valid_bins, dtype=np.float64)

    shape = hourly[0][13].shape
    h, w = shape

    anom_per_hour: List[np.ndarray] = []
    btd_score_per_hour: List[np.ndarray] = []
    in_band_per_hour: List[np.ndarray] = []
    glint_per_hour: List[np.ndarray] = []

    for slot in hourly:
        bt13 = slot[13]
        bt7 = slot.get(7)
        bt14 = slot.get(14)
        v_t = np.isfinite(bt13) & valid_bins
        anom = _fine_anomaly(bt13, v_t, c.fine_median_sizes)
        anom_per_hour.append(anom)

        if bt7 is not None and bt14 is not None:
            btd_s, in_band = _btd_band_score(
                bt7,
                bt14,
                v_t,
                low_pct=c.btd_low_percentile,
                high_pct=c.btd_high_percentile,
            )
            btd_score_per_hour.append(btd_s)
            in_band_per_hour.append(in_band)
            glint_per_hour.append(_glint_mask(bt7, v_t, pct=c.bt7_glint_percentile))
        else:
            btd_score_per_hour.append(np.zeros(shape, dtype=np.float64))
            in_band_per_hour.append(np.ones(shape, dtype=bool) & v_t)
            glint_per_hour.append(np.zeros(shape, dtype=bool))

    A = np.stack(anom_per_hour, axis=0)        # (T, H, W)
    B = np.stack(btd_score_per_hour, axis=0)
    IB = np.stack(in_band_per_hour, axis=0)
    G = np.stack(glint_per_hour, axis=0)

    nh = A.shape[0]

    # Persistência: fracção de horas onde a célula tem anomalia positiva
    #               e está dentro da banda BTD e não tem glint
    active = (A > 0.05) & IB & ~G
    persist_frac = active.mean(axis=0)
    n_active = active.sum(axis=0)

    # Sinal por célula: média do score sobre horas em que está activa (evita zeros mascararem)
    with np.errstate(invalid="ignore", divide="ignore"):
        anom_mean_when_active = np.where(
            n_active > 0,
            (A * active).sum(axis=0) / np.maximum(n_active, 1),
            0.0,
        )
        btd_mean_when_active = np.where(
            n_active > 0,
            (B * active).sum(axis=0) / np.maximum(n_active, 1),
            0.0,
        )

    # Normalizações robustas só nas células válidas
    q_lo, q_hi = c.robust_norm_quantiles
    anom_n = _robust_unit(anom_mean_when_active, valid_bins, q_lo, q_hi)
    btd_n = _robust_unit(btd_mean_when_active, valid_bins, q_lo, q_hi)

    # Risco combinado
    wA, wB, wP = c.weight_anomaly, c.weight_btd_band, c.weight_persistence
    wsum = wA + wB + wP
    risk = (wA * anom_n + wB * btd_n + wP * persist_frac) / max(wsum, 1e-9)

    # Anular glint e fora-da-banda em todas as horas
    never_in_band = ~IB.any(axis=0)
    always_glint = G.all(axis=0)
    risk = np.where(never_in_band | always_glint, 0.0, risk)

    # Persistência mínima
    if nh >= c.min_active_hours and c.min_active_hours > 1:
        risk = np.where(n_active >= c.min_active_hours, risk, 0.0)

    # Limiar global
    pool = risk[valid_bins & np.isfinite(risk) & (risk > 0)]
    if pool.size < 32:
        pred = np.zeros(shape, dtype=bool)
    else:
        thr = float(np.percentile(pool, c.risk_top_percentile))
        pred = (risk >= thr) & valid_bins

    # Morfologia leve
    if c.weak_open_iterations > 0 and pred.any():
        struct = generate_binary_structure(2, 1)
        for _ in range(c.weak_open_iterations):
            pred = binary_opening(pred, structure=struct)

    # Filtros de tamanho de componente
    if (c.min_component_cells > 1 or c.max_component_cells > 0) and pred.any():
        lab, n = label(pred)
        keep = np.zeros_like(pred, dtype=bool)
        for k in range(1, n + 1):
            m = lab == k
            sz = int(m.sum())
            if sz < c.min_component_cells:
                continue
            if c.max_component_cells > 0 and sz > c.max_component_cells:
                continue
            keep |= m
        pred = keep

    return pred & valid_bins, risk
