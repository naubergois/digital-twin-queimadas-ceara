"""
Treino e validação DTEC **multi-dia** (DTEC §6).

Junta features de muitos dias (cada um produz uma cena GOES H×W) num
único conjunto de treino. Mantém o gêmeo digital no centro: por dia,
constrói-se ``twin_risk`` + outras 5 features, depois empilha-se com
máscara de verdade (focos INPE dilatados 1 célula) e treina-se um único
classificador HGB.

Validação:

- **Leave-one-day-out** (LODO) — para cada dia ``d_test``, treina nos
  outros. Mais honesto que a CV bloqueada num só dia.
- **Bloco temporal** com buffer — para datasets maiores.

A pipeline aceita lista de dias e a localização local dos NetCDFs;
quando o NetCDF de um dia não existe, esse dia é silenciosamente saltado
(o `download_goes_multiday.py` resolve a obtenção).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure

from config.ceara_config import CEARA_BBOX
from src.dtec_supervised import (
    DTECSupervisedConfig,
    DTECSupervisedModel,
    build_features,
    stack_features,
    train_dtec_supervised,
)
from src.event_centric import (
    day_window_utc,
    evaluate_event_centric,
)
from src.unsupervised_fire_goes import (
    collect_hourly_band_grids,
    find_local_goes_nc,
    intersect_valid_bins_hourly,
    truth_presence_grid,
)


@dataclass
class DayData:
    day: date
    feats: Dict[str, np.ndarray]
    valid_bins: np.ndarray
    truth_dil: np.ndarray
    df_day: pd.DataFrame


@dataclass
class MultiDayConfig:
    grid_hw: Tuple[int, int] = (144, 144)
    hours: Tuple[int, ...] = (16, 17, 18)
    channels: Tuple[int, ...] = (7, 13, 14)
    dtec_cfg: DTECSupervisedConfig = field(default_factory=lambda: DTECSupervisedConfig(
        classifier="hgb", smooth_sigma=1.2, nms_radius=1, nms_min_prob=0.5,
    ))


def _day_has_all_files(raw_dir: Path, day: date, hours, channels) -> bool:
    for h in hours:
        for ch in channels:
            try:
                find_local_goes_nc(raw_dir, day, ch, hour=int(h))
            except FileNotFoundError:
                return False
    return True


def load_day_data(
    day: date,
    df_inpe: pd.DataFrame,
    raw_dir: Path,
    cfg: Optional[MultiDayConfig] = None,
) -> Optional[DayData]:
    """Carrega features + verdade dilatada de um único dia; ``None`` se faltam NetCDFs."""
    c = cfg or MultiDayConfig()
    if not _day_has_all_files(raw_dir, day, c.hours, c.channels):
        return None
    hourly, _ = collect_hourly_band_grids(
        day, c.hours, c.channels, CEARA_BBOX, c.grid_hw, raw_dir,
        skip_download=True, overwrite=False, use_dqf=True, show_progress=False,
    )
    valid_bins = intersect_valid_bins_hourly(hourly, list(c.channels))
    feats = build_features(hourly, valid_bins)

    d0, d1 = day_window_utc(day.isoformat())
    df_day = df_inpe.loc[(df_inpe["datetime"] >= d0) & (df_inpe["datetime"] < d1)].reset_index(drop=True)
    truth_raw = truth_presence_grid(df_day, CEARA_BBOX, c.grid_hw)
    truth_dil = binary_dilation(truth_raw, structure=np.ones((3, 3), dtype=bool))
    return DayData(day=day, feats=feats, valid_bins=valid_bins, truth_dil=truth_dil, df_day=df_day)


def stack_days(days_data: Sequence[DayData], cfg: DTECSupervisedConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Empilha (X, y) de muitos dias para um único treino."""
    feat_names = cfg.feature_names
    Xs, ys = [], []
    for dd in days_data:
        x, _ = stack_features({n: dd.feats[n] for n in feat_names})
        y = dd.truth_dil.ravel().astype(np.int8)
        v = dd.valid_bins.ravel()
        Xs.append(x[v])
        ys.append(y[v])
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)


def train_multiday(
    days_data: Sequence[DayData],
    *,
    cfg: Optional[MultiDayConfig] = None,
) -> DTECSupervisedModel:
    """
    Treina HGB sobre features de múltiplos dias de uma vez. Cabeça única,
    aprende um perfil de fogo agregado em vez de decorar um dia específico.
    """
    c = cfg or MultiDayConfig()
    # Usa o primeiro dia como "âncora" da shape, depois substitui pelo conjunto agregado.
    from sklearn.preprocessing import StandardScaler

    X, y = stack_days(days_data, c.dtec_cfg)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    if c.dtec_cfg.classifier == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(
            max_depth=c.dtec_cfg.hgb_max_depth,
            learning_rate=c.dtec_cfg.hgb_learning_rate,
            max_iter=c.dtec_cfg.hgb_max_iter,
            class_weight=c.dtec_cfg.class_weight,
            random_state=42,
        )
    else:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            C=c.dtec_cfg.C,
            class_weight=c.dtec_cfg.class_weight,
            max_iter=c.dtec_cfg.max_iter,
            solver="liblinear",
        )
    clf.fit(Xs, y)
    return DTECSupervisedModel(
        scaler=scaler,
        clf=clf,
        feature_names=c.dtec_cfg.feature_names,
        threshold=c.dtec_cfg.threshold,
        smooth_sigma=c.dtec_cfg.smooth_sigma,
        nms_radius=c.dtec_cfg.nms_radius,
        nms_min_prob=c.dtec_cfg.nms_min_prob,
    )


def leave_one_day_out(
    days_data: Sequence[DayData],
    *,
    cfg: Optional[MultiDayConfig] = None,
    radius_km: float = 10.0,
    nms_threshold: float = 0.7258,
    dilate_iters: int = 2,
) -> List[Dict]:
    """
    Para cada dia ``d``, treina nos outros e avalia event-centric em ``d``.
    Métrica honesta multi-dia.
    """
    c = cfg or MultiDayConfig()
    rows = []
    struct = generate_binary_structure(2, 2)
    for i, dd_test in enumerate(days_data):
        train = [dd for j, dd in enumerate(days_data) if j != i]
        if not train:
            continue
        model = train_multiday(train, cfg=c)
        peaks = model.predict_nms(
            dd_test.feats, dd_test.valid_bins,
            threshold=nms_threshold, nms_radius=1, smooth_sigma=1.2,
        )
        pred = peaks.copy()
        for _ in range(int(dilate_iters)):
            pred = binary_dilation(pred, structure=struct)
        pred &= dd_test.valid_bins

        m = evaluate_event_centric(
            pred, dd_test.df_day, CEARA_BBOX, c.grid_hw,
            radius_km=radius_km, valid_bins=dd_test.valid_bins,
        )
        rows.append({
            "day": dd_test.day.isoformat(),
            "n_focos": int(m.n_focos),
            "n_pred": int(pred.sum()),
            "f1": float(m.f1),
            "precision": float(m.precision),
            "recall": float(m.recall),
            "tp_recall": int(m.tp_for_recall),
        })
    return rows
