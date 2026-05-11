"""
Busca de parâmetros DTEC: maximiza F1 event-centric (R configurável) em 2024-10-31.

Não usa rótulos para *treinar* — usa para *escolher hiper-parâmetros*. Em
implantação a calibração viria de validação cruzada espaço-temporal
(DTEC §6), mas para esta iteração de pesquisa basta uma busca em grade.
"""

from __future__ import annotations

import itertools
import json
from datetime import date
from pathlib import Path

import numpy as np

from config.ceara_config import CEARA_BBOX
from src.dtec_detector import DTECConfig, detect_dtec
from src.event_centric import (
    day_window_utc,
    evaluate_event_centric,
    evaluate_event_centric_multi_radius,
)
from src.unsupervised_fire_goes import (
    collect_hourly_band_grids,
    intersect_valid_bins_hourly,
    load_inpe_focos,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DAY = date(2024, 10, 31)
GRID_HW = (72, 72)


def main() -> None:
    inpe_csv = REPO_ROOT / "data" / "inpe_focos_ce" / "focos_ce_INPE_2024_2026.csv"
    raw_dir = REPO_ROOT / "data" / "goes16_raw"
    df = load_inpe_focos(inpe_csv)
    d0, d1 = day_window_utc(DAY.isoformat())

    hourly, _ = collect_hourly_band_grids(
        DAY,
        (16, 17, 18),
        (7, 13, 14),
        CEARA_BBOX,
        GRID_HW,
        raw_dir,
        skip_download=True,
        overwrite=False,
        use_dqf=True,
        show_progress=False,
    )
    valid_bins = intersect_valid_bins_hourly(hourly, [7, 13, 14])

    eval_radius = 5.0

    grid = {
        "fine_median_sizes": [(3, 5), (3, 5, 7), (5, 7)],
        "btd_low_percentile": [25.0, 35.0, 50.0],
        "btd_high_percentile": [80.0, 90.0, 95.0],
        "risk_top_percentile": [95.0, 97.0, 98.5, 99.2],
        "min_active_hours": [1, 2, 3],
        "weight_anomaly": [0.45, 0.55, 0.65],
        "weight_persistence": [0.15, 0.25, 0.35],
        "max_component_cells": [80, 200, 0],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"Configurações a testar: {len(combos)}")

    best_overall = []
    for combo in combos:
        kw = dict(zip(keys, combo))
        # weight_btd_band derivado dos outros
        kw["weight_btd_band"] = max(0.05, 1.0 - kw["weight_anomaly"] - kw["weight_persistence"])
        cfg = DTECConfig(**kw)
        pred, _ = detect_dtec(hourly, valid_bins, cfg=cfg)
        m = evaluate_event_centric(
            pred,
            df,
            CEARA_BBOX,
            GRID_HW,
            radius_km=eval_radius,
            day_utc=(d0, d1),
            valid_bins=valid_bins,
        )
        best_overall.append((m.f1, m.precision, m.recall, m.n_components, int(pred.sum()), kw))

    best_overall.sort(key=lambda t: (t[0], t[1]), reverse=True)
    print(f"\nTop 12 configurações DTEC por F1 (R={eval_radius} km):")
    print(f"{'F1':>6} {'P':>6} {'R':>6} {'#cmp':>5} {'#cel':>5}  parâmetros")
    for f1, p, r, ncmp, ncel, kw in best_overall[:12]:
        kshort = (
            f"fine={kw['fine_median_sizes']} btdLo={kw['btd_low_percentile']} "
            f"btdHi={kw['btd_high_percentile']} top={kw['risk_top_percentile']} "
            f"hAct={kw['min_active_hours']} wA={kw['weight_anomaly']:.2f} "
            f"wP={kw['weight_persistence']:.2f} maxCC={kw['max_component_cells']}"
        )
        print(f"{f1:>6.3f} {p:>6.3f} {r:>6.3f} {ncmp:>5d} {ncel:>5d}  {kshort}")

    # Salvar tudo para o JSON e o melhor para inspecção posterior
    out = REPO_ROOT / "data" / "goes16_eval" / "dtec_grid_search_2024-10-31.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "f1": f1,
            "precision": p,
            "recall": r,
            "n_components": ncmp,
            "n_pred_cells": ncel,
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
        }
        for f1, p, r, ncmp, ncel, kw in best_overall
    ]
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nGravado: {out}")

    # Curva F1(R) para o melhor
    best_kw = best_overall[0][5]
    best_cfg = DTECConfig(**best_kw)
    pred, _ = detect_dtec(hourly, valid_bins, cfg=best_cfg)
    curve = evaluate_event_centric_multi_radius(
        pred,
        df,
        CEARA_BBOX,
        GRID_HW,
        radii_km=(1.5, 3.0, 5.0, 8.0, 12.0),
        day_utc=(d0, d1),
        valid_bins=valid_bins,
    )
    print("\nCurva F1(R) do melhor:")
    for k, v in curve.items():
        print(f"  {k}: P={v['ec_precision']:.3f} R={v['ec_recall']:.3f} F1={v['ec_f1']:.3f}  comps={int(v['ec_n_components'])} focos={int(v['ec_n_focos'])}")


if __name__ == "__main__":
    main()
