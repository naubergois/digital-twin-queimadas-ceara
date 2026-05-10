"""Máscara de visualização para mapas (evita grade toda como prevista)."""

import numpy as np

from src.compare_ceara_maps import (
    build_ceara_folium_real_pred_goes,
    display_positive_mask,
    map_points_to_json_fields,
    pred_points_from_display_mask,
)


def test_display_positive_mask_caps_saturated_grid():
    g = np.full((3, 3), 0.95, dtype=np.float32)
    mask, thr, mode = display_positive_mask(g, 0.2, max_positive_fraction=0.2)
    assert mode == "adapted_topk"
    assert float(mask.sum()) == 1.0  # floor(0.2 * 9) = 1 célula
    assert thr >= 0.2
    pts = pred_points_from_display_mask(g, 3, 3, mask)
    assert len(pts) == int(mask.sum())


def test_display_positive_mask_base_when_sparse():
    g = np.zeros((4, 4), dtype=np.float32)
    g[1, 1] = 0.9
    mask, thr, mode = display_positive_mask(g, 0.35, max_positive_fraction=0.2)
    assert mode == "base"
    assert thr == 0.35
    assert mask[1, 1] == 1.0 and float(mask.sum()) == 1.0


def test_map_points_to_json_fields_roundtrip():
    real = [(-5.1, -39.2), (-5.2, -39.3)]
    pred = [(-5.11, -39.21, 0.7), (-5.25, -39.31, 0.4)]
    d = map_points_to_json_fields(real, pred)
    assert d["map_real_latlon"] == [[-5.1, -39.2], [-5.2, -39.3]]
    assert len(d["map_pred_latlon_score"]) == 2
    assert d["map_pred_latlon_score"][0][2] == 0.7


def test_build_ceara_folium_real_pred_goes_smoke():
    m = build_ceara_folium_real_pred_goes(
        real_latlon=[[-5.1, -39.2]],
        pred_latlon_score=[[-5.15, -39.25, 0.5]],
        goes_latlon=[(-5.12, -39.22)],
        pred_layer_label="Previsto (teste)",
    )
    html = m.get_root().render()
    assert "folium" in html.lower() or "leaflet" in html.lower()
