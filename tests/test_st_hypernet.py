"""Testes ST-HyperNet (requer PyTorch)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.st_hypernet import (
    STHyperNetConfig,
    STHyperNetMVP,
    exp_map_poincare_origin,
    infer_fire_score_cube,
    train_st_hypernet,
)


def test_poincare_map_bounded():
    v = torch.randn(4, 16) * 3.0
    h = exp_map_poincare_origin(v, c=1.0)
    assert float(h.norm(dim=-1).max()) <= 1.0 + 1e-5


def test_st_hypernet_forward():
    cfg = STHyperNetConfig(tile_hw=3, patch_t=3, patch_hw=5, d_enc=16, d_ctx=16, d_hyper_hidden=32)
    m = STHyperNetMVP(cfg)
    x = torch.randn(2, 3, cfg.patch_t, cfg.patch_hw, cfg.patch_hw)
    c = torch.randn(2, 8)
    y = m(x, c)
    assert y.shape == (2, 2, cfg.tile_hw, cfg.tile_hw)


def test_train_and_infer_smoke():
    T, H, W = 24, 12, 12
    rng = np.random.default_rng(0)
    bt7 = (47.0 + rng.standard_normal((T, H, W))).astype(np.float32)
    bt14 = (41.0 + 0.5 * rng.standard_normal((T, H, W))).astype(np.float32)
    delta = (bt7 - bt14).astype(np.float32)
    times = pd.date_range("2024-06-01", periods=T, freq="h")
    cfg = STHyperNetConfig(
        patch_t=4,
        patch_hw=5,
        tile_hw=3,
        epochs=2,
        max_patches_per_epoch=128,
        batch_size=32,
        infer_batch=64,
        inference_stride=2,
        lambda_tv=0.05,
    )
    model, meta = train_st_hypernet(bt7, bt14, delta, times, cfg)
    cube = infer_fire_score_cube(model, meta, bt7, bt14, delta, times, cfg)
    assert cube.shape == (T, H, W)
    assert float(cube.max()) <= 1.0 + 1e-5
