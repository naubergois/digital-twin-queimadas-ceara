"""
ST-HyperNet (MVP): campo de fundo contextual + ruptura espectro-temporal para fogo.

Self-supervised Spatio-Temporal Hyperbolic-inspired Background Field for
Geostationary Fire Anomaly — sem rótulos.

Este MVP implementa a mecânica central da proposta:
  - **Hiper-rede** condicionada em (lat, lon, DOY, hora) → vetor de contexto.
  - **Encoder** Conv3D leve sobre bloco (T_win × H_patch × W_patch) × bandas.
  - **Embedding tipo Poincaré** (mapa explícito do espaço tangente na origem, sem geoopt).
  - **Decoder** MLP reconstrói telha 3×3 de B7/B14 no último instante do bloco.
  - **Perdas**: L1 na reconstrução + TV espacial na telha prevista.
  - **Score de fogo**: resíduo L1 normalizado + coerência temporal B7↔B14 (produto de
    variações temporais normalizadas), destacando frentes espectrais coerentes.

Entrada: cubo GOES-proxy (BT7, BT14, ΔBT) como em `pyro_caatinga` / `goes_unsupervised_twin`.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.ceara_config import CEARA_BBOX

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_ERR = e
else:
    _TORCH_ERR = None


def _require_torch():
    if torch is None:
        raise RuntimeError(
            "ST-HyperNet requer PyTorch. Instale com: pip install torch"
        ) from _TORCH_ERR


@dataclass
class STHyperNetConfig:
    patch_t: int = 5
    patch_hw: int = 7
    d_enc: int = 32
    d_ctx: int = 32
    d_hyper_hidden: int = 64
    tile_hw: int = 3
    lambda_tv: float = 0.15
    lambda_temp_score: float = 0.32
    score_delta_weight: float = 0.38
    score_residual_weight: float = 0.42
    epochs: int = 12
    lr: float = 1e-3
    batch_size: int = 64
    max_patches_per_epoch: int = 2048
    infer_batch: int = 512
    inference_stride: int = 1
    poincare_c: float = 1.0
    device: str = "cpu"
    seed: int = 42
    grid_resolution: float = 0.5
    frame_minutes: int = 60
    max_days_history: int = 0


def _grid_norm_ij(i: int, j: int, n_lat: int, n_lon: int) -> Tuple[float, float]:
    return float(i) / max(n_lat - 1, 1), float(j) / max(n_lon - 1, 1)


def _lat_lon_from_ij(i: int, j: int, n_lat: int, n_lon: int) -> Tuple[float, float]:
    lat = CEARA_BBOX["min_lat"] + (i + 0.5) * (CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]) / max(n_lat, 1)
    lon = CEARA_BBOX["min_lon"] + (j + 0.5) * (CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]) / max(n_lon, 1)
    return float(lat), float(lon)


def exp_map_poincare_origin(v: "torch.Tensor", c: float = 1.0) -> "torch.Tensor":
    """Mapa exponencial aproximado na origem da bola de Poincaré (||h|| < 1)."""
    _require_torch()
    norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    # tanh comprime raio; escala estável para c≈1
    r = torch.tanh(math.sqrt(c) * norm) / (math.sqrt(c) * norm)
    return torch.clamp(r * v, min=-0.98, max=0.98)


class STHyperNetMVP(nn.Module):
    """Hiper-rede + encoder Conv3D + decoder telha (2 × h × w)."""

    def __init__(self, cfg: STHyperNetConfig):
        _require_torch()
        super().__init__()
        self.cfg = cfg
        ph = cfg.tile_hw
        in_ch = 3
        self.encoder = nn.Sequential(
            nn.Conv3d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
        )
        self.enc_lin = nn.Linear(32, cfg.d_enc)
        self.hyper = nn.Sequential(
            nn.Linear(8, cfg.d_hyper_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.d_hyper_hidden, cfg.d_ctx),
        )
        dec_in = cfg.d_enc + cfg.d_ctx
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * ph * ph),
        )
        self.ph = ph

    def forward(self, patch: "torch.Tensor", ctx: "torch.Tensor") -> "torch.Tensor":
        """
        patch: (B, 3, T, H, W) normalizado
        ctx: (B, 8) lat_n, lon_n, sin/cos DOY, sin/cos hod, sin/cos dow
        """
        z = self.encoder(patch).flatten(1)
        e0 = self.enc_lin(z)
        e = exp_map_poincare_origin(e0, c=self.cfg.poincare_c)
        h = self.hyper(ctx)
        y = self.decoder(torch.cat([e, h], dim=-1))
        return y.view(-1, 2, self.ph, self.ph)


def _ctx_vector(ts: pd.Timestamp, i: int, j: int, n_lat: int, n_lon: int) -> List[float]:
    lat, lon = _lat_lon_from_ij(i, j, n_lat, n_lon)
    lat_n = (lat - CEARA_BBOX["min_lat"]) / (CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"])
    lon_n = (lon - CEARA_BBOX["min_lon"]) / (CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"])
    doy = ts.dayofyear
    hod = ts.hour + ts.minute / 60.0
    return [
        float(lat_n),
        float(lon_n),
        math.sin(2 * math.pi * doy / 366.0),
        math.cos(2 * math.pi * doy / 366.0),
        math.sin(2 * math.pi * hod / 24.0),
        math.cos(2 * math.pi * hod / 24.0),
        math.sin(2 * math.pi * ts.dayofweek / 7.0),
        math.cos(2 * math.pi * ts.dayofweek / 7.0),
    ]


def _normalize_bands(bt7: np.ndarray, bt14: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
    m7, s7 = float(bt7.mean()), float(bt7.std()) + 1e-6
    m14, s14 = float(bt14.mean()), float(bt14.std()) + 1e-6
    return (bt7 - m7) / s7, (bt14 - m14) / s14, m7, s7, m14, s14


def _tv_spatial_tile(pred: "torch.Tensor") -> "torch.Tensor":
    """pred (B,2,h,w)"""
    dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).abs().mean()
    dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).abs().mean()
    return dx + dy


def build_cube_from_df(df: pd.DataFrame, cfg: STHyperNetConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    from src.pyro_caatinga import PyroCaatingaConfig, build_goes_proxy_cube

    dfx = df.copy()
    for col in ["datetime", "data_hora", "data_hora_gmt", "acq_date"]:
        if col in dfx.columns:
            dfx["datetime"] = pd.to_datetime(dfx[col], errors="coerce")
            break
    if "datetime" not in dfx.columns:
        raise ValueError("CSV precisa de datetime.")
    if "lat" not in dfx.columns or "lon" not in dfx.columns:
        raise ValueError("CSV precisa de lat/lon.")
    pyro_cfg = PyroCaatingaConfig(
        grid_resolution=cfg.grid_resolution,
        frame_minutes=cfg.frame_minutes,
        max_days_history=cfg.max_days_history,
    )
    ds = build_goes_proxy_cube(dfx.dropna(subset=["datetime", "lat", "lon"]), pyro_cfg)
    times = pd.DatetimeIndex(ds["time"].values)
    bt7 = np.asarray(ds["bt7"].values, dtype=np.float32)
    bt14 = np.asarray(ds["bt14"].values, dtype=np.float32)
    delta = np.asarray(ds["delta_bt"].values, dtype=np.float32)
    return bt7, bt14, delta, times


def _slice_target_tile(
    bt7n: np.ndarray,
    bt14n: np.ndarray,
    t_end: int,
    i: int,
    j: int,
    ph: int,
) -> np.ndarray:
    h2 = ph // 2
    t0 = t_end
    return np.stack(
        [
            bt7n[t0, i - h2 : i + h2 + 1, j - h2 : j + h2 + 1],
            bt14n[t0, i - h2 : i + h2 + 1, j - h2 : j + h2 + 1],
        ],
        axis=0,
    ).astype(np.float32)


def train_st_hypernet(
    bt7: np.ndarray,
    bt14: np.ndarray,
    delta: np.ndarray,
    times: pd.DatetimeIndex,
    cfg: STHyperNetConfig,
) -> Tuple[STHyperNetMVP, Dict[str, float]]:
    _require_torch()
    torch.manual_seed(cfg.seed)
    T, H, W = bt7.shape
    pw = cfg.patch_t
    hw = cfg.patch_hw
    hhw = hw // 2
    ph = cfg.tile_hw

    bt7n, bt14n, m7, s7, m14, s14 = _normalize_bands(bt7, bt14)

    model = STHyperNetMVP(cfg).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    indices: List[Tuple[int, int, int]] = []
    for t in range(pw - 1, T):
        for i in range(hhw, H - hhw):
            for j in range(hhw, W - hhw):
                indices.append((t, i, j))
    rng = np.random.default_rng(cfg.seed)
    if len(indices) == 0:
        raise ValueError("Cubo pequeno demais para ST-HyperNet (aumente dados ou reduza patch_t/patch_hw).")

    losses_ep: List[float] = []
    for ep in range(cfg.epochs):
        model.train()
        rng.shuffle(indices)
        take = indices[: cfg.max_patches_per_epoch]
        loss_acc = 0.0
        n_steps = 0
        for s in range(0, len(take), cfg.batch_size):
            batch_idx = take[s : s + cfg.batch_size]
            if not batch_idx:
                break
            patches = []
            ctxs = []
            targets = []
            for (t, i, j) in batch_idx:
                t0 = t - (pw - 1)
                p = np.stack(
                    [
                        bt7n[t0 : t + 1, i - hhw : i + hhw + 1, j - hhw : j + hhw + 1],
                        bt14n[t0 : t + 1, i - hhw : i + hhw + 1, j - hhw : j + hhw + 1],
                        delta[t0 : t + 1, i - hhw : i + hhw + 1, j - hhw : j + hhw + 1],
                    ],
                    axis=0,
                )
                patches.append(p)
                ts = pd.Timestamp(times[t])
                ctxs.append(_ctx_vector(ts, i, j, H, W))
                targets.append(_slice_target_tile(bt7n, bt14n, t, i, j, ph))
            x = torch.from_numpy(np.stack(patches, axis=0)).to(cfg.device)
            c = torch.tensor(ctxs, dtype=torch.float32, device=cfg.device)
            y = torch.from_numpy(np.stack(targets, axis=0)).to(cfg.device)
            pred = model(x, c)
            l1 = F.l1_loss(pred, y)
            ltv = _tv_spatial_tile(pred)
            loss = l1 + cfg.lambda_tv * ltv
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc += float(loss.detach().cpu())
            n_steps += 1
        losses_ep.append(loss_acc / max(n_steps, 1))

    meta = {
        "m7": m7,
        "s7": s7,
        "m14": m14,
        "s14": s14,
        "loss_last": float(losses_ep[-1]) if losses_ep else 0.0,
        "loss_curve": [float(x) for x in losses_ep],
    }
    return model, meta


@torch.no_grad()
def infer_fire_score_cube(
    model: STHyperNetMVP,
    meta: Dict[str, float],
    bt7: np.ndarray,
    bt14: np.ndarray,
    delta: np.ndarray,
    times: pd.DatetimeIndex,
    cfg: STHyperNetConfig,
) -> np.ndarray:
    """Retorna cubo (T,H,W) score em [0,1] (maior = mais provável fogo / ruptura)."""
    _require_torch()
    model.eval()
    T, H, W = bt7.shape
    ph = cfg.tile_hw
    h2 = ph // 2
    pw = cfg.patch_t
    hw = cfg.patch_hw
    hhw = hw // 2
    m7, s7, m14, s14 = meta["m7"], meta["s7"], meta["m14"], meta["s14"]
    bt7n = (bt7 - m7) / s7
    bt14n = (bt14 - m14) / s14

    residual = np.zeros((T, H, W), dtype=np.float32)
    counts = np.zeros((T, H, W), dtype=np.float32)
    coh = np.zeros((T, H, W), dtype=np.float32)

    # coerência B7–B14 ao longo do tempo (|dt b7|*|dt b14|) normalizada
    d7 = np.zeros_like(bt7n, dtype=np.float32)
    d14 = np.zeros_like(bt14n, dtype=np.float32)
    d7[1:] = np.abs(bt7n[1:] - bt7n[:-1])
    d14[1:] = np.abs(bt14n[1:] - bt14n[:-1])
    sig = np.std(d7) * np.std(d14) + 1e-6
    coh_t = (d7 * d14) / sig
    coh[:] = coh_t

    batch_tij: List[Tuple[int, int, int]] = []
    strd = max(1, int(cfg.inference_stride))

    def flush_batch():
        nonlocal batch_tij, residual, counts
        if not batch_tij:
            return
        patches = []
        ctxs = []
        tij = list(batch_tij)
        batch_tij = []
        for (t, i, j) in tij:
            t0 = t - (pw - 1)
            p = np.stack(
                [
                    bt7n[t0 : t + 1, i - hhw : i + hhw + 1, j - hhw : j + hhw + 1],
                    bt14n[t0 : t + 1, i - hhw : i + hhw + 1, j - hhw : j + hhw + 1],
                    delta[t0 : t + 1, i - hhw : i + hhw + 1, j - hhw : j + hhw + 1],
                ],
                axis=0,
            )
            patches.append(p)
            ts = pd.Timestamp(times[t])
            ctxs.append(_ctx_vector(ts, i, j, H, W))
        x = torch.from_numpy(np.stack(patches, axis=0)).to(cfg.device)
        c = torch.tensor(ctxs, dtype=torch.float32, device=cfg.device)
        pred = model(x, c).cpu().numpy()
        tgt = np.stack([_slice_target_tile(bt7n, bt14n, t, i, j, ph) for (t, i, j) in tij], axis=0)
        err_pix = np.abs(pred - tgt).mean(axis=1)
        for k, (t, i, j) in enumerate(tij):
            sl_i = slice(i - h2, i + h2 + 1)
            sl_j = slice(j - h2, j + h2 + 1)
            residual[t, sl_i, sl_j] += err_pix[k].astype(np.float32)
            counts[t, sl_i, sl_j] += 1.0

    for t in range(pw - 1, T):
        for i in range(hhw, H - hhw, strd):
            for j in range(hhw, W - hhw, strd):
                batch_tij.append((t, i, j))
                if len(batch_tij) >= cfg.infer_batch:
                    flush_batch()
    flush_batch()

    res_mean = np.divide(residual, np.maximum(counts, 1e-6))
    mx = float(res_mean.max()) + 1e-9
    res_n = (res_mean / mx).astype(np.float32)

    # Sinal físico do proxy: |ΔBT| alto (fogo) mesmo quando o autoencoder já reconstrói bem.
    dabs = np.abs(np.asarray(delta, dtype=np.float32))
    pr = float(np.percentile(dabs, 93.0))
    delta_n = np.clip(dabs / (pr + 1e-6), 0.0, 1.0).astype(np.float32)

    score = (
        float(cfg.score_residual_weight) * res_n
        + float(cfg.lambda_temp_score) * coh.astype(np.float32)
        + float(cfg.score_delta_weight) * delta_n
    )
    mx2 = float(score.max()) + 1e-9
    return (score / mx2).astype(np.float32)


def run_st_hypernet_pipeline(
    df: pd.DataFrame,
    cfg: Optional[STHyperNetConfig] = None,
) -> Dict[str, Any]:
    """Treina ST-HyperNet e devolve cubo de scores + metadados."""
    cfg = cfg or STHyperNetConfig()
    bt7, bt14, delta, times = build_cube_from_df(df, cfg)
    model, meta = train_st_hypernet(bt7, bt14, delta, times, cfg)
    cube = infer_fire_score_cube(model, meta, bt7, bt14, delta, times, cfg)
    agg = np.max(cube, axis=0)
    return {
        "technique": "ST-HyperNet MVP (hypernet + Poincaré-origem + Conv3D + telha L1/TV + score resíduo+coh)",
        "cube_shape": list(cube.shape),
        "max_score": float(agg.max()),
        "mean_score": float(agg.mean()),
        "config": asdict(cfg),
        "train_meta": {k: v for k, v in meta.items() if k != "loss_curve"},
        "_score_cube": cube,
        "_agg_grid": agg,
        "_model_state": model.state_dict(),
        "_meta": meta,
        "_times": times,
    }


def save_st_hypernet_artifact(result: Dict[str, Any], path: str | Path) -> None:
    """Salva state_dict + meta mínimos para inferência futura."""
    _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": result["_model_state"],
            "meta": result["_meta"],
            "config": result["config"],
        },
        path,
    )


def write_st_hypernet_best_params_json(
    result: Dict[str, Any],
    path: str | Path,
    extras: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Grava JSON reprodutível: config de treino, metadados de normalização, curva de loss
    e campos extras (ex.: métricas agregadas da comparação por dia).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    public = export_public_st_report(result)
    meta = dict(result.get("_meta") or {})
    loss_curve = meta.get("loss_curve")
    if isinstance(loss_curve, list):
        meta["loss_curve"] = [float(x) for x in loss_curve]
    record: Dict[str, Any] = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "artifact_note": "Use save_st_hypernet_artifact (.pt) + este JSON para reproduzir inferência.",
        "st_hypernet_public": public,
        "normalization_and_train_meta": {
            k: meta[k]
            for k in ("m7", "s7", "m14", "s14", "loss_last", "loss_curve")
            if k in meta
        },
        "extras": extras or {},
    }
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_st_hypernet_artifact(path: str | Path, cfg: Optional[STHyperNetConfig] = None) -> Tuple[STHyperNetMVP, Dict[str, float], STHyperNetConfig]:
    _require_torch()
    blob = torch.load(path, map_location="cpu")
    valid = {f.name for f in fields(STHyperNetConfig)}
    cfg = cfg or STHyperNetConfig(**{k: v for k, v in blob.get("config", {}).items() if k in valid})
    model = STHyperNetMVP(cfg)
    model.load_state_dict(blob["model"])
    return model, blob["meta"], cfg


def export_public_st_report(result: Dict[str, Any]) -> Dict[str, Any]:
    """Remove tensores / state para JSON."""
    out = {k: v for k, v in result.items() if not str(k).startswith("_")}
    tm = out.get("train_meta")
    if isinstance(tm, dict) and "loss_curve" in tm:
        out["train_meta"] = {k: v for k, v in tm.items() if k != "loss_curve"}
    return out


def merge_st_hypernet_into_digital_twin(twin: Any, result: Dict[str, Any], weight: float = 0.22) -> None:
    """Mescla grade agregada ST-HyperNet (max ao longo do tempo) no `risk_grid` do twin."""
    from src.goes_unsupervised_twin import resample_risk_to_shape

    risk = result.get("_agg_grid")
    if risk is None or getattr(twin, "risk_grid", None) is None:
        return
    w = float(np.clip(weight, 0.0, 1.0))
    up = resample_risk_to_shape(risk.astype(np.float32), int(twin.n_lat), int(twin.n_lon))
    twin.risk_grid = np.clip((1.0 - w) * twin.risk_grid + w * np.clip(up, 0.0, 1.0), 0.0, 1.0).astype(np.float32)
