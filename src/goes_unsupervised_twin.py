"""
Predição não supervisionada a partir de imagens GOES-16 (proxy ABI).

O projeto ainda não ingere L1b ABI direto do bucket NOAA; aqui usamos o **cubo
imagem multi-banda** gerado a partir dos focos GOES-16 (mesma geometria do
PYRO-Caatinga: BT7, BT14, ΔBT e resíduos climatológicos), tratado como sequência
de *frames* de satélite sobre o Ceará.

Técnica principal: **Isolation Forest** em espaço de características radiométricas
(por pixel–tempo), sem rótulos. Fallback: anomalia robusta em ΔBT quando há
poucas amostras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config.ceara_config import CEARA_BBOX

try:
    import xarray as xr
except ImportError:  # pragma: no cover
    xr = None  # type: ignore


@dataclass
class GOESUnsupervisedConfig:
    """Parâmetros do cubo GOES-proxy e do detector de anomalias."""

    grid_resolution: float = 0.1
    frame_minutes: int = 5
    max_days_history: int = 14
    contamination: float = 0.04
    iforest_n_estimators: int = 120
    max_fit_samples: int = 150_000
    random_state: int = 42
    gaussian_smooth_sigma: float = 0.8


def _load_proxy_dataset_from_netcdf(path: str | Path) -> "xr.Dataset":
    if xr is None:
        raise RuntimeError("xarray é necessário para ler NetCDF GOES-proxy.")
    return xr.open_dataset(Path(path))


def _build_stack_from_xarray(ds: "xr.Dataset") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Retorna (bt7, bt14, delta_bt) e opcionalmente res_delta se existir."""
    bt7 = np.asarray(ds["bt7"].values, dtype=np.float32)
    bt14 = np.asarray(ds["bt14"].values, dtype=np.float32)
    delta = np.asarray(ds["delta_bt"].values, dtype=np.float32)
    res_d = None
    if "res_delta" in ds.data_vars:
        res_d = np.asarray(ds["res_delta"].values, dtype=np.float32)
    return bt7, bt14, delta, res_d


def _build_stack_from_dataframe(df: pd.DataFrame, cfg: GOESUnsupervisedConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    from src.pyro_caatinga import ClimatologyResidualFrontEnd, PyroCaatingaConfig, build_goes_proxy_cube

    pyro_cfg = PyroCaatingaConfig(
        grid_resolution=cfg.grid_resolution,
        frame_minutes=cfg.frame_minutes,
        max_days_history=cfg.max_days_history,
    )
    ds = build_goes_proxy_cube(df, pyro_cfg)
    times = pd.DatetimeIndex(ds["time"].values)
    front_b7 = ClimatologyResidualFrontEnd(ewma_lambda=pyro_cfg.ewma_lambda)
    front_b14 = ClimatologyResidualFrontEnd(ewma_lambda=pyro_cfg.ewma_lambda)
    res_b7, _ = front_b7.fit_transform(np.asarray(ds["bt7"].values, dtype=np.float32), times)
    res_b14, _ = front_b14.fit_transform(np.asarray(ds["bt14"].values, dtype=np.float32), times)
    res_delta = (res_b7 - res_b14).astype(np.float32)
    bt7 = np.asarray(ds["bt7"].values, dtype=np.float32)
    bt14 = np.asarray(ds["bt14"].values, dtype=np.float32)
    delta = np.asarray(ds["delta_bt"].values, dtype=np.float32)
    return bt7, bt14, delta, res_delta


def _anomaly_scores_isolation_forest(
    X: np.ndarray,
    cfg: GOESUnsupervisedConfig,
) -> np.ndarray:
    """X: (N, n_features) — retorna score de fogo (maior = mais anômalo / fogo)."""
    X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    n = X.shape[0]
    if n < 64:
        return np.zeros(n, dtype=np.float32)

    rng = np.random.default_rng(cfg.random_state)
    if n > cfg.max_fit_samples:
        idx = rng.choice(n, size=cfg.max_fit_samples, replace=False)
        X_fit = X[idx]
    else:
        X_fit = X

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_fit)
    clf = IsolationForest(
        n_estimators=cfg.iforest_n_estimators,
        contamination=float(np.clip(cfg.contamination, 0.001, 0.5)),
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    clf.fit(Xs)

    X_all = scaler.transform(X)
    raw = clf.score_samples(X_all)
    # Outliers têm score mais baixo → inverter para "probabilidade de anomalia"
    z = raw - float(np.median(raw))
    mad = float(np.median(np.abs(z - np.median(z))) + 1e-6)
    fire = np.clip(-(z / mad), 0.0, 8.0).astype(np.float32)
    mx = float(fire.max()) + 1e-9
    return fire / mx


def _fallback_scores_spatiotemporal(delta: np.ndarray) -> np.ndarray:
    """Anomalia robusta em |ΔBT| por pixel–tempo (sem sklearn)."""
    d = np.abs(np.nan_to_num(delta.astype(np.float64), nan=0.0)).astype(np.float64)
    q50 = float(np.percentile(d, 50))
    q92 = float(np.percentile(d, 92))
    spread = max(q92 - q50, 1e-3)
    z = np.clip((d - q50) / spread, 0.0, 1.0).astype(np.float32)
    return z.reshape(-1)


def build_goes_image_feature_matrix(
    bt7: np.ndarray,
    bt14: np.ndarray,
    delta_bt: np.ndarray,
    res_delta: np.ndarray | None,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    T, H, W = bt7.shape
    parts = [
        bt7[..., None],
        bt14[..., None],
        delta_bt[..., None],
    ]
    if res_delta is not None:
        parts.append(res_delta[..., None])
    X = np.concatenate(parts, axis=-1)
    Xf = X.reshape(-1, X.shape[-1])
    return Xf, (T, H, W)


def predict_unsupervised_anomaly_cube(
    bt7: np.ndarray,
    bt14: np.ndarray,
    delta_bt: np.ndarray,
    res_delta: np.ndarray | None,
    cfg: GOESUnsupervisedConfig,
) -> np.ndarray:
    """
    Retorna cubo (T, H, W) com score de anomalia bruto por instante (normalização global [0,1]).

    Não aplica suavização Gaussiana aqui (fica na agregação `predict_unsupervised_fire_risk` ou no pós-processo por dia).
    """
    Xf, (T, H, W) = build_goes_image_feature_matrix(bt7, bt14, delta_bt, res_delta)
    n = Xf.shape[0]
    use_if = n >= 512 and T >= 2
    try:
        if use_if:
            flat = _anomaly_scores_isolation_forest(Xf, cfg)
        else:
            flat = _fallback_scores_spatiotemporal(delta_bt)
    except Exception:
        flat = _fallback_scores_spatiotemporal(delta_bt)

    cube = flat.reshape(T, H, W).astype(np.float32)
    mx = float(cube.max()) + 1e-9
    return np.clip(cube / mx, 0.0, 1.0).astype(np.float32)


def predict_unsupervised_fire_risk(
    bt7: np.ndarray,
    bt14: np.ndarray,
    delta_bt: np.ndarray,
    res_delta: np.ndarray | None,
    cfg: GOESUnsupervisedConfig,
) -> np.ndarray:
    """
    Retorna grade (H, W) em [0, 1]: risco agregado no tempo por célula.
    """
    cube = predict_unsupervised_anomaly_cube(bt7, bt14, delta_bt, res_delta, cfg)
    agg = np.max(cube, axis=0)
    if cfg.gaussian_smooth_sigma > 0:
        agg = gaussian_filter(agg, sigma=cfg.gaussian_smooth_sigma)
    mx = float(agg.max()) + 1e-9
    return np.clip(agg / mx, 0.0, 1.0).astype(np.float32)


def resample_risk_to_shape(risk: np.ndarray, n_lat: int, n_lon: int) -> np.ndarray:
    """Reamostra grade GOES-proxy para o tamanho da grade do FireDigitalTwin."""
    h, w = risk.shape
    if h == n_lat and w == n_lon:
        return risk.astype(np.float32)
    zy = n_lat / max(h, 1)
    zx = n_lon / max(w, 1)
    out = zoom(risk, (zy, zx), order=1)
    if out.shape[0] != n_lat or out.shape[1] != n_lon:
        out = np.asarray(out, dtype=np.float32)
        # crop ou pad simples
        z = np.zeros((n_lat, n_lon), dtype=np.float32)
        sh0 = min(out.shape[0], n_lat)
        sh1 = min(out.shape[1], n_lon)
        z[:sh0, :sh1] = out[:sh0, :sh1]
        out = z
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def grid_cell_to_lat_lon(i: int, j: int, n_lat: int, n_lon: int) -> Tuple[float, float]:
    lat = CEARA_BBOX["min_lat"] + (float(i) + 0.5) * (CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]) / max(n_lat, 1)
    lon = CEARA_BBOX["min_lon"] + (float(j) + 0.5) * (CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]) / max(n_lon, 1)
    return float(lat), float(lon)


def run_goes16_unsupervised_from_foci(
    df: pd.DataFrame,
    cfg: GOESUnsupervisedConfig | None = None,
    netcdf_path: str | None = None,
) -> Dict[str, Any]:
    """
    Executa o preditor não supervisionado sobre imagens GOES-16 (proxy).

    Args:
        df: Focos com datetime, lat, lon (usado se netcdf_path for None).
        cfg: Hiperparâmetros.
        netcdf_path: Se informado, lê cubo `pyro_goes_proxy_cube.nc` (ou compatível).

    Returns:
        Dicionário com grade de risco, picos e metadados (JSON-serializável).
    """
    cfg = cfg or GOESUnsupervisedConfig()
    if netcdf_path and Path(netcdf_path).is_file():
        ds = _load_proxy_dataset_from_netcdf(netcdf_path)
        bt7, bt14, delta, res_d = _build_stack_from_xarray(ds)
        if res_d is None:
            from src.pyro_caatinga import ClimatologyResidualFrontEnd

            times = pd.DatetimeIndex(ds["time"].values)
            front_b7 = ClimatologyResidualFrontEnd(ewma_lambda=0.05)
            front_b14 = ClimatologyResidualFrontEnd(ewma_lambda=0.05)
            res_b7, _ = front_b7.fit_transform(np.asarray(ds["bt7"].values, dtype=np.float32), times)
            res_b14, _ = front_b14.fit_transform(np.asarray(ds["bt14"].values, dtype=np.float32), times)
            res_d = res_b7 - res_b14
    else:
        bt7, bt14, delta, res_d = _build_stack_from_dataframe(df, cfg)

    risk = predict_unsupervised_fire_risk(bt7, bt14, delta, res_d, cfg)
    n_lat, n_lon = risk.shape
    thr = float(np.percentile(risk, 92)) if risk.size > 10 else 0.5
    thr = max(thr, 0.35)
    peaks: List[Dict[str, Any]] = []
    flat = risk.ravel()
    top_k = min(30, flat.size)
    if top_k > 0:
        idx = np.argpartition(flat, -top_k)[-top_k:]
        for lin in idx:
            ii, jj = divmod(int(lin), n_lon)
            sc = float(risk[ii, jj])
            if sc < thr:
                continue
            lat, lon = grid_cell_to_lat_lon(ii, jj, n_lat, n_lon)
            peaks.append({"lat": round(lat, 4), "lon": round(lon, 4), "score": round(sc, 4)})
    peaks.sort(key=lambda p: p["score"], reverse=True)

    return {
        "technique": "Unsupervised IsolationForest on GOES-16 proxy ABI cube (BT7, BT14, ΔBT, residual Δ)",
        "grid_shape": [int(n_lat), int(n_lon)],
        "grid_resolution_deg": cfg.grid_resolution,
        "max_risk": float(risk.max()),
        "mean_risk": float(risk.mean()),
        "cells_above_0.5": int((risk >= 0.5).sum()),
        "top_peaks": peaks[:15],
        "config": {
            "contamination": cfg.contamination,
            "max_days_history": cfg.max_days_history,
            "frame_minutes": cfg.frame_minutes,
            "source_netcdf": netcdf_path,
        },
        "_risk_grid": risk,
    }


def merge_goes_risk_into_digital_twin(twin: Any, goes_report: Dict[str, Any], weight: float = 0.28) -> None:
    """
    Injeta risco GOES não supervisionado na grade do `FireDigitalTwin` (média ponderada).

    Requer `twin.initialize_from_history` já chamado para existir `risk_grid`.
    """
    risk = goes_report.get("_risk_grid")
    if risk is None or getattr(twin, "risk_grid", None) is None:
        return
    w = float(np.clip(weight, 0.0, 1.0))
    up = resample_risk_to_shape(risk, int(twin.n_lat), int(twin.n_lon))
    ext = np.clip(up, 0.0, 1.0)
    twin.risk_grid = np.clip((1.0 - w) * twin.risk_grid + w * ext, 0.0, 1.0).astype(np.float32)


def export_public_report(goes_report: Dict[str, Any]) -> Dict[str, Any]:
    """Remove arrays numpy para serialização JSON."""
    return {k: v for k, v in goes_report.items() if not k.startswith("_")}
