"""PYRO-Caatinga MVP.

Implementa um esqueleto funcional da proposta:
1) Front-end climatology-residual online (EWMA por pixel, dia-do-ano e hora)
2) Destilacao cruzada VIIRS -> GOES via soft labels gaussianos
3) Cabeca fisica simplificada (mascara de fogo e proxy de FRP)
4) Loop de feedback com gêmeo digital para pseudo-rotulos em t+5

Este modulo foi desenhado para funcionar com os dados atuais do projeto
(focos pontuais), criando um cubo proxy GOES em grade temporal de 5 min.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import gaussian_filter
from sklearn.metrics import f1_score, precision_score, recall_score

from config.ceara_config import CEARA_BBOX


@dataclass
class PyroCaatingaConfig:
    grid_resolution: float = 0.1
    frame_minutes: int = 5
    max_days_history: int = 7
    ewma_lambda: float = 0.05
    soft_label_sigma_cells: float = 1.2
    uncertainty_tau: float = 0.015
    baseline_fire_threshold: float = 0.45
    residual_fire_threshold: float = 0.35


class ClimatologyResidualFrontEnd:
    """Mantem climatologia pixel-a-pixel por (doy, hod) e calcula residual."""

    def __init__(self, ewma_lambda: float = 0.05):
        self.ewma_lambda = float(ewma_lambda)
        self._mu: Dict[Tuple[int, int], np.ndarray] = {}

    def fit_transform(self, bt_cube: np.ndarray, times: pd.DatetimeIndex) -> tuple[np.ndarray, dict]:
        residual = np.zeros_like(bt_cube, dtype=np.float32)

        for t_idx, ts in enumerate(times):
            key = (int(ts.dayofyear), int(ts.hour))
            frame = bt_cube[t_idx].astype(np.float32)

            if key not in self._mu:
                self._mu[key] = frame.copy()
            else:
                self._mu[key] = (1.0 - self.ewma_lambda) * self._mu[key] + self.ewma_lambda * frame

            residual[t_idx] = frame - self._mu[key]

        meta = {
            "keys": len(self._mu),
            "ewma_lambda": self.ewma_lambda,
        }
        return residual, meta


class ViirsGoesDistiller:
    """Construtor de soft labels VIIRS reprojetadas para grade GOES."""

    def __init__(self, sigma_cells: float = 1.2):
        self.sigma_cells = float(sigma_cells)

    @staticmethod
    def _lat_lon_to_grid(lat: float, lon: float, n_lat: int, n_lon: int) -> tuple[int, int]:
        i = int((n_lat - 1) * (lat - CEARA_BBOX["min_lat"]) / (CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]))
        j = int((n_lon - 1) * (lon - CEARA_BBOX["min_lon"]) / (CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]))
        i = max(0, min(n_lat - 1, i))
        j = max(0, min(n_lon - 1, j))
        return i, j

    def build_soft_labels(
        self,
        viirs_df: pd.DataFrame,
        times: pd.DatetimeIndex,
        n_lat: int,
        n_lon: int,
        frame_minutes: int,
    ) -> np.ndarray:
        labels = np.zeros((len(times), n_lat, n_lon), dtype=np.float32)
        if viirs_df.empty:
            return labels

        dfx = viirs_df.copy()
        dfx["datetime"] = pd.to_datetime(dfx["datetime"], errors="coerce")
        dfx = dfx.dropna(subset=["datetime", "lat", "lon"])
        if dfx.empty:
            return labels

        start = times.min()
        freq = max(1, int(frame_minutes))

        for _, row in dfx.iterrows():
            dt = pd.Timestamp(row["datetime"])
            idx = int(((dt - start).total_seconds() // 60) // freq)
            if idx < 0 or idx >= len(times):
                continue
            i, j = self._lat_lon_to_grid(float(row["lat"]), float(row["lon"]), n_lat, n_lon)
            labels[idx, i, j] = 1.0

        for t in range(labels.shape[0]):
            if labels[t].max() > 0:
                labels[t] = gaussian_filter(labels[t], sigma=self.sigma_cells)
                labels[t] /= max(1e-6, float(labels[t].max()))

        return labels


class DigitalTwinFeedbackLoop:
    """Gera pseudo-rotulos confiaveis com base no erro do twin em t+5."""

    def __init__(self, uncertainty_tau: float = 0.015):
        self.uncertainty_tau = float(uncertainty_tau)

    @staticmethod
    def mc_dropout_uncertainty(prob_mc: np.ndarray) -> np.ndarray:
        # prob_mc: [T_mc, time, y, x]
        return np.var(prob_mc, axis=0)

    def build_pseudo_labels(
        self,
        pred_prob_t1: np.ndarray,
        twin_pred_t1: np.ndarray,
        goes_obs_t1: np.ndarray,
        uncertainty_t1: np.ndarray,
    ) -> np.ndarray:
        confident = (uncertainty_t1 < self.uncertainty_tau).astype(np.float32)
        consistency = (1.0 - np.abs(twin_pred_t1 - goes_obs_t1)).astype(np.float32)
        pseudo = confident * (0.65 * pred_prob_t1 + 0.35 * consistency)
        return np.clip(pseudo, 0.0, 1.0)


def _lat_lon_to_grid(lat: float, lon: float, n_lat: int, n_lon: int) -> tuple[int, int]:
    i = int((n_lat - 1) * (lat - CEARA_BBOX["min_lat"]) / (CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]))
    j = int((n_lon - 1) * (lon - CEARA_BBOX["min_lon"]) / (CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]))
    i = max(0, min(n_lat - 1, i))
    j = max(0, min(n_lon - 1, j))
    return i, j


def build_goes_proxy_cube(df: pd.DataFrame, cfg: PyroCaatingaConfig) -> xr.Dataset:
    dfx = df.copy()
    dfx["datetime"] = pd.to_datetime(dfx["datetime"], errors="coerce")
    dfx = dfx.dropna(subset=["datetime", "lat", "lon"])
    if dfx.empty:
        raise ValueError("Sem dados validos para montar cubo GOES proxy.")

    if int(cfg.max_days_history) > 0:
        end_dt = pd.Timestamp(dfx["datetime"].max())
        start_cut = end_dt - pd.Timedelta(days=int(cfg.max_days_history))
        dfx = dfx[dfx["datetime"] >= start_cut].copy()
        if dfx.empty:
            raise ValueError("Sem dados no recorte temporal configurado para o PYRO-Caatinga.")

    n_lat = int((CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]) / cfg.grid_resolution)
    n_lon = int((CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]) / cfg.grid_resolution)
    n_lat = max(8, n_lat)
    n_lon = max(8, n_lon)

    t0 = pd.Timestamp(dfx["datetime"].min()).floor(f"{cfg.frame_minutes}min")
    t1 = pd.Timestamp(dfx["datetime"].max()).ceil(f"{cfg.frame_minutes}min")
    times = pd.date_range(t0, t1, freq=f"{cfg.frame_minutes}min")

    counts = np.zeros((len(times), n_lat, n_lon), dtype=np.float32)
    dt_to_idx = {t: i for i, t in enumerate(times)}

    for _, row in dfx.iterrows():
        ts = pd.Timestamp(row["datetime"]).floor(f"{cfg.frame_minutes}min")
        idx = dt_to_idx.get(ts)
        if idx is None:
            continue
        i, j = _lat_lon_to_grid(float(row["lat"]), float(row["lon"]), n_lat, n_lon)
        counts[idx, i, j] += 1.0

    # Proxy simples de bandas termicas a partir de contagem espacial-temporal.
    smoothed = np.zeros_like(counts)
    for k in range(len(times)):
        smoothed[k] = gaussian_filter(counts[k], sigma=1.0)

    hour = np.array([t.hour for t in times], dtype=np.float32)
    diurnal = 6.0 * np.sin(2.0 * np.pi * (hour - 13.0) / 24.0)

    bt7 = 47.0 + diurnal[:, None, None] + 8.0 * smoothed
    bt14 = 41.0 + 0.8 * diurnal[:, None, None] + 4.0 * smoothed
    delta_bt = bt7 - bt14
    glm = (counts > 0).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            "bt7": (("time", "y", "x"), bt7.astype(np.float32)),
            "bt14": (("time", "y", "x"), bt14.astype(np.float32)),
            "delta_bt": (("time", "y", "x"), delta_bt.astype(np.float32)),
            "glm": (("time", "y", "x"), glm.astype(np.float32)),
            "fire_obs": (("time", "y", "x"), (counts > 0).astype(np.float32)),
        },
        coords={
            "time": times,
            "y": np.arange(n_lat),
            "x": np.arange(n_lon),
        },
        attrs={
            "grid_resolution": cfg.grid_resolution,
            "frame_minutes": cfg.frame_minutes,
            "max_days_history": cfg.max_days_history,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": "Proxy GOES cube built from point detections",
        },
    )
    return ds


def run_pyro_caatinga_mvp(
    goes_df: pd.DataFrame,
    output_dir: str,
    cfg: PyroCaatingaConfig | None = None,
    viirs_df: pd.DataFrame | None = None,
) -> dict:
    cfg = cfg or PyroCaatingaConfig()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = build_goes_proxy_cube(goes_df, cfg)

    # 1) Climatology-residual front-end.
    front_b7 = ClimatologyResidualFrontEnd(ewma_lambda=cfg.ewma_lambda)
    front_b14 = ClimatologyResidualFrontEnd(ewma_lambda=cfg.ewma_lambda)
    residual_bt7, clim_meta = front_b7.fit_transform(ds["bt7"].values, pd.DatetimeIndex(ds["time"].values))
    residual_bt14, _ = front_b14.fit_transform(ds["bt14"].values, pd.DatetimeIndex(ds["time"].values))
    residual_delta = residual_bt7 - residual_bt14

    ds_res = xr.Dataset(
        data_vars={
            "res_bt7": (("time", "y", "x"), residual_bt7.astype(np.float32)),
            "res_bt14": (("time", "y", "x"), residual_bt14.astype(np.float32)),
            "res_delta": (("time", "y", "x"), residual_delta.astype(np.float32)),
        },
        coords=ds.coords,
        attrs={"ewma_lambda": cfg.ewma_lambda, "climatology_keys": clim_meta["keys"]},
    )

    # 2) Destilacao VIIRS -> GOES.
    distiller = ViirsGoesDistiller(sigma_cells=cfg.soft_label_sigma_cells)
    viirs_ref = goes_df if viirs_df is None else viirs_df
    soft_labels = distiller.build_soft_labels(
        viirs_df=viirs_ref,
        times=pd.DatetimeIndex(ds["time"].values),
        n_lat=ds.sizes["y"],
        n_lon=ds.sizes["x"],
        frame_minutes=cfg.frame_minutes,
    )

    # 3) Cabeca fisica simplificada: prob de fogo por residual termico.
    score_raw = 0.55 * ds["delta_bt"].values + 0.35 * ds["bt7"].values + 0.10 * ds["glm"].values
    score_res = 0.60 * residual_delta + 0.30 * residual_bt7 + 0.10 * ds["glm"].values

    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    p_raw = _sigmoid((score_raw - np.nanmean(score_raw)) / (np.nanstd(score_raw) + 1e-6))
    p_res = _sigmoid((score_res - np.nanmean(score_res)) / (np.nanstd(score_res) + 1e-6))

    # FRP proxy consistente com energia termica relativa (simplificado).
    frp_proxy = np.clip((np.maximum(0.0, residual_bt7) ** 4) - (np.maximum(0.0, residual_bt14) ** 4), 0.0, None)

    # 4) Loop de feedback do twin em t+5 (proxy).
    twin_pred_t1 = np.roll((p_res >= cfg.residual_fire_threshold).astype(np.float32), shift=1, axis=0)
    goes_obs_t1 = np.roll(ds["fire_obs"].values.astype(np.float32), shift=1, axis=0)

    # MC-dropout proxy: ruido gaussiano em 20 amostras.
    mc_samples = []
    rng = np.random.default_rng(42)
    for _ in range(20):
        noise = rng.normal(0.0, 0.035, size=p_res.shape).astype(np.float32)
        mc_samples.append(np.clip(p_res + noise, 0.0, 1.0))
    prob_mc = np.stack(mc_samples, axis=0)

    loop = DigitalTwinFeedbackLoop(uncertainty_tau=cfg.uncertainty_tau)
    unc = loop.mc_dropout_uncertainty(prob_mc)
    pseudo = loop.build_pseudo_labels(
        pred_prob_t1=np.roll(p_res, shift=1, axis=0),
        twin_pred_t1=twin_pred_t1,
        goes_obs_t1=goes_obs_t1,
        uncertainty_t1=unc,
    )

    y_true = ds["fire_obs"].values.reshape(-1)
    y_raw = (p_raw >= cfg.baseline_fire_threshold).astype(np.uint8).reshape(-1)
    y_res = (p_res >= cfg.residual_fire_threshold).astype(np.uint8).reshape(-1)

    metrics = {
        "baseline": {
            "precision": float(precision_score(y_true, y_raw, zero_division=0)),
            "recall": float(recall_score(y_true, y_raw, zero_division=0)),
            "f1": float(f1_score(y_true, y_raw, zero_division=0)),
        },
        "residual": {
            "precision": float(precision_score(y_true, y_res, zero_division=0)),
            "recall": float(recall_score(y_true, y_res, zero_division=0)),
            "f1": float(f1_score(y_true, y_res, zero_division=0)),
        },
        "uncertainty": {
            "mean": float(np.mean(unc)),
            "p90": float(np.quantile(unc, 0.90)),
            "tau": cfg.uncertainty_tau,
        },
    }

    # Persistencia dos artefatos.
    cube_path = out / "pyro_goes_proxy_cube.nc"
    residual_path = out / "pyro_residual_cube.nc"
    soft_path = out / "pyro_viirs_soft_labels.npy"
    pseudo_path = out / "pyro_twin_pseudo_labels.npy"
    frp_path = out / "pyro_frp_proxy.npy"
    report_path = out / "pyro_caatinga_report.json"

    ds.to_netcdf(cube_path)
    ds_res.to_netcdf(residual_path)
    np.save(soft_path, soft_labels.astype(np.float32))
    np.save(pseudo_path, pseudo.astype(np.float32))
    np.save(frp_path, frp_proxy.astype(np.float32))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "technique": "PYRO-Caatinga MVP",
        "config": {
            "grid_resolution": cfg.grid_resolution,
            "frame_minutes": cfg.frame_minutes,
            "max_days_history": cfg.max_days_history,
            "ewma_lambda": cfg.ewma_lambda,
            "soft_label_sigma_cells": cfg.soft_label_sigma_cells,
            "uncertainty_tau": cfg.uncertainty_tau,
            "baseline_fire_threshold": cfg.baseline_fire_threshold,
            "residual_fire_threshold": cfg.residual_fire_threshold,
        },
        "dataset": {
            "frames": int(ds.sizes["time"]),
            "grid_shape": [int(ds.sizes["y"]), int(ds.sizes["x"])],
            "fire_positive_rate": float(ds["fire_obs"].values.mean()),
        },
        "metrics": metrics,
        "outputs": {
            "goes_proxy_cube": str(cube_path),
            "residual_cube": str(residual_path),
            "viirs_soft_labels": str(soft_path),
            "twin_pseudo_labels": str(pseudo_path),
            "frp_proxy": str(frp_path),
            "report": str(report_path),
        },
    }

    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
