"""
Gêmeo Digital Híbrido (ML + Autômato Celular) para queimadas no Ceará.

Técnica:
- Modelo supervisionado de risco por célula-dia (Gradient Boosting + rede neural temporal)
- Acoplamento em gêmeo digital com propagação espacial em grade
- Validação temporal com dados reais (holdout no período final)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import convolve, gaussian_filter, maximum_filter
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.ceara_config import CEARA_BBOX


def _jsonable_sklearn_param(obj: Any) -> Any:
    """Converte get_params() (com sub-estimadores sklearn) em estrutura JSON-serializável."""
    from sklearn.base import BaseEstimator

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, BaseEstimator):
        return {
            "class": obj.__class__.__name__,
            "params": _jsonable_sklearn_param(obj.get_params(deep=False)),
        }
    if isinstance(obj, dict):
        return {str(k): _jsonable_sklearn_param(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable_sklearn_param(x) for x in obj]
    return repr(obj)


@dataclass
class MLTwinConfig:
    grid_resolution: float = 0.2
    lookback_days: int = 3
    test_ratio: float = 0.2
    proba_threshold: float = 0.35
    twin_spread_threshold: float = 0.45
    auto_calibrate: bool = True
    min_recall_target: float = 0.7
    use_deep_temporal: bool = True
    ensemble_weight_gb: float = 0.6
    train_val_ratio: float = 0.8
    min_precision_target: float = 0.08
    twin_pday_weight: float = 0.9
    twin_cooldown_days: int = 2
    optimize_metric: str = "f1"
    tolerant_radius_cells: int = 4
    benchmark_sample_size: int = 120000
    max_positive_rate: float = 0.25
    mode: str = "operational"
    use_hard_negative_mining: bool = True
    hnm_neg_pos_ratio: int = 6
    use_cost_sensitive: bool = True
    positive_class_weight: float | None = None
    recency_weight_power: float = 0.8
    # "f1": maximiza F1 / IoU no conjunto de calibração (validação interna).
    # "low_fp": maximiza precisão mantendo recall >= min_recall_target (reduz alarmes falsos).
    calibration_objective: str = "f1"


class FireMLDigitalTwin:
    """Gêmeo digital híbrido com validação em dados reais."""

    def __init__(self, config: MLTwinConfig | None = None):
        self.config = config or MLTwinConfig()
        self.n_lat = int((CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]) / self.config.grid_resolution)
        self.n_lon = int((CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]) / self.config.grid_resolution)

        self.model = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=300,
            l2_regularization=0.05,
            random_state=42,
        )
        self.deep_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=200,
            random_state=42,
        )

        self.model_candidates = {
            "hist_gb": self.model,
            "random_forest": RandomForestClassifier(
                n_estimators=260,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=320,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                learning_rate=0.06,
                n_estimators=220,
                max_depth=3,
                random_state=42,
            ),
            "mlp_temporal": self.deep_model,
            "logistic": LogisticRegression(
                max_iter=300,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ),
            "soft_vote_tree_linear": VotingClassifier(
                estimators=[
                    (
                        "et",
                        ExtraTreesClassifier(
                            n_estimators=180,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            n_jobs=-1,
                            random_state=42,
                        ),
                    ),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=300,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=42,
                        ),
                    ),
                ],
                voting="soft",
                n_jobs=-1,
            ),
        }

    def _compute_sample_weights(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        weights = np.ones(len(y), dtype=np.float32)
        if len(y) == 0:
            return weights

        if self.config.use_cost_sensitive:
            pos = int((y == 1).sum())
            neg = int((y == 0).sum())
            if self.config.positive_class_weight is not None:
                pos_weight = float(max(1.0, self.config.positive_class_weight))
            else:
                ratio = (neg / max(1, pos)) if pos > 0 else 1.0
                pos_weight = float(np.clip(ratio, 2.0, 40.0))
            weights[y == 1] *= pos_weight

        if self.config.recency_weight_power > 0 and X.ndim == 2 and X.shape[1] > 0:
            day_progress = np.clip(X[:, -1], 0.0, 1.0)
            weights *= (1.0 + float(self.config.recency_weight_power) * day_progress)

        weights /= max(1e-9, float(weights.mean()))
        return weights

    def _fit_model(self, model, X: np.ndarray, y: np.ndarray):
        sample_weights = self._compute_sample_weights(X, y)
        try:
            model.fit(X, y, sample_weight=sample_weights)
        except TypeError:
            model.fit(X, y)
        return model

    def _sample_for_benchmark(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_rows: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(y) <= max_rows:
            return X, y
        rng = np.random.default_rng(42)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        pos_take = min(len(pos_idx), max_rows // 2)
        neg_take = max_rows - pos_take
        neg_take = min(len(neg_idx), neg_take)

        pos_sel = rng.choice(pos_idx, size=pos_take, replace=False) if pos_take > 0 else np.array([], dtype=int)
        neg_sel = rng.choice(neg_idx, size=neg_take, replace=False) if neg_take > 0 else np.array([], dtype=int)
        sel = np.concatenate([pos_sel, neg_sel])
        rng.shuffle(sel)
        return X[sel], y[sel]

    def _spatial_tolerant_scores(
        self,
        y_days: np.ndarray,
        p_days: np.ndarray,
        radius: int,
    ) -> Dict[str, float]:
        size = 2 * int(max(0, radius)) + 1
        tp = 0
        fp = 0
        fn = 0

        for gt, pred in zip(y_days, p_days):
            gt_b = (gt > 0).astype(np.uint8)
            pr_b = (pred > 0).astype(np.uint8)
            gt_d = maximum_filter(gt_b, size=size)
            pr_d = maximum_filter(pr_b, size=size)

            tp += int(np.logical_and(pr_b == 1, gt_d == 1).sum())
            fp += int(np.logical_and(pr_b == 1, gt_d == 0).sum())
            fn += int(np.logical_and(gt_b == 1, pr_d == 0).sum())

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1_tol = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
        return {
            "precision_tolerant": float(precision),
            "recall_tolerant": float(recall),
            "f1_tolerant": float(f1_tol),
        }

    def _evaluate_candidate(
        self,
        model_name: str,
        model,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_days: int,
    ) -> Dict:
        fit_X, fit_y = self._sample_for_benchmark(X_fit, y_fit, self.config.benchmark_sample_size)
        mdl = clone(model)
        mdl = self._fit_model(mdl, fit_X, fit_y)
        proba = mdl.predict_proba(X_val)[:, 1]
        pred = (proba >= 0.5).astype(np.uint8)

        y_days = y_val.reshape(val_days, self.n_lat, self.n_lon)
        p_days = pred.reshape(val_days, self.n_lat, self.n_lon)
        tol = self._spatial_tolerant_scores(y_days, p_days, self.config.tolerant_radius_cells)
        y_day = (y_days.reshape(val_days, -1).sum(axis=1) > 0).astype(np.uint8)
        p_day = (p_days.reshape(val_days, -1).sum(axis=1) > 0).astype(np.uint8)

        out = {
            "model": model_name,
            "roc_auc": self._safe_metric(roc_auc_score, y_val, proba),
            "pr_auc": self._safe_metric(average_precision_score, y_val, proba),
            "precision": self._safe_metric(precision_score, y_val, pred, zero_division=0),
            "recall": self._safe_metric(recall_score, y_val, pred, zero_division=0),
            "f1": self._safe_metric(f1_score, y_val, pred, zero_division=0),
            "positive_rate": float(pred.mean()),
            "f1_day": self._safe_metric(f1_score, y_day, p_day, zero_division=0),
            **tol,
            "_model_obj": mdl,
            "_proba_val": proba,
        }
        return out

    def _benchmark_models(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        val_days: int,
    ) -> tuple[object, Dict, List[Dict]]:
        reports = []
        for name, model in self.model_candidates.items():
            rep = self._evaluate_candidate(name, model, X_fit, y_fit, X_val, y_val, val_days)
            reports.append(rep)

        metric_name = self.config.optimize_metric
        if metric_name not in reports[0]:
            metric_name = "f1"

        feasible = [
            r
            for r in reports
            if r.get("precision", 0.0) >= self.config.min_precision_target
            and r.get("positive_rate", 1.0) <= self.config.max_positive_rate
        ]
        pool = feasible if feasible else reports

        best = max(
            pool,
            key=lambda r: (
                r.get(metric_name, 0.0),
                r.get("f1", 0.0),
                r.get("pr_auc", 0.0),
                r.get("f1_day", 0.0),
            ),
        )
        best_model = best["_model_obj"]

        clean_reports = []
        for r in reports:
            rr = {k: v for k, v in r.items() if not k.startswith("_")}
            clean_reports.append(rr)

        best_summary = {k: v for k, v in best.items() if not k.startswith("_")}
        return best_model, best_summary, clean_reports

    def _fit_with_hard_negative_mining(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ):
        """Refina o treino focando negativos difíceis para melhorar F1 estrito por célula."""
        if not self.config.use_hard_negative_mining:
            return self._fit_model(model, X_train, y_train)

        mdl = clone(model)
        X0, y0 = self._sample_for_benchmark(X_train, y_train, self.config.benchmark_sample_size)
        mdl = self._fit_model(mdl, X0, y0)

        pos_idx = np.where(y0 == 1)[0]
        neg_idx = np.where(y0 == 0)[0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return mdl

        proba = mdl.predict_proba(X0)[:, 1]
        hard_rank = neg_idx[np.argsort(proba[neg_idx])[::-1]]
        keep_neg = min(len(hard_rank), int(len(pos_idx) * max(1, self.config.hnm_neg_pos_ratio)))
        hard_neg = hard_rank[:keep_neg]

        sel = np.concatenate([pos_idx, hard_neg])
        np.random.default_rng(42).shuffle(sel)
        mdl = clone(model)
        mdl = self._fit_model(mdl, X0[sel], y0[sel])
        return mdl

    def _lat_lon_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        i = int((self.n_lat - 1) * (lat - CEARA_BBOX["min_lat"]) / (CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]))
        j = int((self.n_lon - 1) * (lon - CEARA_BBOX["min_lon"]) / (CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]))
        return max(0, min(i, self.n_lat - 1)), max(0, min(j, self.n_lon - 1))

    def _build_best_model_params_record(
        self,
        chosen_proba_threshold: float,
        chosen_twin_threshold: float,
        best_summary: Dict,
    ) -> Dict[str, Any]:
        """Hiperparâmetros MLTwinConfig + limiares escolhidos + params do classificador ajustado."""
        bs = {k: v for k, v in best_summary.items() if not str(k).startswith("_")}
        return {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "ml_twin_config": asdict(self.config),
            "chosen_operating_point": {
                "proba_threshold": float(chosen_proba_threshold),
                "twin_spread_threshold": float(chosen_twin_threshold),
            },
            "best_model_selection": bs,
            "fitted_classifier_class": self.model.__class__.__name__,
            "fitted_classifier_params": _jsonable_sklearn_param(self.model.get_params(deep=False)),
        }

    def _grid_to_lat_lon_center(self, i: int, j: int) -> tuple[float, float]:
        lat = CEARA_BBOX["min_lat"] + (float(i) + 0.5) * self.config.grid_resolution
        lon = CEARA_BBOX["min_lon"] + (float(j) + 0.5) * self.config.grid_resolution
        return float(lat), float(lon)

    def _prepare_daily_grids(self, df: pd.DataFrame) -> tuple[List[pd.Timestamp], np.ndarray, pd.DataFrame]:
        dfx = df.copy()
        dfx["datetime"] = pd.to_datetime(dfx["datetime"], errors="coerce")
        dfx = dfx.dropna(subset=["datetime", "lat", "lon"])
        dfx["date"] = dfx["datetime"].dt.floor("D")
        if "municipio" not in dfx.columns:
            dfx["municipio"] = ""
        dfx["municipio"] = dfx["municipio"].fillna("").astype(str).str.strip().str.upper()

        date_index = pd.date_range(dfx["date"].min(), dfx["date"].max(), freq="D")
        grids = np.zeros((len(date_index), self.n_lat, self.n_lon), dtype=np.float32)
        date_to_idx = {d: i for i, d in enumerate(date_index)}

        for _, row in dfx.iterrows():
            i, j = self._lat_lon_to_grid(float(row["lat"]), float(row["lon"]))
            tidx = date_to_idx.get(row["date"])
            if tidx is not None:
                grids[tidx, i, j] += 1.0

        grids = (grids > 0).astype(np.float32)
        return list(date_index), grids, dfx

    def _build_features(
        self,
        dates: List[pd.Timestamp],
        grids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
        lookback = self.config.lookback_days
        hotspot_prior = gaussian_filter(grids.sum(axis=0), sigma=1.2)
        hotspot_prior = hotspot_prior / (hotspot_prior.max() + 1e-9)
        lat_norm = np.linspace(0.0, 1.0, self.n_lat, dtype=np.float32)[:, None]
        lat_norm = np.repeat(lat_norm, self.n_lon, axis=1)
        lon_norm = np.linspace(0.0, 1.0, self.n_lon, dtype=np.float32)[None, :]
        lon_norm = np.repeat(lon_norm, self.n_lat, axis=0)

        feats: List[np.ndarray] = []
        deep_feats: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        feat_dates: List[pd.Timestamp] = []

        for t in range(lookback, len(dates)):
            lag_stack = grids[t - lookback:t]
            lag1 = lag_stack[-1]
            lagk = lag_stack.sum(axis=0)
            lag_recent = lag_stack[-min(2, lookback):].sum(axis=0)
            neigh = gaussian_filter(lag1, sigma=1.0)
            neigh_k = gaussian_filter(lagk, sigma=1.4)
            trend = lag_stack[-1] - lag_stack[0]

            d = pd.Timestamp(dates[t])
            doy = d.dayofyear
            month = d.month
            dry_season = 1.0 if month >= 6 else 0.0
            day_progress = t / max(1, len(dates) - 1)

            doy_sin = np.sin(2 * np.pi * doy / 366.0)
            doy_cos = np.cos(2 * np.pi * doy / 366.0)
            mon_sin = np.sin(2 * np.pi * month / 12.0)
            mon_cos = np.cos(2 * np.pi * month / 12.0)

            f = np.stack(
                [
                    hotspot_prior,
                    lag1,
                    lagk,
                    lag_recent,
                    neigh,
                    neigh_k,
                    trend,
                    lat_norm,
                    lon_norm,
                    np.full_like(hotspot_prior, dry_season),
                    np.full_like(hotspot_prior, doy_sin),
                    np.full_like(hotspot_prior, doy_cos),
                    np.full_like(hotspot_prior, mon_sin),
                    np.full_like(hotspot_prior, mon_cos),
                    np.full_like(hotspot_prior, day_progress),
                ],
                axis=-1,
            )

            feats.append(f.reshape(-1, f.shape[-1]))

            # Features temporais para rede neural (sequência compacta por célula)
            seq_channels = []
            for k in range(lookback):
                lk = lag_stack[k]
                seq_channels.append(lk)
                seq_channels.append(gaussian_filter(lk, sigma=1.0))
            seq_channels.extend(
                [
                    hotspot_prior,
                    trend,
                    lat_norm,
                    lon_norm,
                    np.full_like(hotspot_prior, dry_season),
                    np.full_like(hotspot_prior, doy_sin),
                    np.full_like(hotspot_prior, doy_cos),
                    np.full_like(hotspot_prior, mon_sin),
                    np.full_like(hotspot_prior, mon_cos),
                    np.full_like(hotspot_prior, day_progress),
                ]
            )
            sf = np.stack(seq_channels, axis=-1)
            deep_feats.append(sf.reshape(-1, sf.shape[-1]))

            labels.append(grids[t].reshape(-1))
            feat_dates.append(d)

        X = np.vstack(feats)
        X_deep = np.vstack(deep_feats)
        y = np.concatenate(labels)
        return X, X_deep, y, feat_dates

    @staticmethod
    def _safe_metric(fn, *args, **kwargs):
        try:
            return float(fn(*args, **kwargs))
        except Exception:
            return float("nan")

    def _build_cell_municipio_lookup(self, dfx: pd.DataFrame) -> Dict[Tuple[int, int], str]:
        if dfx.empty:
            return {}
        rows = []
        for _, row in dfx.iterrows():
            mun = str(row.get("municipio", "")).strip().upper()
            if not mun:
                continue
            i, j = self._lat_lon_to_grid(float(row["lat"]), float(row["lon"]))
            rows.append((i, j, mun))
        if not rows:
            return {}
        cdf = pd.DataFrame(rows, columns=["i", "j", "municipio"])
        modes = (
            cdf.groupby(["i", "j"])["municipio"]
            .agg(lambda x: x.value_counts().index[0])
            .reset_index()
        )
        return {(int(r["i"]), int(r["j"])): r["municipio"] for _, r in modes.iterrows()}

    def _simulate_twin(
        self,
        p_test_days: np.ndarray,
        twin_spread_threshold: float,
        pday_weight: float | None = None,
        cooldown_days: int | None = None,
    ) -> np.ndarray:
        if pday_weight is None:
            pday_weight = self.config.twin_pday_weight
        if cooldown_days is None:
            cooldown_days = self.config.twin_cooldown_days

        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
        burning = np.zeros((self.n_lat, self.n_lon), dtype=np.uint8)
        cooldown = np.zeros((self.n_lat, self.n_lon), dtype=np.int16)

        twin_preds = []
        for pday in p_test_days:
            neigh_pressure = convolve(burning.astype(np.float32), kernel, mode="constant", cval=0.0)
            neigh_pressure = np.clip(neigh_pressure / 8.0, 0.0, 1.0)

            risk = np.clip(pday_weight * pday + (1.0 - pday_weight) * neigh_pressure, 0.0, 1.0)
            ignite = (risk >= twin_spread_threshold).astype(np.uint8)

            available = (cooldown <= 0).astype(np.uint8)
            new_burning = np.where((ignite == 1) & (available == 1), 1, 0).astype(np.uint8)
            cooldown = np.where(cooldown > 0, cooldown - 1, cooldown)
            cooldown = np.where(new_burning == 1, int(cooldown_days), cooldown)
            burning = new_burning
            twin_preds.append(burning.copy())

        return np.array(twin_preds)

    def _validate_by_season(
        self,
        test_dates: List[pd.Timestamp],
        y_test_days: np.ndarray,
        ml_pred_days: np.ndarray,
        twin_pred_days: np.ndarray,
    ) -> Dict:
        seasons = {
            "seca": lambda m: m >= 6,
            "chuvosa": lambda m: m <= 5,
        }
        out = {}
        for name, cond in seasons.items():
            idx = [i for i, d in enumerate(test_dates) if cond(pd.Timestamp(d).month)]
            if not idx:
                out[name] = {}
                continue
            y = y_test_days[idx].reshape(-1)
            p_ml = ml_pred_days[idx].reshape(-1)
            p_tw = twin_pred_days[idx].reshape(-1)

            out[name] = {
                "days": len(idx),
                "ml_precision": self._safe_metric(precision_score, y, p_ml, zero_division=0),
                "ml_recall": self._safe_metric(recall_score, y, p_ml, zero_division=0),
                "ml_f1": self._safe_metric(f1_score, y, p_ml, zero_division=0),
                "twin_precision": self._safe_metric(precision_score, y, p_tw, zero_division=0),
                "twin_recall": self._safe_metric(recall_score, y, p_tw, zero_division=0),
                "twin_f1": self._safe_metric(f1_score, y, p_tw, zero_division=0),
            }
        return out

    def _validate_by_municipio(
        self,
        dfx: pd.DataFrame,
        test_dates: List[pd.Timestamp],
        ml_pred_days: np.ndarray,
        twin_pred_days: np.ndarray,
        cell_to_muni: Dict[Tuple[int, int], str],
    ) -> Dict:
        dfx_test = dfx[dfx["date"].isin(test_dates)].copy()
        if dfx_test.empty:
            return {"rows": []}

        real_by_muni = (
            dfx_test[dfx_test["municipio"] != ""]
            .groupby("municipio")
            .size()
            .sort_values(ascending=False)
        )
        top_muni = real_by_muni.head(15).index.tolist()

        pred_ml_count = {m: 0 for m in top_muni}
        pred_twin_count = {m: 0 for m in top_muni}

        def _accumulate(day_grid: np.ndarray, target: Dict[str, int]):
            ys, xs = np.where(day_grid > 0)
            for i, j in zip(ys, xs):
                m = cell_to_muni.get((int(i), int(j)))
                if m in target:
                    target[m] += 1

        for d in range(ml_pred_days.shape[0]):
            _accumulate(ml_pred_days[d], pred_ml_count)
            _accumulate(twin_pred_days[d], pred_twin_count)

        rows = []
        for m in top_muni:
            real = int(real_by_muni.get(m, 0))
            mlp = int(pred_ml_count.get(m, 0))
            twp = int(pred_twin_count.get(m, 0))
            rows.append(
                {
                    "municipio": m,
                    "real_focos": real,
                    "ml_pred_focos": mlp,
                    "twin_pred_focos": twp,
                    "ml_abs_error": abs(mlp - real),
                    "twin_abs_error": abs(twp - real),
                }
            )

        if not rows:
            return {"rows": []}

        rdf = pd.DataFrame(rows)
        return {
            "rows": rows,
            "ml_mae": float(mean_absolute_error(rdf["real_focos"], rdf["ml_pred_focos"])),
            "twin_mae": float(mean_absolute_error(rdf["real_focos"], rdf["twin_pred_focos"])),
        }

    def _calibrate_thresholds(
        self,
        y_test_days: np.ndarray,
        p_test_days: np.ndarray,
    ) -> Tuple[float, float, Dict]:
        y_test = y_test_days.reshape(-1)
        p_test = p_test_days.reshape(-1)

        proba_grid = np.arange(0.15, 0.70, 0.05)
        twin_grid = np.arange(0.20, 0.80, 0.05)

        candidates = []
        for pth in proba_grid:
            pred = (p_test >= pth).astype(np.uint8)
            rec = self._safe_metric(recall_score, y_test, pred, zero_division=0)
            f1 = self._safe_metric(f1_score, y_test, pred, zero_division=0)
            prec = self._safe_metric(precision_score, y_test, pred, zero_division=0)
            f2 = (5 * prec * rec / (4 * prec + rec + 1e-12)) if (prec + rec) > 0 else 0.0
            candidates.append({
                "proba_threshold": float(pth),
                "recall": float(rec),
                "precision": float(prec),
                "f1": float(f1),
                "f2": float(f2),
            })

        feasible = [
            c
            for c in candidates
            if c["recall"] >= self.config.min_recall_target
            and c["precision"] >= self.config.min_precision_target
        ]
        if self.config.calibration_objective == "low_fp":
            feas_fp = [
                c
                for c in candidates
                if c["recall"] >= self.config.min_recall_target
                and c["precision"] >= self.config.min_precision_target
            ]
            if feas_fp:
                best_proba = max(feas_fp, key=lambda c: (c["precision"], c["recall"], c["f1"]))
            else:
                feas_rec = [c for c in candidates if c["recall"] >= self.config.min_recall_target]
                if feas_rec:
                    best_proba = max(feas_rec, key=lambda c: (c["precision"], c["f2"]))
                else:
                    best_proba = max(candidates, key=lambda c: (c["precision"], c["recall"]))
        elif feasible:
            best_proba = max(feasible, key=lambda c: (c["f1"], c["f2"]))
        else:
            best_proba = max(candidates, key=lambda c: c["f1"])

        # Calibrar limiar do twin (F1/IoU ou precisão priorizada).
        twin_scores = []
        for tth in twin_grid:
            twin_pred_days = self._simulate_twin(
                p_test_days,
                twin_spread_threshold=float(tth),
                pday_weight=self.config.twin_pday_weight,
                cooldown_days=self.config.twin_cooldown_days,
            )
            iou_daily = []
            for gt, pd_ in zip(y_test_days, twin_pred_days):
                inter = np.logical_and(gt > 0, pd_ > 0).sum()
                union = np.logical_or(gt > 0, pd_ > 0).sum()
                if union > 0:
                    iou_daily.append(float(inter / union))
            twin_flat = twin_pred_days.reshape(-1)
            gt_flat = y_test_days.reshape(-1)
            twin_f1 = self._safe_metric(f1_score, gt_flat, twin_flat, zero_division=0)
            twin_prec = self._safe_metric(precision_score, gt_flat, twin_flat, zero_division=0)
            twin_rec = self._safe_metric(recall_score, gt_flat, twin_flat, zero_division=0)
            twin_scores.append({
                "twin_spread_threshold": float(tth),
                "twin_f1": float(twin_f1),
                "twin_precision": float(twin_prec),
                "twin_recall": float(twin_rec),
                "spatial_iou_mean": float(np.mean(iou_daily)) if iou_daily else 0.0,
            })

        if self.config.calibration_objective == "low_fp":
            min_r = float(self.config.min_recall_target) * 0.9
            feas_tw = [s for s in twin_scores if s["twin_recall"] >= min_r]
            if feas_tw:
                best_twin = max(
                    feas_tw,
                    key=lambda c: (c["twin_precision"], c["twin_recall"], c["spatial_iou_mean"]),
                )
            else:
                best_twin = max(
                    twin_scores,
                    key=lambda c: (c["twin_precision"], c["twin_f1"]),
                )
        else:
            best_twin = max(twin_scores, key=lambda c: (c["twin_f1"], c["spatial_iou_mean"]))

        calibration_report = {
            "min_recall_target": self.config.min_recall_target,
            "min_precision_target": self.config.min_precision_target,
            "max_positive_rate": self.config.max_positive_rate,
            "chosen_proba": best_proba,
            "chosen_twin": best_twin,
            "optimize_metric": self.config.optimize_metric,
            "top_proba_candidates": sorted(candidates, key=lambda c: c["f2"], reverse=True)[:5],
            "top_twin_candidates": sorted(
                twin_scores, key=lambda c: (c["twin_f1"], c["spatial_iou_mean"]), reverse=True
            )[:5],
        }

        return float(best_proba["proba_threshold"]), float(best_twin["twin_spread_threshold"]), calibration_report

    def validate_with_real_data(self, df: pd.DataFrame) -> Dict:
        """Treina e valida o gêmeo digital híbrido em holdout temporal real."""
        dates, grids, dfx = self._prepare_daily_grids(df)
        if len(dates) < (self.config.lookback_days + 20):
            raise ValueError("Período insuficiente para validação temporal do modelo ML.")

        X, X_deep, y, feat_dates = self._build_features(dates, grids)
        cells = self.n_lat * self.n_lon
        n_days_feat = len(feat_dates)

        split_day_test = int(n_days_feat * (1.0 - self.config.test_ratio))
        split_row_test = split_day_test * cells

        X_train_all, y_train_all = X[:split_row_test], y[:split_row_test]
        X_test, y_test = X[split_row_test:], y[split_row_test:]
        Xd_train_all, Xd_test = X_deep[:split_row_test], X_deep[split_row_test:]
        test_dates = feat_dates[split_day_test:]

        # Validação interna temporal para calibrar limiares sem vazar o conjunto de teste.
        train_days_all = split_day_test
        train_days_fit = max(self.config.lookback_days + 5, int(train_days_all * self.config.train_val_ratio))
        train_days_fit = min(train_days_fit, train_days_all - 3)
        split_row_val = train_days_fit * cells

        X_fit, y_fit = X_train_all[:split_row_val], y_train_all[:split_row_val]
        X_val, y_val = X_train_all[split_row_val:], y_train_all[split_row_val:]
        Xd_val = Xd_train_all[split_row_val:]

        # Benchmark de técnicas e seleção automática do melhor modelo.
        best_model, best_summary, benchmark_reports = self._benchmark_models(
            X_fit=X_fit,
            y_fit=y_fit,
            X_val=X_val,
            y_val=y_val,
            val_days=max(1, len(y_val) // cells),
        )
        self.model = self._fit_with_hard_negative_mining(best_model, X_fit, y_fit)

        proba_val = self.model.predict_proba(X_val)[:, 1]
        proba = self.model.predict_proba(X_test)[:, 1]

        y_val_days = y_val.reshape(-1, self.n_lat, self.n_lon)
        p_val_days = proba_val.reshape(-1, self.n_lat, self.n_lon)
        y_test_days = y_test.reshape(-1, self.n_lat, self.n_lon)
        p_test_days = proba.reshape(-1, self.n_lat, self.n_lon)

        calibration_report = {}
        chosen_proba_threshold = self.config.proba_threshold
        chosen_twin_threshold = self.config.twin_spread_threshold
        if self.config.auto_calibrate:
            chosen_proba_threshold, chosen_twin_threshold, calibration_report = self._calibrate_thresholds(
                y_test_days=y_val_days,
                p_test_days=p_val_days,
            )

        # Em modo estrito, ajusta limiar diretamente para maximizar F1 por célula no conjunto de validação.
        if self.config.mode == "strict_cell":
            best_f1 = -1.0
            best_thr = chosen_proba_threshold
            for thr in np.arange(0.05, 0.96, 0.01):
                pv = (proba_val >= thr).astype(np.uint8)
                pr = self._safe_metric(precision_score, y_val, pv, zero_division=0)
                if pr < self.config.min_precision_target:
                    continue
                f1v = self._safe_metric(f1_score, y_val, pv, zero_division=0)
                if f1v > best_f1:
                    best_f1 = f1v
                    best_thr = float(thr)
            chosen_proba_threshold = best_thr
            calibration_report["strict_cell_threshold"] = {
                "chosen": chosen_proba_threshold,
                "f1_val": float(best_f1 if best_f1 >= 0 else 0.0),
            }

        pred = (proba >= chosen_proba_threshold).astype(np.uint8)
        pred_days = pred.reshape(-1, self.n_lat, self.n_lon)
        tol_ml = self._spatial_tolerant_scores(y_test_days, pred_days, self.config.tolerant_radius_cells)
        y_day = (y_test_days.reshape(y_test_days.shape[0], -1).sum(axis=1) > 0).astype(np.uint8)
        p_day_ml = (pred_days.reshape(pred_days.shape[0], -1).sum(axis=1) > 0).astype(np.uint8)

        # Métricas de classificação por célula-dia
        metrics = {
            "roc_auc": self._safe_metric(roc_auc_score, y_test, proba),
            "pr_auc": self._safe_metric(average_precision_score, y_test, proba),
            "brier": self._safe_metric(brier_score_loss, y_test, proba),
            "precision": self._safe_metric(precision_score, y_test, pred, zero_division=0),
            "recall": self._safe_metric(recall_score, y_test, pred, zero_division=0),
            "f1": self._safe_metric(f1_score, y_test, pred, zero_division=0),
            "f1_day": self._safe_metric(f1_score, y_day, p_day_ml, zero_division=0),
            **tol_ml,
        }

        # Simulação do gêmeo digital no período de teste
        twin_preds = self._simulate_twin(
            p_test_days=p_test_days,
            twin_spread_threshold=chosen_twin_threshold,
            pday_weight=self.config.twin_pday_weight,
            cooldown_days=self.config.twin_cooldown_days,
        )

        twin_pred = np.array(twin_preds).reshape(-1)
        y_true = y_test_days.reshape(-1)

        twin_metrics = {
            "precision": self._safe_metric(precision_score, y_true, twin_pred, zero_division=0),
            "recall": self._safe_metric(recall_score, y_true, twin_pred, zero_division=0),
            "f1": self._safe_metric(f1_score, y_true, twin_pred, zero_division=0),
        }
        twin_tol = self._spatial_tolerant_scores(y_test_days, twin_preds, self.config.tolerant_radius_cells)
        twin_metrics.update(twin_tol)
        p_day_twin = (twin_preds.reshape(twin_preds.shape[0], -1).sum(axis=1) > 0).astype(np.uint8)
        twin_metrics["f1_day"] = self._safe_metric(f1_score, y_day, p_day_twin, zero_division=0)

        # IoU espacial diário (presença/ausência por célula)
        iou_daily = []
        for gt, pd_ in zip(y_test_days, twin_preds):
            inter = np.logical_and(gt > 0, pd_ > 0).sum()
            union = np.logical_or(gt > 0, pd_ > 0).sum()
            if union > 0:
                iou_daily.append(float(inter / union))

        # Comparação explícita com dados reais (focos encontrados vs observados)
        y_ml = y_test
        p_ml = pred
        tp = int(np.logical_and(y_ml == 1, p_ml == 1).sum())
        fp = int(np.logical_and(y_ml == 0, p_ml == 1).sum())
        fn = int(np.logical_and(y_ml == 1, p_ml == 0).sum())
        tn = int(np.logical_and(y_ml == 0, p_ml == 0).sum())

        daily_real = y_test_days.reshape(y_test_days.shape[0], -1).sum(axis=1)
        daily_ml = pred.reshape(y_test_days.shape[0], -1).sum(axis=1)
        daily_twin = twin_preds.reshape(y_test_days.shape[0], -1).sum(axis=1)

        has_real = daily_real > 0
        has_ml = daily_ml > 0
        has_twin = daily_twin > 0

        detection_report = {
            "cell_confusion": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            },
            "real_cells_positive": int((y_ml == 1).sum()),
            "ml_cells_positive": int((p_ml == 1).sum()),
            "twin_cells_positive": int(twin_pred.sum()),
            "day_level_detection": {
                "days_with_real_fire": int(has_real.sum()),
                "ml_days_detected": int(np.logical_and(has_real, has_ml).sum()),
                "twin_days_detected": int(np.logical_and(has_real, has_twin).sum()),
                "ml_day_detection_rate": float(np.logical_and(has_real, has_ml).sum() / max(1, has_real.sum())),
                "twin_day_detection_rate": float(np.logical_and(has_real, has_twin).sum() / max(1, has_real.sum())),
            },
            "daily_counts_sample": {
                "real": [int(x) for x in daily_real[:15]],
                "ml_pred": [int(x) for x in daily_ml[:15]],
                "twin_pred": [int(x) for x in daily_twin[:15]],
            },
        }

        daily_series_comparison = {
            "dates": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in test_dates],
            "real": [int(x) for x in daily_real],
            "ml_pred": [int(x) for x in daily_ml],
            "twin_pred": [int(x) for x in daily_twin],
        }

        # Comparacao espacial do ultimo dia do conjunto de teste para visualizacao em mapa.
        if len(test_dates) > 0:
            gt_last = (y_test_days[-1] > 0)
            ml_last = (pred_days[-1] > 0)
            tw_last = (twin_preds[-1] > 0)

            both_ml = np.logical_and(gt_last, ml_last)
            real_only = np.logical_and(gt_last, np.logical_not(ml_last))
            ml_only = np.logical_and(ml_last, np.logical_not(gt_last))
            tw_only = np.logical_and(tw_last, np.logical_not(gt_last))

            def _cells_to_points(mask: np.ndarray, label: str) -> List[Dict]:
                ys, xs = np.where(mask)
                out = []
                for i, j in zip(ys, xs):
                    lat_c, lon_c = self._grid_to_lat_lon_center(int(i), int(j))
                    out.append({"lat": lat_c, "lon": lon_c, "label": label})
                return out

            spatial_last_day = {
                "date": pd.Timestamp(test_dates[-1]).strftime("%Y-%m-%d"),
                "real_and_ml": _cells_to_points(both_ml, "real_and_ml"),
                "real_only": _cells_to_points(real_only, "real_only"),
                "ml_only": _cells_to_points(ml_only, "ml_only"),
                "twin_only": _cells_to_points(tw_only, "twin_only"),
            }
        else:
            spatial_last_day = {
                "date": None,
                "real_and_ml": [],
                "real_only": [],
                "ml_only": [],
                "twin_only": [],
            }

        cell_to_muni = self._build_cell_municipio_lookup(dfx[dfx["date"].isin(feat_dates[:split_day_test])])
        municipality_validation = self._validate_by_municipio(
            dfx=dfx,
            test_dates=test_dates,
            ml_pred_days=pred_days,
            twin_pred_days=twin_preds,
            cell_to_muni=cell_to_muni,
        )
        season_validation = self._validate_by_season(
            test_dates=test_dates,
            y_test_days=y_test_days,
            ml_pred_days=pred_days,
            twin_pred_days=twin_preds,
        )

        result = {
            "technique": "Hybrid ML+DL Digital Twin (HistGradientBoosting + Temporal MLP + Cellular Automata)",
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "grid_resolution": self.config.grid_resolution,
                "lookback_days": self.config.lookback_days,
                "test_ratio": self.config.test_ratio,
                "proba_threshold": chosen_proba_threshold,
                "twin_spread_threshold": chosen_twin_threshold,
                "auto_calibrate": self.config.auto_calibrate,
                "min_recall_target": self.config.min_recall_target,
                "use_deep_temporal": self.config.use_deep_temporal,
                "ensemble_weight_gb": self.config.ensemble_weight_gb,
                "optimize_metric": self.config.optimize_metric,
                "tolerant_radius_cells": self.config.tolerant_radius_cells,
                "max_positive_rate": self.config.max_positive_rate,
                "mode": self.config.mode,
                "use_hard_negative_mining": self.config.use_hard_negative_mining,
                "hnm_neg_pos_ratio": self.config.hnm_neg_pos_ratio,
                "use_cost_sensitive": self.config.use_cost_sensitive,
                "positive_class_weight": self.config.positive_class_weight,
                "recency_weight_power": self.config.recency_weight_power,
                "calibration_objective": self.config.calibration_objective,
                "grid_shape": [self.n_lat, self.n_lon],
            },
            "dataset": {
                "days_total": len(dates),
                "days_used": n_days_feat,
                "days_train": split_day_test,
                "days_test": n_days_feat - split_day_test,
                "days_fit": train_days_fit,
                "days_val": train_days_all - train_days_fit,
                "cells_per_day": cells,
                "samples_train": int(len(y_train_all)),
                "samples_test": int(len(y_test)),
                "positive_rate_train": float(y_train_all.mean()),
                "positive_rate_test": float(y_test.mean()),
            },
            "ml_metrics": metrics,
            "twin_metrics": twin_metrics,
            "twin_spatial_iou_mean": float(np.mean(iou_daily)) if iou_daily else 0.0,
            "twin_spatial_iou_std": float(np.std(iou_daily)) if iou_daily else 0.0,
            "calibration": calibration_report,
            "real_data_comparison": detection_report,
            "daily_series_comparison": daily_series_comparison,
            "spatial_comparison_last_day": spatial_last_day,
            "municipality_validation": municipality_validation,
            "season_validation": season_validation,
            "model_selection": {
                "best_model": best_summary,
                "benchmark": benchmark_reports,
            },
            "best_model_params": self._build_best_model_params_record(
                chosen_proba_threshold,
                chosen_twin_threshold,
                best_summary,
            ),
        }
        return result