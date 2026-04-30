"""
Módulo de análise exploratória e estatística dos dados de queimadas no Ceará.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import warnings

from config.ceara_config import CEARA_BBOX, MESOREGIOES, AREAS_CRITICAS

warnings.filterwarnings("ignore")


class FireAnalysis:
    """
    Análises estatísticas e temporais dos focos de queimadas no Ceará.
    """

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df
        self._results = {}

    def load_data(self, df: pd.DataFrame):
        """Carrega dataset de focos."""
        self.df = df.copy()
        self._preprocess()

    def load_from_csv(self, path: str):
        """Carrega dataset de arquivo CSV."""
        self.df = pd.read_csv(path)
        self._preprocess()
        print(f"[ANALYSIS] {len(self.df)} focos carregados")

    def _preprocess(self):
        """Pré-processamento básico do dataset."""
        if self.df is None or self.df.empty:
            return

        # Converter datetime
        for col in ["datetime", "data_hora", "data_hora_gmt", "acq_date"]:
            if col in self.df.columns:
                self.df["datetime"] = pd.to_datetime(self.df[col], errors="coerce")
                break

        if "datetime" not in self.df.columns:
            self.df["datetime"] = pd.NaT

        # Extrair componentes temporais
        if not self.df["datetime"].isna().all():
            self.df["year"] = self.df["datetime"].dt.year
            self.df["month"] = self.df["datetime"].dt.month
            self.df["day"] = self.df["datetime"].dt.day
            self.df["doy"] = self.df["datetime"].dt.dayofyear  # day of year
            self.df["weekday"] = self.df["datetime"].dt.weekday

        # Padronizar coordenadas
        lat_col = None
        lon_col = None
        for cl in ["lat", "latitude", "lat_"]:
            if cl in self.df.columns:
                lat_col = cl
                break
        for cl in ["lon", "longitude", "lon_", "lng"]:
            if cl in self.df.columns:
                lon_col = cl
                break

        if lat_col and lon_col:
            self.df["lat"] = pd.to_numeric(self.df[lat_col], errors="coerce")
            self.df["lon"] = pd.to_numeric(self.df[lon_col], errors="coerce")

    # ==========================================================================
    # Análises Temporais
    # ==========================================================================

    def monthly_distribution(self, year: Optional[int] = None) -> pd.DataFrame:
        """Distribuição mensal de focos."""
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        df = self.df.copy()
        if year and "year" in df.columns:
            df = df[df["year"] == year]

        if "month" not in df.columns:
            return pd.DataFrame()

        monthly = df.groupby("month").size().reset_index(name="count")
        monthly["pct"] = (monthly["count"] / monthly["count"].sum() * 100).round(1)

        self._results["monthly"] = monthly
        return monthly

    def yearly_trend(self) -> pd.DataFrame:
        """Tendência anual de focos."""
        if self.df is None or "year" not in self.df.columns:
            return pd.DataFrame()

        yearly = self.df.groupby("year").size().reset_index(name="count")
        yearly["pct_change"] = yearly["count"].pct_change() * 100

        self._results["yearly"] = yearly
        return yearly

    def peak_season(self) -> Dict:
        """Identifica a temporada de pico de queimadas."""
        monthly = self.monthly_distribution()
        if monthly.empty:
            return {}

        peak_month = monthly.loc[monthly["count"].idxmax()]
        # Estação seca (Junho-Dezembro) vs chuvosa (Janeiro-Maio)
        dry_months = monthly[monthly["month"].isin(range(6, 13))]
        wet_months = monthly[monthly["month"].isin(range(1, 6))]

        result = {
            "peak_month": int(peak_month["month"]),
            "peak_count": int(peak_month["count"]),
            "dry_season_total": int(dry_months["count"].sum()),
            "wet_season_total": int(wet_months["count"].sum()),
            "dry_season_pct": round(dry_months["count"].sum() / monthly["count"].sum() * 100, 1)
            if monthly["count"].sum() > 0 else 0,
        }
        self._results["season"] = result
        return result

    # ==========================================================================
    # Análises Geográficas
    # ==========================================================================

    def top_municipios(self, n: int = 10) -> pd.DataFrame:
        """Top N municípios com mais focos."""
        if self.df is None or "municipio" not in self.df.columns:
            # Tentar com coluna de geometria
            if "lat" in self.df.columns:
                return pd.DataFrame({"info": ["Municipio data not available - use lat/lon directly"]})

            return pd.DataFrame()

        top = (
            self.df.groupby("municipio")
            .size()
            .sort_values(ascending=False)
            .head(n)
            .reset_index(name="count")
        )
        top["rank"] = range(1, len(top) + 1)

        self._results["top_municipios"] = top
        return top

    def density_map_data(self) -> pd.DataFrame:
        """
        Prepara dados para mapa de densidade de Kernel.

        Returns:
            DataFrame com coordenadas limpas
        """
        if self.df is None or "lat" not in self.df.columns:
            return pd.DataFrame()

        coords = self.df[["lat", "lon"]].dropna()

        # Filtrar outliers fora do Ceará
        mask = (
            (coords["lat"] >= CEARA_BBOX["min_lat"])
            & (coords["lat"] <= CEARA_BBOX["max_lat"])
            & (coords["lon"] >= CEARA_BBOX["min_lon"])
            & (coords["lon"] <= CEARA_BBOX["max_lon"])
        )
        coords = coords[mask]

        # Amostrar se for muito grande
        if len(coords) > 10000:
            coords = coords.sample(10000, random_state=42)

        return coords

    def satellite_comparison(self) -> pd.DataFrame:
        """Comparação de detecções por satélite."""
        col = None
        for c in ["satellite", "satelite", "sat_name"]:
            if c in self.df.columns:
                col = c
                break

        if not col:
            return pd.DataFrame()

        sat = (
            self.df.groupby(col)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="count")
        )
        sat["pct"] = (sat["count"] / sat["count"].sum() * 100).round(1)
        return sat

    def daily_anomaly_detection(self, year: int) -> pd.DataFrame:
        """
        Detecta dias com número anômalo de queimadas (acima de 2 desvios padrão).

        Args:
            year: Ano para análise

        Returns:
            DataFrame com dias anômalos
        """
        if self.df is None or "doy" not in self.df.columns:
            return pd.DataFrame()

        df_year = self.df[self.df["year"] == year].copy()
        if df_year.empty:
            return pd.DataFrame()

        daily = df_year.groupby("doy").size().reset_index(name="count")
        mean = daily["count"].mean()
        std = daily["count"].std()
        threshold = mean + 2 * std

        anomalies = daily[daily["count"] > threshold].copy()
        anomalies["z_score"] = (anomalies["count"] - mean) / std

        # Converter DOY para data
        anomalies["date"] = anomalies["doy"].apply(
            lambda d: datetime(year, 1, 1) + pd.Timedelta(days=int(d) - 1)
        )

        return anomalies.sort_values("count", ascending=False)

    # ==========================================================================
    # Relatórios Resumidos
    # ==========================================================================

    def summary_report(self) -> Dict:
        """Gera relatório resumido completo."""
        if self.df is None or self.df.empty:
            return {"error": "Sem dados carregados"}

        report = {
            "total_focos": len(self.df),
            "periodo": {},
            "dados": {
                "fonte": self.df.get("source", ["N/A"]).iloc[0]
                if "source" in self.df.columns else "N/A",
            },
        }

        # Período
        if "datetime" in self.df.columns and not self.df["datetime"].isna().all():
            report["periodo"] = {
                "inicio": self.df["datetime"].min().strftime("%Y-%m-%d"),
                "fim": self.df["datetime"].max().strftime("%Y-%m-%d"),
                "dias": (self.df["datetime"].max() - self.df["datetime"].min()).days,
            }

        # Anos
        if "year" in self.df.columns:
            report["anos"] = sorted(self.df["year"].dropna().unique().tolist())
            report["media_anual"] = round(
                self.df.groupby("year").size().mean(), 1
            )

        # Sazonalidade
        season = self.peak_season()
        if season:
            report["sazonalidade"] = season

        # Biomas
        if "bioma" in self.df.columns:
            biomas = self.df["bioma"].value_counts().head(3)
            report["top_biomas"] = [
                {"bioma": k, "focos": int(v)}
                for k, v in biomas.items()
            ]

        # Satélites
        sat = self.satellite_comparison()
        if not sat.empty:
            report["satelites"] = sat.head(5).to_dict("records")

        return report
