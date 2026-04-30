"""
Motor do Gêmeo Digital — Simulação de Propagação de Queimadas.

Implementa um modelo simplificado baseado em autômato celular para
simular a propagação do fogo no território cearense.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import json

from config.ceara_config import (
    CEARA_BBOX,
    fire_config,
    AREAS_CRITICAS,
)


class FireDigitalTwin:
    """
    Gêmeo Digital simplificado para simulação de propagação de queimadas.

    Usa um autômato celular 2D onde cada célula representa uma área no terreno.
    A propagação depende de:
      - Tipo de combustível (cobertura vegetal)
      - Densidade histórica de queimadas na região
      - Proximidade de focos ativos
    """

    def __init__(self, resolution: Optional[float] = None):
        self.resolution = resolution or fire_config.grid_resolution
        self.config = fire_config

        # Dimensões da grade
        self.n_lat = int(
            (CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]) / self.resolution
        )
        self.n_lon = int(
            (CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]) / self.resolution
        )

        # Grids de estado
        self.fuel_grid = None  # Tipo de combustível (cobertura)
        self.fire_grid = None  # Estado atual do fogo (0 = sem, 1 = queimando)
        self.burned_grid = None  # Já queimado (acumulador)
        self.risk_grid = None  # Risco de fogo (0-1)
        self.history_grid = None  # Densidade histórica

        # Metadados
        self.current_step = 0
        self.last_update = None

        print(f"[TWIN] Grade criada: {self.n_lat}x{self.n_lon} células "
              f"({self.n_lat * self.n_lon} total)")
        print(f"[TWIN] Resolução: ~{self.resolution * 111:.1f} km por célula")

    def lat_lon_to_grid(self, lat: float, lon: float) -> Tuple[int, int]:
        """Converte coordenadas geográficas para índices da grade."""
        i = int((self.n_lat - 1) * (lat - CEARA_BBOX["min_lat"])
                / (CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]))
        j = int((self.n_lon - 1) * (lon - CEARA_BBOX["min_lon"])
                / (CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]))
        return max(0, min(i, self.n_lat - 1)), max(0, min(j, self.n_lon - 1))

    def initialize_from_history(self, fire_df: pd.DataFrame):
        """
        Inicializa o twin com base em dados históricos de queimadas.

        Args:
            fire_df: DataFrame com focos de calor (colunas: lat, lon)
        """
        print(f"[TWIN] Inicializando com {len(fire_df)} focos históricos...")

        self.fuel_grid = np.ones((self.n_lat, self.n_lon)) * 1.5  # default fuel
        self.fire_grid = np.zeros((self.n_lat, self.n_lon))
        self.burned_grid = np.zeros((self.n_lat, self.n_lon))
        self.risk_grid = np.zeros((self.n_lat, self.n_lon))
        self.history_grid = np.zeros((self.n_lat, self.n_lon))

        # Calcular densidade histórica
        for _, row in fire_df.iterrows():
            try:
                lat, lon = float(row["lat"]), float(row["lon"])
                i, j = self.lat_lon_to_grid(lat, lon)
                self.history_grid[i, j] += 1
            except (ValueError, KeyError):
                continue

        # Suavizar densidade (kernel gaussiano)
        from scipy.ndimage import gaussian_filter

        self.history_grid = gaussian_filter(self.history_grid, sigma=5)
        self.history_grid = self.history_grid / (self.history_grid.max() + 1e-10)

        # Calcular grid de risco combinado
        self._compute_risk()

        print(f"[TWIN] Inicialização concluída. "
              f"Células com queimadas históricas: {(self.history_grid > 0).sum()}")

    def _compute_risk(self):
        """Calcula o grid de risco combinando histórico + fatores ambientais."""
        weights = self.config.fire_danger_weights

        # Por enquanto usamos apenas densidade histórica + fator de combustível
        # Em versões futuras: adicionar vento, umidade, NDVI
        if self.history_grid is not None and self.fuel_grid is not None:
            # Normalizar fuel grid (0-1)
            fuel_norm = self.fuel_grid / max(self.fuel_grid.max(), 1)
            fuel_norm = np.clip(fuel_norm, 0, 1)

            self.risk_grid = (
                weights["historical_density"] * self.history_grid
                + weights["land_use"] * fuel_norm
                + weights["vegetation_dryness"] * 0.5  # placeholder - usar NDVI real
            )

    def add_active_fires(self, active_fires: pd.DataFrame):
        """
        Adiciona focos ativos como novos pontos de ignição.

        Args:
            active_fires: DataFrame com focos ativos (lat, lon)
        """
        count = 0
        for _, row in active_fires.iterrows():
            try:
                lat, lon = float(row["lat"]), float(row["lon"])
                i, j = self.lat_lon_to_grid(lat, lon)

                if self.fire_grid[i, j] == 0 and self.burned_grid[i, j] == 0:
                    self.fire_grid[i, j] = 1
                    count += 1
            except (ValueError, KeyError):
                continue

        print(f"[TWIN] {count} novos focos ativos adicionados à simulação")
        self.last_update = datetime.now()

    def step(self) -> Dict:
        """
        Executa um passo da simulação de propagação.

        Usa um autômato celular: cada célula em chamas pode propagar
        para as 8 células vizinhas, com probabilidade baseada no risco.

        Returns:
            Dict com estatísticas do passo
        """
        if self.fire_grid is None:
            return {"error": "Twin não inicializado"}

        new_fire = np.zeros_like(self.fire_grid)

        for i in range(1, self.n_lat - 1):
            for j in range(1, self.n_lon - 1):
                if self.fire_grid[i, j] == 1 and self.burned_grid[i, j] < 1:
                    # Propagar para vizinhos (8-direções)
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue

                            ni, nj = i + di, j + dj
                            if 0 <= ni < self.n_lat and 0 <= nj < self.n_lon:
                                if self.burned_grid[ni, nj] > 0:
                                    continue  # já queimou

                                # Probabilidade de propagação
                                fuel = self.fuel_grid[ni, nj]
                                risk = self.risk_grid[i, j]
                                prob = min(0.8, fuel * risk * 0.3)

                                # Fator de distância (diagonais mais longas)
                                dist_factor = 1.0 if di == 0 or dj == 0 else 0.7

                                if np.random.random() < prob * dist_factor:
                                    new_fire[ni, nj] = 1

                    # Marcar célula atual como queimada
                    self.burned_grid[i, j] = 1
                    self.fire_grid[i, j] = 0

        # Adicionar novas células em chamas
        self.fire_grid = np.clip(self.fire_grid + new_fire, 0, 1)

        self.current_step += 1

        stats = {
            "step": self.current_step,
            "burning_cells": int(self.fire_grid.sum()),
            "burned_cells": int(self.burned_grid.sum()),
            "total_affected": int((self.fire_grid + self.burned_grid > 0).sum()),
        }
        return stats

    def simulate(self, steps: int = 48) -> List[Dict]:
        """
        Executa múltiplos passos da simulação.

        Args:
            steps: Número de passos (cada passo ~ 1 hora)

        Returns:
            Lista de estatísticas por passo
        """
        print(f"[TWIN] Simulando {steps} passos de propagação...")
        history = []

        for _ in range(steps):
            stats = self.step()
            history.append(stats)

        print(f"[TWIN] Simulação concluída. "
              f"Total afetado: {stats['total_affected']} células")
        return history

    def get_fire_danger_zones(self, threshold: float = 0.6) -> List[Dict]:
        """
        Identifica zonas de alto risco de queimadas.

        Args:
            threshold: Limiar de risco (0-1)

        Returns:
            Lista de áreas críticas com coordenadas aproximadas
        """
        if self.risk_grid is None:
            return []

        zones = []
        high_risk = self.risk_grid > threshold

        # Encontrar clusters de alto risco
        from scipy.ndimage import label

        labeled, num_features = label(high_risk)

        for feature_id in range(1, num_features + 1):
            mask = labeled == feature_id
            cells = mask.sum()
            if cells < 5:  # Ignorar clusters muito pequenos
                continue

            # Centroide
            ys, xs = np.where(mask)
            center_i, center_j = int(ys.mean()), int(xs.mean())

            lat = CEARA_BBOX["min_lat"] + (center_i / self.n_lat) * (
                CEARA_BBOX["max_lat"] - CEARA_BBOX["min_lat"]
            )
            lon = CEARA_BBOX["min_lon"] + (center_j / self.n_lon) * (
                CEARA_BBOX["max_lon"] - CEARA_BBOX["min_lon"]
            )
            mean_risk = float(self.risk_grid[mask].mean())

            zones.append({
                "centroid_lat": round(lat, 4),
                "centroid_lon": round(lon, 4),
                "risk_level": round(mean_risk, 3),
                "area_cells": int(cells),
                "estimated_area_km2": round(cells * (self.resolution * 111) ** 2, 2),
            })

        # Ordenar por risco
        zones.sort(key=lambda z: z["risk_level"], reverse=True)

        return zones

    def check_critical_areas(self) -> List[Dict]:
        """
        Verifica o status das áreas críticas pré-definidas.

        Returns:
            Lista com status de cada área crítica
        """
        results = []

        for name, area in AREAS_CRITICAS.items():
            lat, lon = area["centroid"]
            i, j = self.lat_lon_to_grid(lat, lon)

            status = {
                "area": name,
                "bioma": area["bioma"],
                "nivel_risco": area["nivel_risco"],
                "focos_historicos": int(self.history_grid[i, j])
                if self.history_grid is not None else 0,
                "risco_atual": round(float(self.risk_grid[i, j]), 3)
                if self.risk_grid is not None else 0,
                "em_chamas": bool(self.fire_grid[i, j] > 0)
                if self.fire_grid is not None else False,
            }
            results.append(status)

        return results

    def export_state(self, path: str):
        """Exporta o estado atual do twin para JSON."""
        state = {
            "metadata": {
                "resolution": self.resolution,
                "grid_shape": (self.n_lat, self.n_lon),
                "current_step": self.current_step,
                "last_update": self.last_update.isoformat() if self.last_update else None,
            },
            "stats": {
                "burning_cells": int(self.fire_grid.sum()) if self.fire_grid is not None else 0,
                "burned_cells": int(self.burned_grid.sum()) if self.burned_grid is not None else 0,
                "total_affected": int(
                    (self.fire_grid + self.burned_grid > 0).sum()
                ) if self.fire_grid is not None and self.burned_grid is not None else 0,
            },
            "critical_areas": self.check_critical_areas(),
            "danger_zones": self.get_fire_danger_zones()[:10],
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        print(f"[TWIN] Estado exportado para {path}")
        return state
