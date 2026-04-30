"""
Configurações específicas do Ceará para o Digital Twin de Queimadas.

Este arquivo centraliza todos os parâmetros regionais para facilitar
a adaptação do modelo para outros estados/regiões.
"""

from dataclasses import dataclass, field
from typing import List, Dict

# ============================================================================
# Constantes Geográficas
# ============================================================================

CEARA_BBOX = {
    "min_lat": -7.9,  # Sul
    "max_lat": -2.5,  # Norte (litoral)
    "min_lon": -41.5,  # Oeste
    "max_lon": -37.0,  # Leste
}

CEARA_IBGE_CODE = "23"

# ============================================================================
# Mesorregiões do Ceará (IBGE)
# ============================================================================

MESOREGIOES = {
    "Noroeste Cearense": ["Acaraú", "Camocim", "Granja", "Marco", "Martinópole"],
    "Norte Cearense": ["Itapipoca", "Uruburetama", "Sobral", "Santa Quitéria"],
    "Metropolitana de Fortaleza": ["Fortaleza", "Caucaia", "Maracanaú", "Eusébio"],
    "Sertões Cearenses": ["Crateús", "Independência", "Tauá", "Senador Pompeu"],
    "Jaguaribe": ["Jaguaribe", "Aracati", "Russas", "Limoeiro do Norte"],
    "Centro-Sul Cearense": ["Iguatu", "Quixelô", "Acopiara", "Cariús"],
    "Sul Cearense": ["Juazeiro do Norte", "Crato", "Barbalha", "Brejo Santo"],
    "Cariri": ["Juazeiro do Norte", "Crato", "Barbalha", "Santana do Cariri"],
}

# ============================================================================
# Áreas Protegidas Críticas (alta prioridade para monitoramento)
# ============================================================================

AREAS_CRITICAS = {
    "Chapada do Araripe": {
        "centroid": (-7.28, -39.35),
        "bioma": "Cerrado/Caatinga",
        "nivel_risco": "ALTO",
        "descricao": "Floresta Nacional do Araripe + APA — foco histórico de queimadas",
    },
    "Serra de Baturité": {
        "centroid": (-4.25, -38.85),
        "bioma": "Mata Atlântica (Brejo de Altitude)",
        "nivel_risco": "ALTO",
        "descricao": "Reserva da Biosfera, remanescente de Mata Atlântica isolado",
    },
    "Parque Nacional de Ubajara": {
        "centroid": (-3.84, -40.91),
        "bioma": "Caatinga/Mata Atlântica",
        "nivel_risco": "MÉDIO",
        "descricao": "Parque Nacional com grutas e floresta úmida",
    },
    "Delta do Parnaíba (CE)": {
        "centroid": (-2.90, -41.40),
        "bioma": "Manguezal/Restinga",
        "nivel_risco": "MÉDIO",
        "descricao": "Área de mangue — queimadas afetam ecossistema estuarino",
    },
    "Serra da Ibiapaba": {
        "centroid": (-3.80, -41.00),
        "bioma": "Mata Atlântica/Caatinga",
        "nivel_risco": "MÉDIO",
        "descricao": "Encosta úmida com agricultura intensiva",
    },
}

# ============================================================================
# Parâmetros do Modelo de Propagação (Rothermel simplificado)
# ============================================================================

@dataclass
class FireSpreadConfig:
    """Configuração do modelo de propagação do fogo."""

    # Resolução da grade de simulação (graus decimais ~ 1.1km)
    grid_resolution: float = 0.01

    # Fatores de propagação por tipo de cobertura
    fuel_factor: Dict[str, float] = field(
        default_factory=lambda: {
            "Formação Florestal": 1.5,
            "Formação Savânica": 2.0,
            "Pastagem": 2.5,
            "Agricultura": 1.8,
            "Mosaico Agricultura/Pasto": 2.2,
            "Área Não Vegetada": 0.1,
            "Corpo D'água": 0.0,
        }
    )

    # Distância máxima de propagação por passo (km)
    max_spread_km: float = 3.0

    # Número de passos de simulação
    simulation_steps: int = 48

    # Limiar NDVI para vegetação seca (suscetível a fogo)
    ndvi_dry_threshold: float = 0.3

    # Pesos para o índice de perigo (0-1)
    fire_danger_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "historical_density": 0.3,
            "vegetation_dryness": 0.3,
            "proximity_to_roads": 0.1,
            "land_use": 0.2,
            "topography": 0.1,
        }
    )


# ============================================================================
# API de Dados Abertos
# ============================================================================

INPE_API_URL = "https://terrabrasilis.dpi.inpe.br/queimadas/api/focos/"
FIRMS_API_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/"

# Satélites monitorados
SATELITES = ["AQUA_M-T", "TERRA_M-T", "NPP-375", "NOAA-20", "NOAA-21", "GOES-16"]

# ============================================================================
# Configuração de instância
# ============================================================================

fire_config = FireSpreadConfig()
