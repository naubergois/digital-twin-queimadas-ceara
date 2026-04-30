"""
Módulo de imagens de satélite — acesso a dados de observação da Terra.

Fontes suportadas:
  - NASA GIBS (Global Imagery Browse Services) — tiles TrueColor gratuitos
  - NASA FIRMS — focos ativos via API
  - Sentinel Hub — imagens Sentinel-2 (requer client_id/client_secret)
  - ESRI / Google / Bing — tiles de satélite comerciais (gratuitos para uso)

Todas as fontes são acessíveis sem autenticação, exceto Sentinel Hub.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any

import requests

# ─── NASA GIBS ────────────────────────────────────────────────────────────

GIBS_WMTS_URL = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
    "{layer}/default/{date}/{tile_matrix_set}/{z}/{y}/{x}.jpg"
)

GIBS_LAYERS = {
    "modis_terra_truecolor": {
        "layer": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "tile_matrix": "GoogleMapsCompatible_Level9",
        "description": "MODIS Terra — True Color (250m, ~1 dia atrás)",
        "resolution": "250m",
        "delay_days": 1,
    },
    "modis_aqua_truecolor": {
        "layer": "MODIS_Aqua_CorrectedReflectance_TrueColor",
        "tile_matrix": "GoogleMapsCompatible_Level9",
        "description": "MODIS Aqua — True Color (250m, ~1 dia atrás)",
        "resolution": "250m",
        "delay_days": 1,
    },
    "viirs_snpp_truecolor": {
        "layer": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
        "tile_matrix": "GoogleMapsCompatible_Level9",
        "description": "VIIRS Suomi NPP — True Color (250m)",
        "resolution": "250m",
        "delay_days": 0,
    },
    "viirs_noaa20_truecolor": {
        "layer": "VIIRS_NOAA20_CorrectedReflectance_TrueColor",
        "tile_matrix": "GoogleMapsCompatible_Level9",
        "description": "VIIRS NOAA-20 — True Color (250m)",
        "resolution": "250m",
        "delay_days": 0,
    },
    "modis_thermal": {
        "layer": "MODIS_Terra_Brightness_Temp_Band31_Day",
        "tile_matrix": "GoogleMapsCompatible_Level7",
        "description": "MODIS Terra — Temperatura (Band 31, termal)",
        "resolution": "1km",
        "delay_days": 1,
    },
    "viirs_fires": {
        "layer": "VIIRS_SNPP_Thermal_Anomalies_375m_Day",
        "tile_matrix": "GoogleMapsCompatible_Level8",
        "description": "VIIRS — Anomalias Térmicas (focos, 375m)",
        "resolution": "375m",
        "delay_days": 0,
    },
}


def gibs_tile_url(layer_key: str, date: datetime | None = None) -> str:
    """
    Gera URL de template de tiles para NASA GIBS.

    Args:
        layer_key: Nome da camada em GIBS_LAYERS
        date: Data desejada (default: hoje menos delay_days)

    Returns:
        URL template com placeholders {z}/{y}/{x}
    """
    layer = GIBS_LAYERS.get(layer_key)
    if not layer:
        raise ValueError(f"Camada desconhecida: {layer_key}. Opções: {list(GIBS_LAYERS.keys())}")

    if date is None:
        date = datetime.utcnow() - timedelta(days=layer["delay_days"])

    date_str = date.strftime("%Y-%m-%d")

    return (
        "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"{layer['layer']}/default/{date_str}/"
        f"{layer['tile_matrix']}/{{z}}/{{y}}/{{x}}.jpg"
    )


def list_gibs_layers() -> list[dict[str, Any]]:
    """Retorna lista de camadas GIBS disponíveis."""
    return [
        {
            "key": key,
            "name": info["description"],
            "resolution": info["resolution"],
            "delay": f"{info['delay_days']} dia(s)",
        }
        for key, info in GIBS_LAYERS.items()
    ]


# ─── NASA FIRMS (API para focos ativos) ───────────────────────────────────

def fetch_firms_fires(
    api_key: str | None = None,
    source: str = "VIIRS_SNPP_NRT",
    day_range: int = 3,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[dict]:
    """
    Busca focos ativos da NASA FIRMS API.

    Args:
        api_key: Chave da API (se None, tenta env FIRMS_API_KEY)
        source: Fonte ('VIIRS_SNPP_NRT', 'MODIS_NRT', 'VIIRS_NOAA20_NRT')
        day_range: Dias para trás (1-10)
        bbox: (min_lon, min_lat, max_lon, max_lat)

    Returns:
        Lista de dicionários com focos
    """
    if api_key is None:
        api_key = os.environ.get("FIRMS_API_KEY", "")

    if not api_key:
        print("[satellite] FIRMS_API_KEY não configurada. "
              "Use export FIRMS_API_KEY=seu_token ou .env")
        return []

    base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
    day_range = min(day_range, 10)

    params: dict[str, str | float] = {
        "api_key": api_key,
        "source": source,
        "day_range": str(day_range),
    }

    if bbox:
        params["min_lon"], params["min_lat"], params["max_lon"], params["max_lat"] = bbox

    try:
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()

        import csv
        import io

        reader = csv.DictReader(io.StringIO(resp.text))
        fires = []
        for row in reader:
            fires.append({
                "lat": float(row.get("latitude", 0)),
                "lon": float(row.get("longitude", 0)),
                "brightness": float(row.get("bright_ti4", row.get("bright_ti5", 0))),
                "frp": float(row.get("frp", 0)),
                "acq_date": row.get("acq_date", ""),
                "acq_time": row.get("acq_time", ""),
                "satellite": row.get("satellite", source),
                "confidence": row.get("confidence", "n/a"),
            })

        return fires

    except requests.RequestException as e:
        print(f"[satellite] Erro FIRMS API: {e}")
        return []


# ─── Open-Meteo Air Quality (Alternativa gratuita) ────────────────────────

def fetch_open_meteo_fire_index(
    lat: float = -5.2,
    lon: float = -39.5,
) -> dict:
    """
    Busca índice meteorológico de risco de fogo do Open-Meteo (gratuito, sem API key).

    Retorna dados como temperatura, umidade, vento — usados para calcular risco.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                 "wind_speed_10m_max,relative_humidity_2m_max",
        "timezone": "America/Fortaleza",
        "forecast_days": 7,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


def fire_danger_index(weather: dict) -> list[dict]:
    """
    Calcula índice de perigo de fogo a partir de dados meteorológicos.

    Fórmula simplificada (baseada em temperatura, umidade, precipitação, vento).
    Quanto maior o índice, maior o risco.
    """
    if "error" in weather:
        return []

    daily = weather.get("daily", {})
    dates = daily.get("time", [])
    temps = daily.get("temperature_2m_max", [])
    winds = daily.get("wind_speed_10m_max", [])
    hums = daily.get("relative_humidity_2m_max", [])
    precips = daily.get("precipitation_sum", [])

    results = []
    for i in range(len(dates)):
        temp = temps[i] if i < len(temps) else 30
        wind = winds[i] if i < len(winds) else 5
        hum = hums[i] if i < len(hums) else 50
        precip = precips[i] if i < len(precips) else 0

        # Índice simplificado (0-100): temperatura alta + vento + baixa umidade + sem chuva
        t_idx = min(temp / 40 * 30, 30)  # peso 30
        w_idx = min(wind / 20 * 20, 20)  # peso 20
        h_idx = (100 - hum) / 100 * 25   # peso 25
        p_idx = max(0, 25 - precip * 5)  # peso 25, chuva reduz

        fdi = t_idx + w_idx + h_idx + p_idx

        results.append({
            "date": dates[i],
            "fire_danger_index": round(fdi, 1),
            "temp_max": temps[i] if i < len(temps) else None,
            "wind_max": winds[i] if i < len(winds) else None,
            "humidity_max": hums[i] if i < len(hums) else None,
            "precipitation": precips[i] if i < len(precips) else None,
        })

    return results


# ─── Sentinel Hub (opcional, requer autenticação) ─────────────────────────

def fetch_sentinel_image(
    client_id: str | None = None,
    client_secret: str | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    max_cloud: int = 30,
) -> dict | None:
    """
    Busca imagem recente do Sentinel-2 L2A via Sentinel Hub Process API.

    Requer credenciais de OAuth do Sentinel Hub.
    Retorna URL da imagem ou None se indisponível.
    """
    cid = client_id or os.environ.get("SH_CLIENT_ID", "")
    csec = client_secret or os.environ.get("SH_CLIENT_SECRET", "")

    if not cid or not csec:
        print("[satellite] Sentinel Hub não configurado. "
              "Defina SH_CLIENT_ID e SH_CLIENT_SECRET")
        return None

    bbox = bbox or (-41.5, -8.0, -37.0, -2.5)  # Ceará approx

    try:
        # 1. Obter token
        auth_resp = requests.post(
            "https://services.sentinel-hub.com/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": cid,
                "client_secret": csec,
            },
            timeout=15,
        )
        auth_resp.raise_for_status()
        token = auth_resp.json()["access_token"]

        # 2. Buscar cena recente
        search_resp = requests.post(
            "https://services.sentinel-hub.com/api/v1/catalog/search",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "datetime": (
                    datetime.utcnow() - timedelta(days=14)
                ).isoformat() + "Z/"
                + datetime.utcnow().isoformat() + "Z",
                "collections": ["sentinel-2-l2a"],
                "bbox": list(bbox),
                "limit": 1,
                "query": {"eo:cloud_cover": {"lte": max_cloud}},
                "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            },
            timeout=20,
        )
        search_resp.raise_for_status()
        features = search_resp.json().get("features", [])

        if not features:
            return {"error": "Nenhuma cena recente sem nuvens encontrada"}

        return {
            "id": features[0]["id"],
            "date": features[0]["properties"].get("datetime"),
            "cloud_cover": features[0]["properties"].get("eo:cloud_cover"),
        }

    except requests.RequestException as e:
        return {"error": str(e)}


# ─── Funções utilitárias ──────────────────────────────────────────────────

def satellite_layer_for_folium(layer_key: str) -> tuple[str, str, dict]:
    """
    Retorna (tile_url, attribution, options) para usar com folium.TileLayer.

    Exemplos de uso:
        url, attr, opts = satellite_layer_for_folium("gibs_viirs_truecolor")
        folium.TileLayer(url, attr=attr, **opts).add_to(m)
    """
    layers: dict[str, tuple[str, str, dict]] = {
        "esri_satellite": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Esri, Maxar, Earthstar Geographics",
            {"max_zoom": 19, "name": "ESRI Satellite"},
        ),
        "google_satellite": (
            "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            "Google",
            {"max_zoom": 20, "name": "Google Satellite"},
        ),
        "bing_satellite": (
            "https://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1",
            "Microsoft Bing",
            {"max_zoom": 19, "name": "Bing Satellite"},
        ),
        "gibs_modis_terra": (
            gibs_tile_url("modis_terra_truecolor"),
            "NASA GIBS (MODIS Terra)",
            {"max_zoom": 9, "name": "NASA MODIS Terra", "opacity": 0.9},
        ),
        "gibs_modis_aqua": (
            gibs_tile_url("modis_aqua_truecolor"),
            "NASA GIBS (MODIS Aqua)",
            {"max_zoom": 9, "name": "NASA MODIS Aqua", "opacity": 0.9},
        ),
        "gibs_viirs_truecolor": (
            gibs_tile_url("viirs_snpp_truecolor"),
            "NASA GIBS (VIIRS)",
            {"max_zoom": 9, "name": "NASA VIIRS TrueColor", "opacity": 0.9},
        ),
        "gibs_viirs_thermal": (
            gibs_tile_url("viirs_noaa20_truecolor"),
            "NASA GIBS (VIIRS NOAA-20)",
            {"max_zoom": 9, "name": "NASA VIIRS NOAA-20", "opacity": 0.9},
        ),
        "gibs_thermal": (
            gibs_tile_url("modis_thermal"),
            "NASA GIBS (MODIS Termal)",
            {"max_zoom": 7, "name": "MODIS Termal", "opacity": 0.7},
        ),
        "osm": (
            "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
            "OpenStreetMap",
            {"max_zoom": 19, "name": "OpenStreetMap"},
        ),
    }

    return layers.get(layer_key, layers["esri_satellite"])


# ─── Interface simplificada ───────────────────────────────────────────────

def available_satellite_sources() -> list[dict]:
    """Lista todas as fontes de satélite disponíveis para uso."""
    sources = [
        {"id": "esri_satellite", "name": "ESRI World Imagery",
         "type": "Tile", "free": True, "auth": False},
        {"id": "google_satellite", "name": "Google Satellite",
         "type": "Tile", "free": True, "auth": False},
        {"id": "gibs_modis_terra", "name": "NASA MODIS Terra (True Color)",
         "type": "Tile", "free": True, "auth": False, "resolution": "250m",
         "delay": "~1 dia"},
        {"id": "gibs_viirs_truecolor", "name": "NASA VIIRS (True Color)",
         "type": "Tile", "free": True, "auth": False, "resolution": "250m"},
        {"id": "gibs_viirs_thermal", "name": "NASA VIIRS NOAA-20 (True Color)",
         "type": "Tile", "free": True, "auth": False, "resolution": "250m"},
        {"id": "gibs_thermal", "name": "NASA MODIS Termal (Band 31)",
         "type": "Tile", "free": True, "auth": False, "resolution": "1km"},
        {"id": "nasa_firms", "name": "NASA FIRMS (Focos Ativos)",
         "type": "API", "free": True, "auth": True,
         "note": "Requer FIRMS_API_KEY (gratuita em https://firms.modaps.eosdis.nasa.gov)"},
        {"id": "sentinel_hub", "name": "ESA Sentinel-2 L2A",
         "type": "API", "free": True, "auth": True,
         "note": "Requer SH_CLIENT_ID / SH_CLIENT_SECRET"},
    ]
    return sources
