"""
Módulo de coleta de dados de queimadas de fontes abertas (INPE, NASA FIRMS).
"""

import csv
import io
import json
import os
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

from config.ceara_config import (
    CEARA_BBOX,
    INPE_API_URL,
    FIRMS_API_URL,
    SATELITES,
)

from src.satellite import fetch_firms_fires as _fetch_firms_fires_satellite


def _normalize_satellite_name(value: str) -> str:
    """Normaliza nomes de satélite para comparação robusta."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _build_inpe_url(path: str = "") -> str:
    """Monta URL INPE sem duplicar segmentos quando a base já inclui /focos."""
    base = INPE_API_URL.rstrip("/")
    if not path:
        return base
    return f"{base}/{path.lstrip('/')}"


def _parse_kml_description(description: str) -> dict:
    """Extrai pares chave/valor do campo description do KML do INPE."""
    if not description:
        return {}

    txt = re.sub(r"<br\s*/?>", "\n", description, flags=re.IGNORECASE)
    txt = re.sub(r"</?b>", "", txt, flags=re.IGNORECASE)
    txt = re.sub(r"<[^>]+>", "", txt)
    txt = txt.replace("&nbsp;", " ")

    out = {}
    for line in txt.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = _normalize_satellite_name(k)
        out[key] = v.strip()
    return out


def fetch_inpe_fire_foci(
    state_code: str = "23",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    satellites: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Baixa focos de calor do INPE (BDQueimadas / TerraBrasilis).

    Args:
        state_code: Código IBGE do estado (23 = Ceará)
        date_from: Data inicial (YYYY-MM-DD). Default: 7 dias atrás
        date_to: Data final (YYYY-MM-DD). Default: hoje
        satellites: Lista de satélites. Default: todos

    Returns:
        DataFrame com focos de calor
    """
    if date_from is None:
        date_from = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    if date_to is None:
        date_to = datetime.now().strftime("%Y-%m-%d")

    normalized_requested: List[str] = []
    if satellites:
        normalized_requested = [
            _normalize_satellite_name(s)
            for s in satellites
            if str(s).strip()
        ]

    params = {
        "estados[]": state_code,
        "data_inicio": date_from,
        "data_fim": date_to,
    }
    if satellites:
        # Algumas versões da API aceitam filtro de satélites via parâmetro em lista.
        params["satelites[]"] = satellites

    url = _build_inpe_url()
    print(f"[INPE] Buscando focos para CE de {date_from} a {date_to}...")

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "dados" in data:
            df = pd.DataFrame(data["dados"])
        else:
            print(f"[INPE] Formato de resposta inesperado: {type(data)}")
            return pd.DataFrame()

        if not df.empty:
            # Padronizar colunas
            col_map = {
                "latitude": "lat",
                "longitude": "lon",
                "data_hora": "datetime",
                "satelite": "satellite",
                "municipio": "municipio",
                "estado": "state",
                "bioma": "bioma",
                "risco_fogo": "fire_risk",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # Fallback local: garante filtro por satélite mesmo se a API ignorar o parâmetro.
            if normalized_requested:
                sat_col = "satellite" if "satellite" in df.columns else None
                if sat_col is None and "satelite" in df.columns:
                    sat_col = "satelite"
                if sat_col:
                    sat_norm = df[sat_col].astype(str).map(_normalize_satellite_name)
                    mask = sat_norm.map(
                        lambda s: any(req in s or s in req for req in normalized_requested)
                    )
                    df = df[mask].copy()

            df["source"] = "INPE"

        print(f"[INPE] {len(df)} focos encontrados.")
        return df

    except requests.exceptions.RequestException as e:
        print(f"[INPE] Erro na requisição: {e}")
        return pd.DataFrame()
    except json.JSONDecodeError as e:
        print(f"[INPE] Erro ao decodificar JSON: {e}")
        return pd.DataFrame()


def fetch_inpe_fire_summary(
    state_code: str = "23", year: int = 2025
) -> pd.DataFrame:
    """
    Baixa sumário estatístico de focos por Município para o Ceará.

    Args:
        state_code: Código IBGE (23 = Ceará)
        year: Ano de interesse

    Returns:
        DataFrame com totais por município
    """
    url = _build_inpe_url("estatisticas/municipios")
    params = {
        "estado": state_code,
        "ano": year,
    }

    print(f"[INPE] Buscando estatísticas por município CE/{year}...")

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and "dados" in data:
            df = pd.DataFrame(data["dados"])
        else:
            return pd.DataFrame()

        if not df.empty:
            df["source"] = "INPE"
            df["year"] = year

        print(f"[INPE] Dados de {len(df)} municípios carregados.")
        return df

    except Exception as e:
        print(f"[INPE] Erro: {e}")
        return pd.DataFrame()


def fetch_goes16_fire_foci_ceara(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> pd.DataFrame:
    """Busca focos GOES no Ceará com prioridade em GOES-16 e fallback GOES-19."""
    df = fetch_inpe_fire_foci(
        state_code="23",
        date_from=date_from,
        date_to=date_to,
        satellites=["GOES-16", "GOES 16", "GOES16"],
    )
    if not df.empty:
        return df
    return fetch_inpe_daily_kml_foci_ceara(satellites=["GOES-16", "GOES 16", "GOES16", "GOES-19"])


def fetch_inpe_daily_kml_foci_ceara(
    satellites: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Fallback de focos pelo KML diário do INPE (dados abertos).

    Nota: GOES-16 foi substituído por GOES-19 em 2025.
    """
    url = "https://dataserver-coids.inpe.br/queimadas/queimadas/focos/kml/focos_diario_web.kml"
    requested = [
        _normalize_satellite_name(s)
        for s in (satellites or [])
        if str(s).strip()
    ]

    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()

        import xml.etree.ElementTree as ET

        ns = {"k": "http://www.opengis.net/kml/2.2"}
        root = ET.fromstring(resp.content)
        rows = []

        for folder in root.findall(".//k:Document/k:Folder/k:Folder", ns):
            sat_folder = (folder.findtext("k:name", default="", namespaces=ns) or "").strip()
            sat_norm = _normalize_satellite_name(sat_folder)

            if requested and not any(req in sat_norm or sat_norm in req for req in requested):
                continue

            for pm in folder.findall("k:Placemark", ns):
                desc = pm.findtext("k:description", default="", namespaces=ns) or ""
                meta = _parse_kml_description(desc)
                coords = (pm.findtext(".//k:coordinates", default="", namespaces=ns) or "").strip()
                if not coords:
                    continue

                parts = [p.strip() for p in coords.split(",")]
                if len(parts) < 2:
                    continue

                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                except ValueError:
                    continue

                if not (
                    CEARA_BBOX["min_lat"] <= lat <= CEARA_BBOX["max_lat"]
                    and CEARA_BBOX["min_lon"] <= lon <= CEARA_BBOX["max_lon"]
                ):
                    continue

                rows.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "datetime": meta.get("data", ""),
                        "satellite": meta.get("satelite", sat_folder),
                        "municipio": meta.get("municipio", ""),
                        "state": meta.get("estado", "CE"),
                        "bioma": "",
                        "source": "INPE_KML",
                    }
                )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["year"] = df["datetime"].dt.year
        df["month"] = df["datetime"].dt.month
        return df

    except requests.exceptions.RequestException as e:
        print(f"[INPE_KML] Erro na requisição: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[INPE_KML] Erro ao processar KML: {e}")
        return pd.DataFrame()


def fetch_firms_active_fires(
    api_key: str = "",
    country: str = "BRA",
    days: int = 7,
) -> pd.DataFrame:
    """
    Baixa dados de focos ativos da NASA FIRMS API.

    Args:
        api_key: Chave da API FIRMS (gratuita em firms.modaps.eosdis.nasa.gov)
        country: Código do país (BRA = Brasil)
        days: Quantidade de dias retroativos

    Returns:
        DataFrame com focos ativos
    """
    if not api_key:
        api_key = os.getenv("FIRMS_API_KEY", "")

    if not api_key:
        print("[FIRMS] Chave de API não configurada. Use $FIRMS_API_KEY")
        print("[FIRMS] Obtenha uma chave gratuita em: firms.modaps.eosdis.nasa.gov")
        return pd.DataFrame()

    # Usar VIIRS para melhor resolução
    source = "VIIRS_SNPP_NRT"

    params = {
        "api_key": api_key,
        "country": country,
        "days": days,
    }

    url = f"{FIRMS_API_URL}{source}/1/{country}/{days}"

    print(f"[FIRMS] Buscando focos ativos para o Brasil ({days}d)...")

    try:
        response = requests.get(url, params={"api_key": api_key}, timeout=20)
        response.raise_for_status()

        csv_content = response.text
        df = pd.read_csv(io.StringIO(csv_content))

        if not df.empty:
            df["source"] = "FIRMS"

            # Filtrar para o Ceará
            if "latitude" in df.columns and "longitude" in df.columns:
                mask = (
                    (df["latitude"] >= CEARA_BBOX["min_lat"])
                    & (df["latitude"] <= CEARA_BBOX["max_lat"])
                    & (df["longitude"] >= CEARA_BBOX["min_lon"])
                    & (df["longitude"] <= CEARA_BBOX["max_lon"])
                )
                df_ce = df[mask].copy()
                print(f"[FIRMS] {len(df)} focos no Brasil, {len(df_ce)} no Ceará")
                return df_ce

        return df

    except Exception as e:
        print(f"[FIRMS] Erro: {e}")
        return pd.DataFrame()


def fetch_firms_ceara(days: int = 3, source: str = "VIIRS_SNPP_NRT") -> pd.DataFrame:
    """
    Focos NASA FIRMS recortados ao bbox do Ceará (requer FIRMS_API_KEY).

    Args:
        days: Janela de dias (1–10)
        source: VIIRS_SNPP_NRT, VIIRS_NOAA20_NRT, MODIS_NRT, etc.

    Returns:
        DataFrame com lat, lon, datetime, satellite, source=FIRMS
    """
    bbox = (
        CEARA_BBOX["min_lon"],
        CEARA_BBOX["min_lat"],
        CEARA_BBOX["max_lon"],
        CEARA_BBOX["max_lat"],
    )
    fires = _fetch_firms_fires_satellite(
        api_key=None,
        source=source,
        day_range=min(max(1, days), 5),
        bbox=bbox,
    )
    if not fires:
        return pd.DataFrame()

    df = pd.DataFrame(fires)
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)
    # acq_time pode vir como 334 ou "0334"
    t = df["acq_time"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(4)
    df["datetime"] = pd.to_datetime(
        df["acq_date"].astype(str) + " " + t.str.slice(0, 2) + ":" + t.str.slice(2, 4),
        errors="coerce",
    )
    df["municipio"] = ""
    df["bioma"] = ""
    df["source"] = "FIRMS"
    if "satellite" not in df.columns:
        df["satellite"] = source
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    return df


def merge_inpe_firms(df_inpe: pd.DataFrame, df_firms: pd.DataFrame) -> pd.DataFrame:
    """Une INPE e FIRMS e remove duplicatas aproximadas (lat/lon/dia)."""
    parts = []
    if df_inpe is not None and not df_inpe.empty:
        parts.append(df_inpe.copy())
    if df_firms is not None and not df_firms.empty:
        parts.append(df_firms.copy())
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    if "datetime" in out.columns:
        day = pd.to_datetime(out["datetime"], errors="coerce").dt.floor("D")
    else:
        day = pd.Series(pd.NaT, index=out.index)
    key = (
        out["lat"].round(3).astype(str)
        + "_"
        + out["lon"].round(3).astype(str)
        + "_"
        + day.astype(str)
    )
    out = out.loc[~key.duplicated(keep="first")].reset_index(drop=True)
    out["source"] = "INPE+FIRMS"
    return out


def load_local_fire_data(path: str) -> pd.DataFrame:
    """
    Carrega dados de focos de um arquivo CSV local.

    Útil para usar dados já baixados sem depender de API.
    """
    if not os.path.exists(path):
        print(f"[LOCAL] Arquivo não encontrado: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"[LOCAL] {len(df)} focos carregados de {path}")
    return df


def update_daily_fire_database(
    output_path: str = "data/focos_ce_diario.csv",
    source: str = "goes",
) -> dict:
    """
    Atualiza base local diária de focos do Ceará (append + dedupe).

    Args:
        output_path: CSV local acumulado
        source: goes | inpe | firms | inpe_firms

    Returns:
        Dict com status da atualização
    """
    today = datetime.now().strftime("%Y-%m-%d")

    if source == "goes":
        df_new = fetch_goes16_fire_foci_ceara(date_from=today, date_to=today)
        source_used = "GOES"
    elif source == "inpe":
        df_new = fetch_inpe_fire_foci(state_code="23", date_from=today, date_to=today)
        source_used = "INPE"
    elif source == "firms":
        df_new = fetch_firms_ceara(days=1)
        source_used = "FIRMS"
    elif source == "inpe_firms":
        df_i = fetch_inpe_fire_foci(state_code="23", date_from=today, date_to=today)
        df_f = fetch_firms_ceara(days=1)
        df_new = merge_inpe_firms(df_i, df_f)
        source_used = "INPE+FIRMS"
    else:
        return {
            "ok": False,
            "error": f"Fonte inválida: {source}",
            "path": output_path,
        }

    if df_new is None:
        df_new = pd.DataFrame()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if os.path.exists(output_path):
        try:
            df_old = pd.read_csv(output_path)
        except Exception:
            df_old = pd.DataFrame()
    else:
        df_old = pd.DataFrame()

    before = len(df_old)
    if df_new.empty and df_old.empty:
        return {
            "ok": True,
            "source": source_used,
            "date": today,
            "added": 0,
            "total": 0,
            "path": output_path,
        }

    merged = pd.concat([df_old, df_new], ignore_index=True) if not df_old.empty else df_new.copy()

    if "datetime" in merged.columns:
        dt = pd.to_datetime(merged["datetime"], errors="coerce")
        dt_str = dt.dt.strftime("%Y-%m-%d %H:%M")
    else:
        dt_str = pd.Series("", index=merged.index)

    lat = pd.to_numeric(merged.get("lat", pd.Series(np.nan, index=merged.index)), errors="coerce").round(4)
    lon = pd.to_numeric(merged.get("lon", pd.Series(np.nan, index=merged.index)), errors="coerce").round(4)
    sat = merged.get("satellite", pd.Series("", index=merged.index)).astype(str)
    key = lat.astype(str) + "_" + lon.astype(str) + "_" + dt_str.astype(str) + "_" + sat

    merged = merged.loc[~key.duplicated(keep="first")].reset_index(drop=True)
    merged.to_csv(output_path, index=False)

    return {
        "ok": True,
        "source": source_used,
        "date": today,
        "added": max(0, len(merged) - before),
        "total": len(merged),
        "path": output_path,
    }


def download_year_data(year: int, output_dir: str = "data") -> pd.DataFrame:
    """
    Baixa dados de um ano inteiro para o Ceará em blocos mensais.

    Args:
        year: Ano (ex: 2024)
        output_dir: Diretório de saída

    Returns:
        DataFrame completo do ano
    """
    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []

    for month in tqdm(range(1, 13), desc=f"Baixando {year}"):
        date_from = f"{year}-{month:02d}-01"
        # Último dia do mês
        if month == 12:
            date_to = f"{year}-12-31"
        else:
            date_to = f"{year}-{month+1:02d}-01"
            # Subtrair 1 dia
            from datetime import date as dt_date
            from datetime import timedelta as td

            d = dt_date.fromisoformat(date_to) - td(days=1)
            date_to = d.strftime("%Y-%m-%d")

        df_month = fetch_inpe_fire_foci(
            state_code="23",
            date_from=date_from,
            date_to=date_to,
        )

        if not df_month.empty:
            all_dfs.append(df_month)

        # Salvar mensalmente
        if not df_month.empty:
            path = os.path.join(output_dir, f"focos_CE_{year}_{month:02d}.csv")
            df_month.to_csv(path, index=False)

    if all_dfs:
        df_year = pd.concat(all_dfs, ignore_index=True)
        output_path = os.path.join(output_dir, f"focos_CE_{year}_completo.csv")
        df_year.to_csv(output_path, index=False)
        print(f"\n[DOWNLOAD] Ano {year} completo: {len(df_year)} focos salvos em {output_path}")
        return df_year

    print(f"[DOWNLOAD] Nenhum dado encontrado para {year}")
    return pd.DataFrame()
