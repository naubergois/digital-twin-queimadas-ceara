"""
Módulo de coleta de dados de queimadas de fontes abertas (INPE, NASA FIRMS).
"""

import csv
import io
import json
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd
import requests
from tqdm import tqdm

from config.ceara_config import (
    CEARA_BBOX,
    INPE_API_URL,
    FIRMS_API_URL,
    SATELITES,
)


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

    params = {
        "estados[]": state_code,
        "data_inicio": date_from,
        "data_fim": date_to,
    }

    url = f"{INPE_API_URL}focos"
    print(f"[INPE] Buscando focos para CE de {date_from} a {date_to}...")

    try:
        response = requests.get(url, params=params, timeout=60)
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
    url = f"{INPE_API_URL}estatisticas/municipios"
    params = {
        "estado": state_code,
        "ano": year,
    }

    print(f"[INPE] Buscando estatísticas por município CE/{year}...")

    try:
        response = requests.get(url, params=params, timeout=60)
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
        response = requests.get(url, params={"api_key": api_key}, timeout=120)
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
