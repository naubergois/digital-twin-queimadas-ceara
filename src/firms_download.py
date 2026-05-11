"""
Descarrega detecções VIIRS Active Fire via **NASA FIRMS API**.

Requer ``FIRMS_API_KEY`` em variável de ambiente. Em https://firms.modaps.eosdis.nasa.gov/api/
o utilizador regista-se gratuitamente e recebe a chave.

Endpoint base (Area API, CSV):

  https://firms.modaps.eosdis.nasa.gov/api/area/csv/{API_KEY}/{SOURCE}/{AREA}/{DAY_RANGE}/{DATE}

Onde ``SOURCE`` é, por exemplo, ``VIIRS_SNPP_NRT``, ``VIIRS_NOAA20_NRT``,
``VIIRS_NOAA21_NRT`` ou ``MODIS_NRT``.

Em modo offline (sem chave), o módulo expõe ``offline_demo_viirs`` que
gera um proxy sintético — útil para correr testes e ver a fusão a operar
sem fazer chamadas externas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.multi_sensor_fusion import load_viirs_firms_csv


FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


@dataclass
class FIRMSRequest:
    source: str = "VIIRS_NOAA20_NRT"
    """Outras opções: VIIRS_SNPP_NRT, VIIRS_NOAA21_NRT, MODIS_NRT."""
    bbox: tuple = (-41.5, -7.9, -37.0, -2.5)  # west, south, east, north
    """Ordem FIRMS: west, south, east, north (note: diferente do CEARA_BBOX)."""
    day: Optional[date] = None
    range_days: int = 1


def firms_url(req: FIRMSRequest, api_key: str) -> str:
    w, s, e, n = req.bbox
    area = f"{w},{s},{e},{n}"
    days = max(1, min(int(req.range_days), 10))
    date_str = (req.day or date.today()).isoformat()
    return f"{FIRMS_BASE}/{api_key}/{req.source}/{area}/{days}/{date_str}"


def download_firms_csv(
    req: FIRMSRequest,
    out_path: Path,
    *,
    api_key: Optional[str] = None,
    timeout: float = 30.0,
) -> Path:
    """Faz GET ao FIRMS e grava CSV em ``out_path`` — devolve o caminho final."""
    key = api_key or os.environ.get("FIRMS_API_KEY")
    if not key:
        raise RuntimeError(
            "FIRMS_API_KEY não definida. Regista-te em https://firms.modaps.eosdis.nasa.gov/api/"
        )
    url = firms_url(req, key)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    out_path.write_bytes(r.content)
    return out_path.resolve()


def load_or_download_viirs(
    day_utc: date,
    bbox: dict,
    cache_dir: Path,
    *,
    source: str = "VIIRS_NOAA20_NRT",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Atalho: tenta CSV local em cache; se não existir e houver
    ``FIRMS_API_KEY``, descarrega; senão devolve DataFrame vazio.
    """
    cache = Path(cache_dir) / f"firms_{source}_{day_utc.isoformat()}.csv"
    if cache.is_file():
        return load_viirs_firms_csv(cache)
    if api_key or os.environ.get("FIRMS_API_KEY"):
        req = FIRMSRequest(
            source=source,
            bbox=(bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]),
            day=day_utc,
            range_days=1,
        )
        try:
            p = download_firms_csv(req, cache, api_key=api_key)
            return load_viirs_firms_csv(p)
        except Exception as e:  # pragma: no cover
            print(f"[firms_download] aviso: download falhou: {e}")
    return pd.DataFrame(columns=["lat", "lon", "datetime", "confidence"])


def offline_demo_viirs(
    truth_focos: pd.DataFrame,
    bbox: dict,
    *,
    detection_rate: float = 0.8,
    spatial_jitter_km: float = 1.5,
    false_positive_rate: float = 0.01,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Atalho de conveniência: encaminha para ``synthesize_viirs_proxy`` para
    demonstrar a fusão sem chamar a API. *Não usar como verdade* — em
    avaliação real a VIIRS deve vir do FIRMS.
    """
    from src.multi_sensor_fusion import synthesize_viirs_proxy

    return synthesize_viirs_proxy(
        truth_focos,
        bbox=bbox,
        detection_rate=detection_rate,
        spatial_jitter_km=spatial_jitter_km,
        false_positive_rate=false_positive_rate,
        seed=seed,
    )
