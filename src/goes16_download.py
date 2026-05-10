"""
Download de produtos GOES-16 (ABI) a partir do bucket público AWS Open Data.

Fonte: ``noaa-goes16`` (região ``us-east-1``), sem credenciais — ver
https://registry.opendata.aws/noaa-goes/

Estrutura típica de chave S3::

    {PRODUTO}/{ANO}/{DIA_JULIANO}/{HH}/OR_ABI-..._G16_....nc

Exemplos de produto:
  - ABI-L2-CMIPF — Cloud and Moisture Imagery, disco completo (um ficheiro por banda)
  - ABI-L1b-RadF — radiâncias L1b, disco completo (ficheiros maiores)
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
except ImportError as e:  # pragma: no cover
    raise ImportError("Instale boto3: pip install boto3") from e


# Canal/tipo no nome do ficheiro CMIPF: ...-M6C13_... (modo 6, canal 13)
_CMIPF_CHANNEL_RE = re.compile(r"M\d+C(\d{2})")


@dataclass
class GOES16AWSSettings:
    bucket: str = "noaa-goes16"
    region: str = "us-east-1"
    satellite_token: str = "G16"


@dataclass
class GOES16DownloadConfig:
    """Configuração de cache local e cliente S3."""

    settings: GOES16AWSSettings = field(default_factory=GOES16AWSSettings)
    cache_dir: Path = field(default_factory=lambda: Path("data/goes16_raw"))

    def client(self):
        return boto3.client(
            "s3",
            region_name=self.settings.region,
            config=Config(signature_version=UNSIGNED),
        )


def hourly_prefix(product: str, when: datetime) -> str:
    """Prefixo S3 para uma hora civil (UTC)."""
    when = _ensure_utc(when)
    doy = when.timetuple().tm_yday
    return f"{product.strip('/')}/{when.year}/{doy:03d}/{when.hour:02d}/"


def list_keys_for_prefix(cfg: GOES16DownloadConfig, prefix: str) -> List[str]:
    """Lista todas as chaves sob ``prefix`` (paginação automática)."""
    keys: List[str] = []
    paginator = cfg.client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=cfg.settings.bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            keys.append(obj["Key"])
    return keys


def filter_cmipf_channel(keys: Iterable[str], channel: int) -> List[str]:
    """Mantém apenas chaves ABI-L2-CMIPF da banda indicada (ex.: 13)."""
    out: List[str] = []
    ch = f"{int(channel):02d}"
    for k in keys:
        if "ABI-L2-CMIPF" not in k or not k.endswith(".nc"):
            continue
        m = _CMIPF_CHANNEL_RE.search(k)
        if m and m.group(1) == ch:
            out.append(k)
    return sorted(out)


def parse_goes_start_time_from_key(key: str) -> Optional[datetime]:
    """
    Lê o instante inicial do segmento a partir do campo ``_s..._`` no nome
    (ano, dia juliano, hora, minuto, segundo, décimo de segundo).
    """
    base = key.split("/")[-1]
    m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)(?:_|\.)", base)
    if not m:
        return None
    year, doy, hh, mm, ss, sec_tenth = map(int, m.groups())
    try:
        t0 = datetime(year, 1, 1, hh, mm, ss, tzinfo=timezone.utc) + timedelta(
            days=doy - 1, milliseconds=100 * sec_tenth
        )
    except ValueError:
        return None
    return t0


def pick_nearest_cmipf(keys: List[str], when: datetime) -> Optional[str]:
    """
    Escolhe o ficheiro cujo instante inicial (campo ``s`` no nome) está mais
    próximo de ``when`` (UTC).
    """
    when = _ensure_utc(when)
    if not keys:
        return None
    best_key: Optional[str] = None
    best_dist: Optional[float] = None
    for k in keys:
        t0 = parse_goes_start_time_from_key(k)
        if t0 is None:
            continue
        dist = abs((t0 - when).total_seconds())
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_key = k
    return best_key


def download_key(
    cfg: GOES16DownloadConfig,
    key: str,
    dest_dir: Optional[Path] = None,
    *,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Descarrega uma chave para ``cache_dir`` (ou ``dest_dir``), preservando o nome do ficheiro.

    Se o ficheiro já existir e ``overwrite`` for False, devolve o caminho sem voltar a baixar.
    """
    dest_root = dest_dir if dest_dir is not None else cfg.cache_dir
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    local = dest_root / key.split("/")[-1]
    if local.is_file() and not overwrite:
        return local.resolve()

    client = cfg.client()

    if show_progress:
        try:
            from tqdm import tqdm

            head = client.head_object(Bucket=cfg.settings.bucket, Key=key)
            total = int(head["ContentLength"])

            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=local.name[:40],
            ) as pbar:

                def _cb(n: int) -> None:
                    pbar.update(n)

                client.download_file(
                    cfg.settings.bucket,
                    key,
                    str(local),
                    Callback=_cb,
                )
        except ImportError:
            client.download_file(cfg.settings.bucket, key, str(local))
    else:
        client.download_file(cfg.settings.bucket, key, str(local))

    return local.resolve()


def download_cmipf_channel(
    when: datetime,
    channel: int,
    cfg: Optional[GOES16DownloadConfig] = None,
    *,
    dest_dir: Optional[Path] = None,
    overwrite: bool = False,
    show_progress: bool = True,
) -> Path:
    """
    Localiza e descarrega **ABI-L2-CMIPF** para a banda ``channel`` na hora de ``when`` (UTC).

    Útil para imagens térmicas (ex. canal **13** ~ 10,3 µm, ou **14** ~ 11,2 µm).
    """
    cfg = cfg or GOES16DownloadConfig()
    prefix = hourly_prefix("ABI-L2-CMIPF", when)
    keys = filter_cmipf_channel(list_keys_for_prefix(cfg, prefix), channel)
    key = pick_nearest_cmipf(keys, when)
    if key is None:
        raise FileNotFoundError(
            f"Nenhum CMIPF canal {channel} em s3://{cfg.settings.bucket}/{prefix}"
        )
    return download_key(cfg, key, dest_dir=dest_dir, overwrite=overwrite, show_progress=show_progress)


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_when(s: str) -> datetime:
    """ISO 8601 em UTC (ex.: 2024-10-31T18:10:00 ou 2024-10-31T18:10:00Z)."""
    t = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(t)
    return _ensure_utc(dt)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Download GOES-16 (NOAA Open Data, S3)")
    p.add_argument(
        "--when",
        required=True,
        help="Instante aproximado em UTC (ISO), ex.: 2024-10-31T18:10:00Z",
    )
    p.add_argument(
        "--channel",
        type=int,
        default=13,
        help="Banda CMIPF (ABI-L2-CMIPF), padrão 13 (limpa infravermelho)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Pasta de destino (padrão: data/goes16_raw no cwd)",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    cfg = GOES16DownloadConfig()
    when = _parse_when(args.when)
    path = download_cmipf_channel(
        when,
        args.channel,
        cfg=cfg,
        dest_dir=args.out_dir,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )
    print(path)


if __name__ == "__main__":
    main()
