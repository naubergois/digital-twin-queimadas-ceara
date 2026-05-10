"""
Agente de inteligência — agrega notícias (RSS), sinais públicos em redes
e contexto para detecção com imagens de satélite (NASA GIBS / INPE).

Fontes sem API paga: RSS abertos, Google News RSS, busca pública Reddit.
Conteúdo social e de notícias não é validado; uso apenas como sinal auxiliar.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import feedparser
import requests

from config.ceara_config import CEARA_BBOX
from src.satellite import GIBS_LAYERS, gibs_tile_url

DEFAULT_UA = (
    "DigitalTwinQueimadasCE/1.0 (monitoramento ambiental; educação; +https://github.com/)"
)

HEADERS = {"User-Agent": DEFAULT_UA, "Accept": "application/rss+xml, application/json, */*"}

FIRE_HINTS = (
    "queimad",
    "incêndio",
    "incendio",
    "fogo",
    "desmat",
    "desmate",
    "calor",
    "sec",
    "cerrado",
    "caatinga",
    "ambient",
    "clima",
    "satélite",
    "satelite",
    "inpe",
)

RSS_SOURCES: list[tuple[str, str, bool]] = [
    ("G1 — Ceará", "https://g1.globo.com/rss/g1/ceara/", True),
    ("Agência Brasil", "https://agenciabrasil.ebc.com.br/rss/ultimasnoticias/feed.xml", False),
]


def _mentions_fire(text: str) -> bool:
    t = text.lower()
    return any(h in t for h in FIRE_HINTS)


def _mentions_ceara(text: str) -> bool:
    t = text.lower()
    return "ceará" in t or "ceara" in t or "nordeste" in t


def parse_rss(
    feed_url: str,
    max_items: int = 15,
    require_ceara_if_fire: bool = False,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        parsed = feedparser.parse(feed_url, agent=DEFAULT_UA)
        entries = getattr(parsed, "entries", []) or []
        for e in entries:
            title = getattr(e, "title", "") or ""
            summary = getattr(e, "summary", "") or getattr(e, "description", "") or ""
            blob = title + " " + summary
            if not _mentions_fire(blob):
                continue
            if require_ceara_if_fire and not _mentions_ceara(blob):
                continue
            link = getattr(e, "link", "") or ""
            pub = getattr(e, "published", "") or getattr(e, "updated", "") or ""
            out.append({
                "title": title.strip(),
                "link": link,
                "published": pub,
                "summary": summary[:400],
                "feed_url": feed_url,
            })
            if len(out) >= max_items:
                break
    except Exception as ex:  # noqa: BLE001
        return [{"title": "Falha ao ler RSS", "link": feed_url, "error": str(ex), "published": ""}]
    return out


def fetch_google_news_ce(max_items: int = 14) -> list[dict[str, Any]]:
    q = quote_plus("queimadas OR incêndio OR fogo florestal Ceará")
    url = f"https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    try:
        parsed = feedparser.parse(url, agent=DEFAULT_UA)
        entries = getattr(parsed, "entries", []) or []
        rows = []
        for e in entries[:max_items]:
            rows.append({
                "title": getattr(e, "title", "") or "",
                "link": getattr(e, "link", "") or "",
                "published": getattr(e, "published", "") or "",
                "source": "Google News RSS",
            })
        return rows
    except Exception as ex:  # noqa: BLE001
        return [{"title": "Google News indisponível", "link": "", "error": str(ex)}]


def fetch_curated_rss(max_per_feed: int = 10) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, feed_url, strict_ce in RSS_SOURCES:
        items = parse_rss(feed_url, max_items=max_per_feed, require_ceara_if_fire=strict_ce)
        for it in items:
            if "error" in it:
                it["feed_label"] = label
                rows.append(it)
                break
            it["feed_label"] = label
            rows.append(it)
    return rows[:30]


def fetch_reddit_posts(query: str = "queimadas Ceará", limit: int = 12) -> list[dict[str, Any]]:
    q = quote_plus(query)
    url = f"https://www.reddit.com/search.json?q={q}&sort=new&limit={limit}&raw_json=1"
    try:
        r = requests.get(url, headers=HEADERS, timeout=22)
        r.raise_for_status()
        data = r.json()
        posts = []
        for child in data.get("data", {}).get("children", []):
            d = child.get("data", {}) or {}
            posts.append({
                "title": d.get("title") or "",
                "url": "https://www.reddit.com" + (d.get("permalink") or ""),
                "subreddit": d.get("subreddit") or "",
                "selftext": (d.get("selftext") or "")[:500],
            })
        return posts
    except Exception as ex:  # noqa: BLE001
        return [{"title": "Reddit indisponível ou limitado", "url": "", "error": str(ex)}]


def satellite_detection_context() -> dict[str, Any]:
    """Camadas GIBS úteis para detecção (térmicas / anomalias)."""
    layers = []
    for key, meta in GIBS_LAYERS.items():
        desc = meta.get("description", "")
        if any(
            x in key.lower() or x in desc.lower()
            for x in ("thermal", "fires", "anomal", "brightness", "termal")
        ):
            layers.append({
                "id": key,
                "description": desc,
                "resolution": meta["resolution"],
                "delay_days": meta["delay_days"],
                "wmts_template": gibs_tile_url(key),
            })
    return {
        "bbox_ceara": CEARA_BBOX,
        "gibs_detection_layers": layers,
        "hint": (
            "Sobreponha **anomalias térmicas VIIRS** ou **MODIS banda termal** aos focos INPE/FIRMS "
            "no mapa do dashboard para cruzar detecção por imagem com pontos oficiais."
        ),
    }


class FireIntelAgent:
    """Orquestra coleta e devolve um pacote único para o Streamlit."""

    def run(
        self,
        focos_count: int | None = None,
        firms_key_configured: bool = False,
    ) -> dict[str, Any]:
        generated = datetime.now().isoformat(timespec="seconds")
        news_google = fetch_google_news_ce()
        news_rss = fetch_curated_rss()
        social = fetch_reddit_posts()
        satellite = satellite_detection_context()

        summary_md = "\n".join([
            f"- **Consulta:** {generated}",
            f"- **Focos carregados no painel:** {focos_count if focos_count is not None else '—'}",
            f"- **Chave FIRMS (NASA):** {'definida' if firms_key_configured else 'ausente'}",
            "- **Notícias:** Google News RSS + RSS (G1 CE, Agência Brasil), filtrados por termos de fogo/ambiente.",
            "- **Redes:** busca pública no Reddit (sem login); pode haver ruído ou bloqueio ocasional.",
            "- **Satélite:** use camadas GIBS listadas abaixo como **detecção por imagem**; confirme com INPE.",
        ])

        return {
            "generated_at": generated,
            "summary_md": summary_md,
            "news_google": news_google,
            "news_rss": news_rss,
            "social_reddit": social,
            "satellite": satellite,
        }
