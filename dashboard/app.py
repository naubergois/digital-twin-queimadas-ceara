"""
Dashboard — foco em resultados ST-HyperNet (métricas por dia + figuras).

Executar: streamlit run dashboard/app.py
(recomendado a partir da raiz do repositório)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.ceara_config import CEARA_BBOX
from src.fire_data import (
    fetch_firms_ceara,
    fetch_goes16_fire_foci_ceara,
    fetch_inpe_fire_foci,
    load_local_fire_data,
    merge_inpe_firms,
    update_daily_fire_database,
)
from src.compare_ceara_maps import build_ceara_folium_real_pred_goes
from src.intel_agent import FireIntelAgent
from src.satellite import available_satellite_sources, gibs_tile_url, satellite_layer_for_folium

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _safe_metrics_base(rel: str) -> Path:
    """Resolve pasta de métricas dentro do repositório (sem path traversal)."""
    rel = (rel or "").strip().strip("/")
    base = (PROJECT_ROOT / rel).resolve()
    root = PROJECT_ROOT.resolve()
    if root != base and root not in base.parents:
        raise ValueError("Caminho fora do repositório.")
    return base


# ---------------------------------------------------------------------------
# ST-HyperNet: carregar métricas e figuras em disco
# ---------------------------------------------------------------------------


def discover_metrics_directories() -> List[str]:
    """Pastas sob `data/` que contêm `metrics_by_day.json`."""
    data_dir = PROJECT_ROOT / "data"
    if not data_dir.is_dir():
        return []
    found: List[str] = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "metrics_by_day.json").is_file():
            found.append(str(p.relative_to(PROJECT_ROOT)))
    return found


def resolve_project_path(path_str: str | None) -> Optional[Path]:
    if not path_str or not str(path_str).strip():
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p if p.is_file() else None
    cand = PROJECT_ROOT / path_str
    return cand if cand.is_file() else None


@st.cache_data(ttl=120)
def load_st_metrics_bundle(metrics_dir_rel: str) -> Optional[Dict[str, Any]]:
    """Lê `metrics_by_day.json` (e agregados se existirem)."""
    try:
        base = _safe_metrics_base(metrics_dir_rel)
    except ValueError:
        return None
    jpath = base / "metrics_by_day.json"
    if not jpath.is_file():
        return None
    with open(jpath, encoding="utf-8") as f:
        bundle = json.load(f)
    bundle["_dir"] = str(base)
    agg_path = base / "metrics_aggregate.json"
    if agg_path.is_file():
        with open(agg_path, encoding="utf-8") as f:
            bundle["_metrics_aggregate_file"] = json.load(f)
    return bundle


@st.cache_data(ttl=120)
@st.cache_data(ttl=7200)
def fetch_goes16_ceara_single_day(day_iso: str) -> pd.DataFrame:
    """Focos GOES-16/-19 (INPE) só para o dia civil indicado (cache 2 h)."""
    try:
        return fetch_goes16_fire_foci_ceara(date_from=day_iso, date_to=day_iso)
    except Exception:
        return pd.DataFrame()


def load_case_explanations(metrics_dir_rel: str) -> Optional[Dict[str, Any]]:
    try:
        base = _safe_metrics_base(metrics_dir_rel)
    except ValueError:
        return None
    p = base / "case_explanations.json"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def render_st_hypernet_tab(metrics_dir_rel: str) -> None:
    bundle = load_st_metrics_bundle(metrics_dir_rel)
    if not bundle:
        st.warning(
            f"Não há `metrics_by_day.json` em `{metrics_dir_rel}`. "
            "Gere com: `python -m src.compare_st_hypernet_days --help`"
        )
        return

    rows = bundle.get("metrics_by_day") or []
    if not rows:
        st.warning("O JSON não contém entradas em `metrics_by_day`.")
        return

    st.caption(bundle.get("caveat") or "")
    st.caption(
        f"Técnica: **{bundle.get('technique', '—')}** · Cubo: "
        f"{bundle.get('cube_time_span', {}).get('start', '?')} → "
        f"{bundle.get('cube_time_span', {}).get('end', '?')}"
    )
    st.caption(
        "Métrica principal deste painel: **precisão** (quanto das células previstas coincidem com queimada real). "
        "Recall (cobertura) aparece só como contexto."
    )

    dfm = pd.DataFrame(rows)
    dfm["date"] = pd.to_datetime(dfm["date"], errors="coerce")
    dfm = dfm.sort_values("date")

    agg = bundle.get("_metrics_aggregate_file") or {}
    if agg:
        a1, a2, a3, a4, a5 = st.columns(5)
        a1.metric("Dias avaliados", int(agg.get("n_days", len(rows))))
        a2.metric("Precisão média", f"{float(agg.get('mean_precision', 0)):.3f}")
        a3.metric("Micro precisão", f"{float(agg.get('micro_precision', 0)):.3f}")
        a4.metric("IoU médio", f"{float(agg.get('mean_iou', 0)):.3f}")
        _tfp = agg.get("total_fp")
        if _tfp is None and "fp" in dfm.columns:
            _tfp = int(dfm["fp"].sum())
        a5.metric("FP totais (grade)", int(_tfp or 0))
    else:
        st.caption(f"{len(rows)} dias no ficheiro (sem `metrics_aggregate.json`).")

    st.subheader("Precisão e IoU ao longo do tempo")
    chart_main = dfm.set_index("date")[["precision", "iou"]].copy()
    st.line_chart(chart_main, use_container_width=True)
    with st.expander("Recall (cobertura) — métrica secundária", expanded=False):
        st.line_chart(dfm.set_index("date")[["recall"]], use_container_width=True)
        if agg:
            st.caption(
                f"Recall médio: {float(agg.get('mean_recall', 0)):.3f} · "
                f"micro-recall: {float(agg.get('micro_recall', 0)):.3f}"
            )

    dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in dfm["date"].dropna()]
    default_ix = len(dates) - 1
    pick = st.selectbox("Dia para inspecionar", options=dates, index=default_ix)
    row = dfm[dfm["date"] == pd.Timestamp(pick)].iloc[0].to_dict()
    # normalizar date para string
    row["date"] = pick

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Precisão", f"{float(row.get('precision', 0)):.3f}")
    m2.metric("IoU", f"{float(row.get('iou', 0)):.3f}")
    m3.metric("FP / FN", f"{int(row.get('fp', 0))} / {int(row.get('fn', 0))}")
    m4.metric("TP", int(row.get("tp", 0)))
    m5.metric("Focos (INPE)", int(row.get("n_focos", 0)))
    m6.metric("Limiar (adapt.)", f"{float(row.get('pred_threshold', 0)):.3f}")
    m7.metric("Recall (sec.)", f"{float(row.get('recall', 0)):.3f}")
    st.caption(
        "Precisão = TP/(TP+FP): penaliza alarmes falsos. "
        f"Frames cubo neste dia: **{int(row.get('n_cube_frames_that_day', 0))}**"
    )

    meta = row.get("adaptive_threshold_meta")
    if isinstance(meta, dict) and meta:
        with st.expander("Metadados do limiar adaptativo", expanded=False):
            st.json(meta)

    cimg1, cimg2 = st.columns(2)
    fig_path = resolve_project_path(row.get("figure"))
    map_png_path = resolve_project_path(row.get("map_png"))
    with cimg1:
        st.markdown("**Real vs previsto (grade)**")
        if fig_path and fig_path.is_file():
            st.image(str(fig_path), use_container_width=True)
        else:
            st.caption("Figura não encontrada em disco.")
    with cimg2:
        st.markdown("**Mapa Ceará (reais vs previstos)**")
        if map_png_path and map_png_path.is_file():
            st.image(str(map_png_path), use_container_width=True)
        else:
            st.caption("Mapa PNG não encontrado.")

    html_path = resolve_project_path(row.get("map_html"))
    if html_path and html_path.is_file():
        with st.expander("Mapa interativo (HTML exportado na comparação)", expanded=False):
            st.caption(str(html_path))
            st.link_button("Abrir HTML no browser (path local)", f"file://{html_path}")

    st.subheader("Mapa reconstruído — real vs previsto + GOES-16 (INPE)")
    rl = row.get("map_real_latlon")
    pl = row.get("map_pred_latlon_score")
    if isinstance(rl, list) and isinstance(pl, list):
        bg = st.radio(
            "Imagem de fundo",
            options=("esri_satellite", "gibs_viirs_case_day"),
            format_func=lambda x: (
                "Satélite ESRI (World Imagery)"
                if x == "esri_satellite"
                else "VIIRS True Color (NASA GIBS, data do caso)"
            ),
            horizontal=True,
            key=f"st_map_bg_{pick}",
        )
        basemap_override = None
        if bg == "gibs_viirs_case_day":
            scene = pd.Timestamp(pick).to_pydatetime()
            basemap_override = (
                gibs_tile_url("viirs_snpp_truecolor", scene),
                "NASA GIBS VIIRS Suomi NPP",
                {"max_zoom": 9, "name": "VIIRS", "opacity": 0.92},
            )

        with st.spinner("A carregar detecções GOES-16/-19 (INPE) para este dia…"):
            dfg = fetch_goes16_ceara_single_day(pick)
        goes_pts: List[tuple[float, float]] = []
        if not dfg.empty and "lat" in dfg.columns and "lon" in dfg.columns:
            sub = dfg.dropna(subset=["lat", "lon"])
            for _, r in sub.head(4000).iterrows():
                goes_pts.append((float(r["lat"]), float(r["lon"])))

        thr_vis = float(row.get("pred_display_threshold", row.get("pred_threshold", 0.0)) or 0.0)
        st.caption(
            "**Vermelho:** focos reais usados na comparação desse dia. **Verde:** células previstas (ST-HyperNet, máscara de visualização). "
            "**Laranja:** detecções térmicas GOES no catálogo INPE (GOES-16 ou GOES-19 consoante o serviço). "
            f"Limiar de visualização do previsto ≈ **{thr_vis:.3f}**."
        )
        try:
            m_live = build_ceara_folium_real_pred_goes(
                real_latlon=rl,
                pred_latlon_score=pl,
                goes_latlon=goes_pts or None,
                pred_layer_label=f"Previsto ST-HyperNet (vis. ≥ {thr_vis:.2f})",
                basemap_id="esri_satellite",
                basemap_override=basemap_override,
            )
            render_folium_map(m_live, height=560)
        except Exception as ex:
            st.warning(f"Não foi possível montar o mapa interactivo: {ex}")
        if dfg.empty:
            st.caption(
                "Sem pontos GOES para este dia na API INPE (rede ou satélite). O mapa mostra só real + previsto."
            )
    else:
        st.info(
            "Este ficheiro de métricas ainda não tem `map_real_latlon` / `map_pred_latlon_score`. "
            "Execute de novo `python -m src.compare_st_hypernet_days ...` para regenerar o JSON e o dashboard poder reconstruir o mapa."
        )

    expl = load_case_explanations(metrics_dir_rel)
    if expl and expl.get("cases"):
        by_date = {c.get("date"): c for c in expl["cases"]}
        hit = by_date.get(pick)
        if hit and hit.get("explanation_pt"):
            st.subheader("Interpretação (DeepSeek)")
            st.markdown(hit["explanation_pt"])
        else:
            st.caption(
                "Existe `case_explanations.json` mas sem entrada para este dia. "
                "Gere com `python -m src.compare_case_explainer_agent --dates "
                f"{pick} ...`"
            )
    else:
        st.caption(
            "Opcional: gere `case_explanations.json` com "
            "`python -m src.compare_case_explainer_agent` (chave `DEEPSEEK_API_KEY`) "
            "para texto interpretativo por dia."
        )


def render_intel_agent_section(focos_n: int) -> None:
    st.caption(
        "Consulta RSS abertos, Google News e Reddit; conteúdo de terceiros sem validação."
    )
    with st.spinner("A carregar…"):
        pack = FireIntelAgent().run(
            focos_count=focos_n,
            firms_key_configured=bool(os.getenv("FIRMS_API_KEY", "").strip()),
        )
    st.markdown(pack["summary_md"])


def render_folium_map(map_obj, height: int) -> None:
    components.html(map_obj.get_root().render(), height=height, scrolling=False)


def _extract_available_dates(df_in: pd.DataFrame) -> list[str]:
    if df_in is None or df_in.empty or "datetime" not in df_in.columns:
        return []
    dt = pd.to_datetime(df_in["datetime"], errors="coerce").dropna()
    if dt.empty:
        return []
    return sorted(dt.dt.strftime("%Y-%m-%d").unique().tolist())


# ============================================================================
# Página
# ============================================================================

st.set_page_config(
    page_title="ST-HyperNet — Queimadas CE",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ST-HyperNet — Queimadas no Ceará")
st.markdown(
    "Painel centrado no **modelo espacial-temporal** (cubo de scores vs grade de focos reais). "
    "A segunda aba mantém **focos de referência** no mapa."
)

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.header("Resultados do modelo")
_discovered = discover_metrics_directories()
_default_st = "data/st_hypernet_2024_all_fire_days"
_opts = list(_discovered)
if (PROJECT_ROOT / _default_st / "metrics_by_day.json").is_file() and _default_st not in _opts:
    _opts.insert(0, _default_st)

if _opts:
    ix = _opts.index(_default_st) if _default_st in _opts else 0
    metrics_dir_rel = st.sidebar.selectbox(
        "Pasta com `metrics_by_day.json`",
        options=_opts,
        index=ix,
        help="Saída típica de `compare_st_hypernet_days` (PNG + JSON por dia).",
    )
else:
    metrics_dir_rel = st.sidebar.text_input(
        "Caminho da pasta (relativo à raiz do repo)",
        value=_default_st,
        help="Ex.: data/st_hypernet_2024_all_fire_days",
    )

if st.sidebar.button("Recarregar ficheiros do modelo", help="Limpa cache de JSON/imagens"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Focos (mapa de referência)")

date_range = st.sidebar.date_input(
    "Período",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
)

if isinstance(date_range, (tuple, list)):
    if len(date_range) >= 2:
        date_from, date_to = date_range[0], date_range[1]
    elif len(date_range) == 1:
        date_from = date_to = date_range[0]
    else:
        date_from = date_to = datetime.now().date()
else:
    date_from = date_to = date_range

_firms_key_ok = bool(os.getenv("FIRMS_API_KEY", "").strip())
_daily_db_path = str(PROJECT_ROOT / "data" / "focos_ce_diario.csv")
_local_csv_path = str(PROJECT_ROOT / "data" / "focos_CE_GOES16_2024.csv")
_available_sources: list[str] = []
if os.path.isfile(_local_csv_path):
    _available_sources.append("Dados Locais (CSV)")
if os.path.isfile(_daily_db_path):
    _available_sources.append("Banco diário (auto)")
_available_sources += ["INPE (online)", "GOES-16 (INPE online)"]
if _firms_key_ok:
    _available_sources += ["NASA FIRMS (online)", "INPE + FIRMS (online)"]

data_source = st.sidebar.selectbox(
    "Fonte de focos",
    _available_sources,
    help="CSV local é o mais rápido para explorar o mapa.",
)
firms_days = 3
if data_source in ("NASA FIRMS (online)", "INPE + FIRMS (online)"):
    firms_days = st.sidebar.slider("Janela FIRMS (dias)", 1, 5, 3)

with st.sidebar.expander("Atualização diária (opcional)", expanded=False):
    daily_update_source = st.selectbox(
        "Fonte",
        options=["goes", "inpe", "firms", "inpe_firms"],
        format_func=lambda x: {
            "goes": "GOES (INPE/KML)",
            "inpe": "INPE",
            "firms": "NASA FIRMS",
            "inpe_firms": "INPE + FIRMS",
        }[x],
        key="daily_src",
    )
    if st.button("Atualizar banco diário agora"):
        with st.spinner("A atualizar…"):
            info = update_daily_fire_database(output_path=_daily_db_path, source=daily_update_source)
        if info.get("ok"):
            st.success(f"+{info.get('added', 0)} registos (total {info.get('total', 0)})")
            st.cache_data.clear()
            st.session_state.df = pd.DataFrame()
        else:
            st.error(info.get("error", "Falha"))

sat_sources = available_satellite_sources()
tile_sources = [s for s in sat_sources if s["type"] == "Tile"]
sat_choice = st.sidebar.selectbox(
    "Camada de satélite (mapa)",
    options=[s["id"] for s in tile_sources],
    format_func=lambda x: next(s["name"] for s in tile_sources if s["id"] == x),
    index=0,
)

# ============================================================================
# Estado e carregamento de focos
# ============================================================================

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "available_dates_current_source" not in st.session_state:
    st.session_state.available_dates_current_source = []
if "last_data_source" not in st.session_state:
    st.session_state.last_data_source = data_source
elif st.session_state.last_data_source != data_source:
    st.session_state.last_data_source = data_source
    st.session_state.df = pd.DataFrame()
    st.cache_data.clear()

dr_key = (pd.Timestamp(date_from).date().isoformat(), pd.Timestamp(date_to).date().isoformat())
if "last_date_range" not in st.session_state:
    st.session_state.last_date_range = dr_key
elif st.session_state.last_date_range != dr_key:
    st.session_state.last_date_range = dr_key
    st.session_state.df = pd.DataFrame()
    st.cache_data.clear()

if data_source in ("NASA FIRMS (online)", "INPE + FIRMS (online)"):
    if "last_firms_days" not in st.session_state:
        st.session_state.last_firms_days = firms_days
    elif st.session_state.last_firms_days != firms_days:
        st.session_state.last_firms_days = firms_days
        st.session_state.df = pd.DataFrame()
        st.cache_data.clear()


@st.cache_data(ttl=1800)
def load_data_inpe(date_from: str, date_to: str):
    return fetch_inpe_fire_foci(state_code="23", date_from=date_from, date_to=date_to)


df = st.session_state.df
source_note = ""
available_dates_source: list[str] = []

_is_online_source = data_source in (
    "INPE (online)",
    "GOES-16 (INPE online)",
    "NASA FIRMS (online)",
    "INPE + FIRMS (online)",
)
_needs_load = df.empty
if df.empty and _is_online_source and st.session_state.get("last_online_empty_source") == data_source:
    _needs_load = False

if _needs_load:
    _loading_placeholder = st.empty()
    _loading_placeholder.info("A carregar focos…")
    _progress = st.progress(0, text="Iniciando…")
    with st.spinner("A carregar dados…"):
        if data_source == "INPE (online)":
            _progress.progress(20, text="INPE…")
            df = load_data_inpe(
                pd.Timestamp(date_from).strftime("%Y-%m-%d"),
                pd.Timestamp(date_to).strftime("%Y-%m-%d"),
            )
            _progress.progress(80, text="A processar…")
            available_dates_source = _extract_available_dates(df)
            source_note = "⚠️ INPE vazio" if df.empty else "✅ INPE"
            if df.empty:
                st.warning(
                    "INPE não devolveu focos (rede ou período). Tente CSV local ou outra fonte."
                )
        elif data_source == "GOES-16 (INPE online)":
            _progress.progress(20, text="GOES-16…")
            df = fetch_goes16_fire_foci_ceara(
                date_from=pd.Timestamp(date_from).strftime("%Y-%m-%d"),
                date_to=pd.Timestamp(date_to).strftime("%Y-%m-%d"),
            )
            available_dates_source = _extract_available_dates(df)
            source_note = "⚠️ GOES-16 vazio" if df.empty else "✅ GOES-16"
            if df.empty:
                st.warning("Sem focos GOES-16 no período ou serviço indisponível.")
        elif data_source == "NASA FIRMS (online)":
            _progress.progress(20, text="FIRMS…")
            df = fetch_firms_ceara(days=firms_days)
            available_dates_source = _extract_available_dates(df)
            source_note = "⚠️ FIRMS vazio" if df.empty else "✅ FIRMS"
            if df.empty:
                st.warning("FIRMS vazio ou `FIRMS_API_KEY` em falta.")
        elif data_source == "INPE + FIRMS (online)":
            _progress.progress(15, text="INPE…")
            df_inpe = load_data_inpe(
                pd.Timestamp(date_from).strftime("%Y-%m-%d"),
                pd.Timestamp(date_to).strftime("%Y-%m-%d"),
            )
            _progress.progress(50, text="FIRMS…")
            df_firms = fetch_firms_ceara(days=firms_days)
            if df_inpe.empty and df_firms.empty:
                df = pd.DataFrame()
                source_note = "⚠️ INPE e FIRMS vazios"
            elif df_firms.empty:
                df = df_inpe
                source_note = "✅ INPE (FIRMS vazio)"
            elif df_inpe.empty:
                df = df_firms
                source_note = "✅ FIRMS (INPE vazio)"
            else:
                df = merge_inpe_firms(df_inpe, df_firms)
                source_note = "✅ INPE+FIRMS"
            available_dates_source = _extract_available_dates(df)
        elif data_source == "Banco diário (auto)":
            _progress.progress(30, text="Banco diário…")
            if os.path.exists(_daily_db_path):
                df_all = load_local_fire_data(_daily_db_path)
                available_dates_source = _extract_available_dates(df_all)
                df = df_all
                if "datetime" in df.columns:
                    dt = pd.to_datetime(df["datetime"], errors="coerce")
                    mask = (dt >= pd.Timestamp(date_from)) & (dt < pd.Timestamp(date_to) + pd.Timedelta(days=1))
                    df = df[mask].copy()
                source_note = "✅ Banco diário"
            else:
                st.warning("Banco diário inexistente — use o expander na sidebar para criar.")
                df = pd.DataFrame()
                source_note = "⚠️ Banco diário ausente"
        elif data_source == "Dados Locais (CSV)":
            _progress.progress(20, text="CSVs…")
            csv_dir = PROJECT_ROOT / "data"
            csvs = sorted([f for f in os.listdir(csv_dir) if f.endswith(".csv")]) if csv_dir.is_dir() else []
            if csvs:
                dfs_local = []
                for i, fname in enumerate(csvs):
                    _progress.progress(20 + int(60 * i / max(1, len(csvs))), text=f"{fname}…")
                    try:
                        dfs_local.append(load_local_fire_data(str(csv_dir / fname)))
                    except Exception:
                        pass
                if dfs_local:
                    df_all = pd.concat(dfs_local, ignore_index=True)
                    if "datetime" in df_all.columns:
                        df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
                        df_all = df_all.drop_duplicates(subset=["lat", "lon", "datetime"])
                    available_dates_source = _extract_available_dates(df_all)
                    if "datetime" in df_all.columns:
                        _dt = pd.to_datetime(df_all["datetime"], errors="coerce")
                        _mask = (_dt >= pd.Timestamp(date_from)) & (_dt < pd.Timestamp(date_to) + pd.Timedelta(days=1))
                        df = df_all[_mask].copy() if _mask.any() else df_all.copy()
                    else:
                        df = df_all
                    anos = sorted(df_all["datetime"].dt.year.dropna().unique().astype(int).tolist()) if "datetime" in df_all.columns else []
                    source_note = f"📁 {len(csvs)} CSV · {len(df_all)} focos · {', '.join(map(str, anos))}"
                else:
                    df = pd.DataFrame()
                    source_note = "⚠️ CSVs ilegíveis"
            else:
                st.warning("Pasta **data/** sem `.csv`.")
                df = pd.DataFrame()
                source_note = "⚠️ Sem CSV"

        _progress.progress(100, text="Concluído")
        _progress.empty()
        _loading_placeholder.empty()

    if _is_online_source and df.empty:
        st.session_state["last_online_empty_source"] = data_source
    elif not df.empty:
        st.session_state.pop("last_online_empty_source", None)

    st.session_state.df = df
    st.session_state.available_dates_current_source = available_dates_source
    st.session_state.source_note = source_note
    st.session_state.last_refresh = datetime.now()

if "source_note" in st.session_state and st.session_state.source_note:
    source_note = st.session_state.source_note

available_dates_source = st.session_state.get("available_dates_current_source", [])
if available_dates_source:
    day_pick = st.sidebar.selectbox(
        "Dia único no mapa (opcional)",
        options=["Todos"] + available_dates_source,
        index=0,
    )
    if day_pick != "Todos" and not df.empty and "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], errors="coerce")
        df = df[dt.dt.strftime("%Y-%m-%d") == day_pick].copy()

# ============================================================================
# Abas principais
# ============================================================================

tab_model, tab_foci = st.tabs(["ST-HyperNet (modelo)", "Focos no mapa"])

with tab_model:
    render_st_hypernet_tab(metrics_dir_rel)

with tab_foci:
    if df.empty:
        st.info(
            "Sem focos para o período e fonte escolhidos. "
            "Use **Dados Locais (CSV)** ou amplie o intervalo de datas."
        )
        if available_dates_source:
            st.caption(
                f"Dias com dados na última leitura completa: {available_dates_source[0]} … "
                f"{available_dates_source[-1]} ({len(available_dates_source)} dias)."
            )
        if st.button("Tentar recarregar da internet"):
            st.cache_data.clear()
            st.session_state.df = pd.DataFrame()
            st.rerun()
        with st.expander("Briefing opcional (notícias / satélite)", expanded=False):
            render_intel_agent_section(0)
    else:
        st.caption(f"Fonte: **{source_note}** · Última leitura: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        c0, c1, c2, c3 = st.columns(4)
        c0.metric("Focos (vista atual)", len(df))
        if "lat" in df.columns:
            c1.metric("Locais únicos (aprox.)", df[["lat", "lon"]].dropna().round(2).drop_duplicates().shape[0])
        if "datetime" in df.columns:
            c2.metric("Primeira data", str(pd.to_datetime(df["datetime"]).min().date()))
            c3.metric("Última data", str(pd.to_datetime(df["datetime"]).max().date()))

        try:
            import folium
            from folium.plugins import Fullscreen, MarkerCluster

            center_lat, center_lon = -5.2, -39.5
            url_sat, attr_sat, opts_sat = satellite_layer_for_folium(sat_choice)
            m_det = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=7,
                control_scale=True,
                tiles="CartoDB dark_matter",
            )
            folium.TileLayer(
                url_sat,
                attr=attr_sat,
                name="Satélite",
                overlay=False,
                control=True,
                **{k: v for k, v in opts_sat.items() if k != "name"},
            ).add_to(m_det)

            cluster = MarkerCluster(name="Focos").add_to(m_det)
            df_pts = df.copy()
            if len(df_pts) > 8000:
                df_pts = df_pts.sample(8000, random_state=42)
                st.caption(f"Amostra de 8.000 de {len(df)} pontos.")

            for _, row in df_pts.iterrows():
                if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                    continue
                popup_html = (
                    f"<b>Foco</b><br>Lat/Lon: {row['lat']:.4f}, {row['lon']:.4f}<br>"
                    f"Município: {row.get('municipio', '—')}<br>Quando: {row.get('datetime', '—')}"
                )
                folium.CircleMarker(
                    location=[float(row["lat"]), float(row["lon"])],
                    radius=6,
                    color="#ff3300",
                    weight=1,
                    fill=True,
                    fillColor="#ff0000",
                    fillOpacity=0.55,
                    popup=folium.Popup(popup_html, max_width=280),
                ).add_to(cluster)

            folium.GeoJson(
                {
                    "type": "Feature",
                    "properties": {"name": "Ceará (bbox)"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [CEARA_BBOX["min_lon"], CEARA_BBOX["min_lat"]],
                            [CEARA_BBOX["max_lon"], CEARA_BBOX["min_lat"]],
                            [CEARA_BBOX["max_lon"], CEARA_BBOX["max_lat"]],
                            [CEARA_BBOX["min_lon"], CEARA_BBOX["max_lat"]],
                            [CEARA_BBOX["min_lon"], CEARA_BBOX["min_lat"]],
                        ]],
                    },
                },
                style_function=lambda _: {
                    "fillColor": "none",
                    "color": "#00ff88",
                    "weight": 2,
                    "dashArray": "5, 8",
                },
            ).add_to(m_det)
            folium.LayerControl(collapsed=False).add_to(m_det)
            Fullscreen().add_to(m_det)
            render_folium_map(m_det, height=520)

            t1, t2 = st.columns(2)
            with t1:
                show_n = min(400, len(df))
                cols_tbl = ["lat", "lon", "datetime", "satellite", "municipio", "source"]
                avail = [c for c in cols_tbl if c in df.columns]
                sub = df[avail].head(show_n)
                if "datetime" in sub.columns:
                    sub = sub.sort_values("datetime", ascending=False, na_position="last")
                st.dataframe(sub, hide_index=True, use_container_width=True, height=260)
            with t2:
                st.download_button(
                    label="Exportar CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"focos_ce_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
                with st.expander("Briefing opcional (notícias)", expanded=False):
                    render_intel_agent_section(len(df))
        except ImportError as e:
            st.error(f"Instale folium: pip install folium — {e}")

st.divider()
st.caption(
    "**Digital Twin Queimadas CE** — ST-HyperNet + focos INPE/FIRMS. "
    "Repositório: [github/naubergois/digital-twin-queimadas-ceara](https://github.com/naubergois/digital-twin-queimadas-ceara)"
)
