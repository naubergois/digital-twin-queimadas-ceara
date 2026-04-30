"""
Dashboard Interativo — Digital Twin Queimadas Ceará

Executar com: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fire_data import (
    fetch_inpe_fire_foci,
    fetch_inpe_fire_summary,
    load_local_fire_data,
)
from src.digital_twin import FireDigitalTwin
from src.analysis import FireAnalysis
from src.satellite import (
    satellite_layer_for_folium,
    available_satellite_sources,
    fetch_open_meteo_fire_index,
    fire_danger_index,
    fetch_firms_fires,
    GIBS_LAYERS,
    gibs_tile_url,
    list_gibs_layers,
)
from config.ceara_config import CEARA_BBOX, AREAS_CRITICAS

# ============================================================================
# Config
# ============================================================================

st.set_page_config(
    page_title="Digital Twin — Queimadas CE",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Se estiver rodando no servidor, aponta assets
if "STREAMLIT_SERVER" in os.environ:
    os.environ["STREAMLIT_THEME_BASE"] = "dark"

st.title("🛰️ Digital Twin — Monitoramento de Queimadas no Ceará")
st.markdown(
    "Monitoramento em **quase tempo real** com imagens de satélite NASA, "
    "focos do INPE/FIRMS e simulação de propagação do fogo."
)

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.header("⚙️ Controles")

# ── Período ──
date_range = st.sidebar.date_input(
    "Período de análise",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
)

# ── Fonte ──
data_source = st.sidebar.selectbox(
    "Fonte de focos",
    ["INPE (online)", "Dados Locais (CSV)", "Demonstração (sintético)"],
)

# ── Satélite ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛰️ Satélite")

sat_sources = available_satellite_sources()
tile_sources = [s for s in sat_sources if s["type"] == "Tile"]
sat_choice = st.sidebar.selectbox(
    "Camada de satélite",
    options=[s["id"] for s in tile_sources],
    format_func=lambda x: next(s["name"] for s in tile_sources if s["id"] == x),
    index=0,  # ESRI Satellite
    help="Fonte das imagens de satélite no mapa. NASA GIBS tem ~1 dia de atraso. ESRI/Google são composições recentes.",
)

# ── Auto-refresh ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Streaming")
auto_refresh = st.sidebar.checkbox("Auto-refresh (a cada 30s)", value=False)
refresh_interval = st.sidebar.slider(
    "Intervalo (segundos)", 10, 120, 30, step=10,
    disabled=not auto_refresh,
)

# ── Simulação ──
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Gêmeo Digital")
enable_simulation = st.sidebar.checkbox("Ativar simulação", value=True)
sim_steps = st.sidebar.slider("Passos", 1, 48, 24)

# ============================================================================
# Estado da sessão (persiste entre re-runs)
# ============================================================================

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()
if "refresh_count" not in st.session_state:
    st.session_state.refresh_count = 0
if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

# ============================================================================
# Carregar dados
# ============================================================================

@st.cache_data(ttl=300)
def load_data_inpe(date_from: str, date_to: str):
    return fetch_inpe_fire_foci(state_code="23", date_from=date_from, date_to=date_to)


@st.cache_data
def load_synthetic():
    np.random.seed(42)
    n = 500
    regions_data = {
        "Sertão Central": {"lat": (-5.8, -5.0), "lon": (-39.5, -38.5), "weight": 0.35},
        "Cariri": {"lat": (-7.5, -6.8), "lon": (-39.8, -38.8), "weight": 0.15},
        "Norte": {"lat": (-4.0, -3.2), "lon": (-40.0, -39.0), "weight": 0.20},
        "Jaguaribe": {"lat": (-6.0, -5.2), "lon": (-38.5, -37.5), "weight": 0.15},
        "Ibiapaba": {"lat": (-4.5, -3.5), "lon": (-41.0, -40.2), "weight": 0.15},
    }
    cities_map = {
        "Sertão Central": ["Tauá", "Crateús", "Independência"],
        "Cariri": ["Juazeiro do Norte", "Crato", "Barbalha"],
        "Norte": ["Sobral", "Itapipoca", "Santa Quitéria"],
        "Jaguaribe": ["Jaguaribe", "Limoeiro do Norte", "Russas"],
        "Ibiapaba": ["Tianguá", "Viçosa do CE", "Ubajara"],
    }

    rows = []
    for _ in range(n):
        reg = np.random.choice(list(regions_data.keys()),
                               p=[r["weight"] for r in regions_data.values()])
        r = regions_data[reg]
        rows.append({
            "lat": round(np.random.uniform(*r["lat"]), 4),
            "lon": round(np.random.uniform(*r["lon"]), 4),
            "datetime": datetime.now() - timedelta(days=np.random.randint(0, 60)),
            "satellite": np.random.choice(["AQUA_M-T", "TERRA_M-T", "NPP-375", "NOAA-20"]),
            "municipio": np.random.choice(cities_map[reg]),
            "bioma": "Caatinga",
            "source": "SYNTHETIC",
        })
    df = pd.DataFrame(rows)
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    return df


df = st.session_state.df

if df.empty or auto_refresh:
    with st.spinner("Carregando dados..."):
        if data_source == "INPE (online)":
            df = load_data_inpe(
                date_range[0].strftime("%Y-%m-%d"),
                date_range[1].strftime("%Y-%m-%d"),
            )
            if df.empty:
                st.warning("INPE sem dados. Usando demo.")
                df = load_synthetic()
                source_note = "⚠️ Sintético"
            else:
                source_note = "✅ INPE"
        elif data_source == "Dados Locais (CSV)":
            csv_dir = os.path.join(os.path.dirname(__file__), "..", "data")
            csvs = [f for f in os.listdir(csv_dir) if f.endswith(".csv")] if os.path.exists(csv_dir) else []
            if csvs:
                sel = st.sidebar.selectbox("CSV", csvs) if len(csvs) > 1 else csvs[0]
                df = load_local_fire_data(os.path.join(csv_dir, sel if len(csvs) > 1 else csvs[0]))
                source_note = f"📁 {csvs[0]}"
            else:
                df = load_synthetic()
                source_note = "⚠️ Sintético"
        else:
            df = load_synthetic()
            source_note = "⚠️ Sintético"

    st.session_state.df = df
    st.session_state.last_refresh = datetime.now()
    st.session_state.refresh_count += 1

# ============================================================================
# Indicadores principais
# ============================================================================

if not df.empty:
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("🔥 Focos totais", len(df))

    with col2:
        if "month" in df.columns:
            month_f = df[df["month"] == datetime.now().month]
            st.metric("📅 Este mês", len(month_f))
        else:
            st.metric("📅 Este mês", "N/A")

    with col3:
        if "lat" in df.columns:
            unique = df[["lat", "lon"]].dropna().round(2).drop_duplicates().shape[0]
            st.metric("📍 Locais únicos", unique)
        else:
            st.metric("📍 Locais únicos", "N/A")

    with col4:
        st.metric("🛰️ Satélite", sat_choice.split("_")[1].upper() if "_" in sat_choice else "ESRI")

    with col5:
        # Índice de perigo (Open-Meteo)
        weather = fetch_open_meteo_fire_index()
        danger = fire_danger_index(weather)
        if danger:
            today = danger[0]["fire_danger_index"]
            st.metric("🔥 Risco hoje", f"{today:.0f} / 100",
                      delta=f"{danger[0]['temp_max']:.0f}°C / {danger[0]['wind_max']:.0f}km/h vento")
        else:
            st.metric("🔥 Risco", "N/A")

    st.divider()

    # ============================================================================
    # ABAS
    # ============================================================================

    tab_sat, tab_map, tab_time, tab_twin, tab_report = st.tabs([
        "🛰️ Satélite Ao Vivo",
        "🗺️ Mapa de Calor",
        "📈 Análise Temporal",
        "🤖 Gêmeo Digital",
        "📋 Relatório",
    ])

    # ────────────────────────────────────────────────────────────────────────
    # TAB 1: SATÉLITE AO VIVO (STREAMING)
    # ────────────────────────────────────────────────────────────────────────
    with tab_sat:
        st.subheader("🛰️ Imagens de Satélite — Ceará (Quase Tempo Real)")

        col_sat_ctl, col_sat_info = st.columns([2, 1])

        with col_sat_ctl:
            st.caption(
                f"**Fonte:** {next(s['name'] for s in tile_sources if s['id'] == sat_choice)} | "
                f"**Atualizado:** {st.session_state.last_refresh.strftime('%H:%M:%S')} "
                f"(refresh #{st.session_state.refresh_count})"
            )

        with col_sat_info:
            if auto_refresh:
                st.info(f"↻ Auto-refresh a cada {refresh_interval}s", icon="🔄")
                # Placeholder para o tempo restante
                next_refresh = st.session_state.last_refresh + timedelta(seconds=refresh_interval)
                remaining = (next_refresh - datetime.now()).total_seconds()
                if remaining > 0:
                    st.caption(f"Próximo refresh em {remaining:.0f}s")
            else:
                if st.button("🔄 Atualizar agora"):
                    st.cache_data.clear()
                    st.session_state.df = pd.DataFrame()
                    st.rerun()

        st.markdown("---")

        # ── Mapa com imagem de satélite ──
        try:
            import folium
            from streamlit_folium import st_folium
            from folium.plugins import Fullscreen, MousePosition

            # Coordenada central do Ceará
            center_lat, center_lon = -5.2, -39.5

            # Criar mapa com layer de satélite
            url_sat, attr_sat, opts_sat = satellite_layer_for_folium(sat_choice)
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=7,
                control_scale=True,
                tiles="CartoDB dark_matter",  # fallback escuro
                attr="OpenStreetMap",
            )

            # Adicionar layer de satélite como principal
            folium.TileLayer(
                url_sat,
                attr=attr_sat,
                name="🛰️ Satélite",
                overlay=False,
                control=True,
                **{k: v for k, v in opts_sat.items() if k != "name"},
            ).add_to(m)

            # Adicionar layer de focos de calor ativos
            fire_group = folium.FeatureGroup(name="🔥 Focos Ativos", show=True)

            df_map = df.copy()
            if len(df_map) > 3000:
                df_map = df_map.sample(3000, random_state=42)

            # Adicionar marcadores de fogo
            for _, row in df_map.iterrows():
                if pd.isna(row.get("lat")) or pd.isna(row.get("lon")):
                    continue

                # Intensidade baseada em fonte (ou aleatória para legibilidade)
                intensity = np.random.uniform(0.3, 1.0)

                folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=3 + intensity * 4,
                    color="#ff4500",
                    fill=True,
                    fillColor="#ff0000",
                    fillOpacity=0.4 + intensity * 0.4,
                    popup=(
                        f"<b>Foco</b><br>"
                        f"Lat: {row['lat']:.3f}<br>"
                        f"Lon: {row['lon']:.3f}<br>"
                        f"Sat: {row.get('satellite', 'N/A')}<br>"
                        f"Mun: {row.get('municipio', 'N/A')}<br>"
                        f"Data: {row.get('datetime', 'N/A')}"
                    ),
                ).add_to(fire_group)

            fire_group.add_to(m)

            # Contorno do Ceará
            folium.GeoJson(
                {
                    "type": "Feature",
                    "properties": {},
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
                style_function=lambda x: {
                    "fillColor": "none",
                    "color": "#00ff88",
                    "weight": 2,
                    "dashArray": "6, 4",
                },
            ).add_to(m)

            # Áreas críticas
            for area_name, area_info in AREAS_CRITICAS.items():
                color = "red" if area_info["nivel_risco"] == "ALTO" else "orange"
                folium.Marker(
                    area_info["centroid"],
                    popup=f"<b>{area_name}</b><br>{area_info['descricao']}",
                    icon=folium.Icon(color=color, icon="info-sign"),
                ).add_to(m)

            # Controle de camadas
            folium.LayerControl(position="topright", collapsed=True).add_to(m)

            # Mouse position
            MousePosition(position="bottomleft", separator=" | ").add_to(m)

            # Fullscreen
            Fullscreen().add_to(m)

            # Timestamp no canto
            from folium.elements import IFrame

            timestamp_html = (
                f'<div style="position:absolute;bottom:10px;right:10px;'
                f'background:rgba(0,0,0,0.7);color:white;padding:4px 10px;'
                f'border-radius:4px;font-size:11px;z-index:9999;">'
                f'🛰️ {st.session_state.last_refresh.strftime("%Y-%m-%d %H:%M:%S")} UTC-3'
                f'</div>'
            )
            m.get_root().html.add_child(folium.Element(timestamp_html))

            # Renderizar mapa
            map_output = st_folium(
                m,
                width=None,
                height=580,
                key=f"sat_map_{st.session_state.refresh_count}",
            )

            # ── Painel informativo ao lado ──
            with st.expander("📊 Informações da imagem", expanded=False):
                col_s1, col_s2, col_s3 = st.columns(3)

                with col_s1:
                    # GIBS layer info
                    if sat_choice.startswith("gibs_"):
                        gibs_info = GIBS_LAYERS.get(sat_choice.replace("gibs_", "", 1))
                        if gibs_info:
                            st.markdown(f"**Camada:** {gibs_info['description']}")
                            st.markdown(f"**Resolução:** {gibs_info['resolution']}")
                            st.markdown(f"**Atraso:** {gibs_info['delay_days']} dia(s)")

                with col_s2:
                    st.markdown(f"**Total de focos:** {len(df)}")
                    st.markdown(f"**Fonte:** {source_note}")
                    if not df.empty and "lat" in df.columns:
                        st.markdown(f"**Área coberta:** "
                                    f"{df['lon'].min():.2f}° a {df['lon'].max():.2f}° W / "
                                    f"{df['lat'].min():.2f}° a {df['lat'].max():.2f}° S")

                with col_s3:
                    st.markdown(f"**Auto-refresh:** {'✅ Ativo' if auto_refresh else '❌ Inativo'}")
                    st.markdown(f"**Lat/Lon centro:** {center_lat:.1f}°, {center_lon:.1f}°")
                    st.markdown(f"**Zoom:** {7}")

        except ImportError as e:
            st.error(
                f"Bibliotecas de mapa não encontradas: {e}. "
                "Instale com: pip install folium streamlit-folium"
            )

        # ── Linha do tempo animada ──
        st.markdown("---")
        st.subheader("⏳ Linha do tempo — Evolução dos focos (últimos dias)")

        if "datetime" in df.columns:
            df_ts = df.copy()
            df_ts["date"] = pd.to_datetime(df_ts["datetime"]).dt.date
            daily_counts = df_ts.groupby("date").size().reset_index(name="count")
            daily_counts = daily_counts.sort_values("date")

            col_tl1, col_tl2 = st.columns([3, 1])

            with col_tl1:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 3))
                ax.fill_between(
                    daily_counts["date"],
                    daily_counts["count"],
                    alpha=0.3,
                    color="#ff6b35",
                )
                ax.plot(
                    daily_counts["date"],
                    daily_counts["count"],
                    color="#e74c3c",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )
                ax.set_xlabel("Data")
                ax.set_ylabel("Focos")
                ax.grid(True, alpha=0.2)
                fig.tight_layout()
                st.pyplot(fig)

            with col_tl2:
                st.metric("Média diária",
                          f"{daily_counts['count'].mean():.0f}")
                st.metric("Máx. em um dia",
                          f"{daily_counts['count'].max():.0f}")
                st.metric("Dias com foco",
                          len(daily_counts))

                # Previsão de risco
                if danger:
                    st.markdown("---")
                    st.markdown("**🌤️ Risco de fogo (próx. dias)**")
                    for d in danger[:3]:
                        emoji = "🔴" if d["fire_danger_index"] > 60 else "🟡" if d["fire_danger_index"] > 30 else "🟢"
                        st.markdown(f"{emoji} **{d['date']}**: {d['fire_danger_index']}")

        else:
            st.info("Dados sem timestamp para linha do tempo.")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 2: MAPA DE CALOR
    # ────────────────────────────────────────────────────────────────────────
    with tab_map:
        st.subheader("🗺️ Mapa de Calor de Queimadas no Ceará")

        if "lat" in df.columns:
            col_m1, col_m2 = st.columns([3, 1])

            with col_m2:
                st.caption("Filtros do mapa")
                if "satellite" in df.columns:
                    sats = ["Todos"] + sorted(df["satellite"].unique().tolist())
                    sel_sat = st.selectbox("Satélite", sats, key="hm_sat")
                else:
                    sel_sat = "Todos"
                if "bioma" in df.columns:
                    biomas = ["Todos"] + sorted(df["bioma"].unique().tolist())
                    sel_bio = st.selectbox("Bioma", biomas, key="hm_bio")
                else:
                    sel_bio = "Todos"

                st.caption("Áreas Críticas")
                for an, ai in AREAS_CRITICAS.items():
                    emoji = "🔴" if ai["nivel_risco"] == "ALTO" else "🟡"
                    st.markdown(f"{emoji} **{an}**")

            with col_m1:
                df_filt = df.copy()
                if sel_sat != "Todos" and "satellite" in df_filt.columns:
                    df_filt = df_filt[df_filt["satellite"] == sel_sat]
                if sel_bio != "Todos" and "bioma" in df_filt.columns:
                    df_filt = df_filt[df_filt["bioma"] == sel_bio]
                if len(df_filt) > 5000:
                    df_filt = df_filt.sample(5000, random_state=42)

                try:
                    import folium
                    from streamlit_folium import st_folium
                    from folium.plugins import HeatMap

                    m = folium.Map(location=[-5.2, -39.5], zoom_start=7,
                                   tiles="CartoDB positron")
                    heat_data = df_filt[["lat", "lon"]].dropna().values.tolist()
                    HeatMap(
                        heat_data, radius=12, blur=15, max_zoom=1,
                        gradient={0.4: "blue", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
                    ).add_to(m)

                    for an, ai in AREAS_CRITICAS.items():
                        color = "red" if ai["nivel_risco"] == "ALTO" else "orange"
                        folium.Marker(
                            ai["centroid"],
                            popup=f"<b>{an}</b><br>{ai['descricao']}",
                            icon=folium.Icon(color=color, icon="warning-sign"),
                        ).add_to(m)

                    st_folium(m, width=700, height=500)

                except ImportError:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.scatter(df_filt["lon"], df_filt["lat"], alpha=0.3, s=5, c="red")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    ax.set_title("Focos no Ceará")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 3: ANÁLISE TEMPORAL
    # ────────────────────────────────────────────────────────────────────────
    with tab_time:
        st.subheader("📈 Análise Temporal")

        analysis = FireAnalysis(df)

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            monthly = analysis.monthly_distribution()
            if not monthly.empty:
                st.subheader("Distribuição Mensal")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ["#ff6b35" if m >= 6 else "#4ecdc4" for m in monthly["month"]]
                bars = ax.bar(monthly["month"], monthly["count"], color=colors)
                ax.set_xlabel("Mês")
                ax.set_ylabel("Focos")
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels(["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"])
                ax.grid(True, alpha=0.3, axis="y")
                from matplotlib.patches import Patch
                ax.legend(handles=[
                    Patch(color="#ff6b35", label="Seca (Jun-Dez)"),
                    Patch(color="#4ecdc4", label="Chuva (Jan-Mai)"),
                ], loc="upper right")
                st.pyplot(fig)

        with col_a2:
            season = analysis.peak_season()
            if season:
                st.subheader("Sazonalidade")
                st.metric("Mês de Pico", season["peak_month"])
                st.metric("Estação Seca", f"{season['dry_season_pct']:.0f}% dos focos")
                st.info(
                    f"🔍 {season['dry_season_pct']:.0f}% dos focos ocorrem na "
                    f"estação seca (jun-dez), quando a Caatinga está mais seca."
                )

        yearly = analysis.yearly_trend()
        if not yearly.empty and len(yearly) > 1:
            st.subheader("Tendência Anual")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(yearly["year"], yearly["count"], marker="o", color="#e74c3c", linewidth=2)
            ax.fill_between(yearly["year"], yearly["count"], alpha=0.2, color="#e74c3c")
            ax.set_xlabel("Ano")
            ax.set_ylabel("Focos")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

    # ────────────────────────────────────────────────────────────────────────
    # TAB 4: GÊMEO DIGITAL
    # ────────────────────────────────────────────────────────────────────────
    with tab_twin:
        st.subheader("🤖 Simulação do Gêmeo Digital")

        if enable_simulation:
            col_t1, col_t2 = st.columns([2, 1])

            with col_t2:
                st.caption("Parâmetros")
                spread_rate = st.slider("Propagação", 0.1, 2.0, 1.0, 0.1, key="sr")
                dry_factor = st.slider("Seca vegetação", 0.0, 1.0, 0.6, 0.1, key="df")
                if st.button("▶️ Simular", type="primary"):
                    st.session_state.run_sim = True
                elif "run_sim" not in st.session_state:
                    st.session_state.run_sim = False

            with col_t1:
                if st.session_state.get("run_sim"):
                    with st.spinner("Simulando..."):
                        twin = FireDigitalTwin(resolution=0.05)
                        twin.initialize_from_history(df)
                        recent = df.copy()
                        if "datetime" in recent.columns:
                            recent = recent[recent["datetime"] >= pd.Timestamp.now() - pd.Timedelta(days=7)]
                        twin.add_active_fires(recent)
                        history = twin.simulate(steps=sim_steps)
                        final = history[-1]

                        c1, c2, c3 = st.columns(3)
                        c1.metric("🔥 Em chamas", final["burning_cells"])
                        c2.metric("🔥 Já queimado", final["burned_cells"])
                        c3.metric("📍 Afetado", final["total_affected"])

                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots(figsize=(10, 3))
                        dh = pd.DataFrame(history)
                        ax.plot(dh["step"], dh["burning_cells"], label="Em chamas", color="#ff6b35")
                        ax.plot(dh["step"], dh["burned_cells"], label="Queimado", color="#8b0000")
                        ax.plot(dh["step"], dh["total_affected"], label="Total", color="#2c3e50", linewidth=2)
                        ax.set_xlabel("Passo")
                        ax.set_ylabel("Células")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                        zones = twin.get_fire_danger_zones(threshold=0.5)
                        if zones:
                            st.subheader("🚨 Zonas de Risco")
                            zdf = pd.DataFrame(zones[:5])
                            zdf["estimated_area_km2"] = zdf["estimated_area_km2"].round(1)
                            st.dataframe(zdf, hide_index=True, use_container_width=True)

                        critical = twin.check_critical_areas()
                        st.subheader("🏞️ Áreas Críticas")
                        for area in critical:
                            risco = area["nivel_risco"]
                            emoji = "🔴" if risco == "ALTO" else "🟡"
                            alert = "🔥 ATIVO" if area["em_chamas"] else "✅ OK"
                            st.markdown(f"{emoji} **{area['area']}** — Risco {risco} | {alert}")
                else:
                    st.info("👈 Clique em **Simular** para rodar o gêmeo digital.")
        else:
            st.info("Ative 'Ativar simulação' na sidebar.")

    # ────────────────────────────────────────────────────────────────────────
    # TAB 5: RELATÓRIO
    # ────────────────────────────────────────────────────────────────────────
    with tab_report:
        st.subheader("📋 Relatório Resumido")

        analysis = FireAnalysis(df)
        report = analysis.summary_report()

        if "error" not in report:
            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.markdown("### 📊 Estatísticas")
                st.markdown(f"- **Total focos:** {report['total_focos']}")
                if "periodo" in report:
                    st.markdown(f"- **Período:** {report['periodo'].get('inicio','N/A')} → {report['periodo'].get('fim','N/A')}")

            with col_r2:
                if "sazonalidade" in report:
                    s = report["sazonalidade"]
                    st.markdown("### 🌦️ Sazonalidade")
                    st.markdown(f"- **Pico:** {s['peak_month']}")
                    st.markdown(f"- **Seca:** {s['dry_season_pct']:.0f}%")

            top_muni = analysis.top_municipios(10)
            if not top_muni.empty and "municipio" in top_muni.columns:
                st.markdown("### 🏙️ Top Municípios")
                st.dataframe(top_muni, hide_index=True, use_container_width=True)

# ============================================================================
# Auto-refresh (rerun automático)
# ============================================================================

if auto_refresh and not df.empty:
    elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if elapsed >= refresh_interval:
        st.cache_data.clear()
        st.session_state.df = pd.DataFrame()
        st.rerun()

# ============================================================================
# Footer
# ============================================================================

st.divider()
st.caption(
    "**Digital Twin Queimadas CE** | "
    "Fontes: INPE BDQueimadas, NASA GIBS/FIRMS, Open-Meteo | "
    "Código: [github/naubergois/digital-twin-queimadas-ceara](https://github.com/naubergois/digital-twin-queimadas-ceara) | "
    f"Último refresh: {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}"
)
