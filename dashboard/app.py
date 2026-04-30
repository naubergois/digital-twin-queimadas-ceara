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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fire_data import (
    fetch_inpe_fire_foci,
    fetch_inpe_fire_summary,
    load_local_fire_data,
)
from src.digital_twin import FireDigitalTwin
from src.analysis import FireAnalysis
from config.ceara_config import CEARA_BBOX, AREAS_CRITICAS

# ============================================================================
# Configuração da página
# ============================================================================

st.set_page_config(
    page_title="Digital Twin — Queimadas CE",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔥 Digital Twin para Detecção de Queimadas no Ceará")
st.markdown(
    "Monitoramento e predição de focos de calor usando dados abertos de satélite "
    "(INPE / NASA FIRMS / Sentinel-2)"
)

# ============================================================================
# Sidebar
# ============================================================================

st.sidebar.header("⚙️ Controles")

# Período de análise
date_range = st.sidebar.date_input(
    "Período de análise",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
)

# Fonte de dados
data_source = st.sidebar.selectbox(
    "Fonte de dados",
    ["INPE (online)", "Dados Locais (CSV)", "Demonstração (sintético)"],
)

# Modo de simulação
enable_simulation = st.sidebar.checkbox("Ativar simulação (Digital Twin)", value=True)
sim_steps = st.sidebar.slider("Passos de simulação", 1, 48, 24)

# ============================================================================
# Carregamento de dados
# ============================================================================

@st.cache_data(ttl=600)
def load_data_inpe(date_from: str, date_to: str):
    """Carrega dados da API do INPE."""
    return fetch_inpe_fire_foci(
        state_code="23",
        date_from=date_from,
        date_to=date_to,
    )


@st.cache_data
def load_data_local():
    """Carrega dados sintéticos para demonstração."""
    # Gerar dados sintéticos plausíveis para o Ceará
    np.random.seed(42)
    n_points = 500

    # Focos concentrados nas regiões de maior incidência
    regions = {
        "Sertão Central": {"lat": (-5.8, -5.0), "lon": (-39.5, -38.5), "weight": 0.35},
        "Cariri": {"lat": (-7.5, -6.8), "lon": (-39.8, -38.8), "weight": 0.15},
        "Norte": {"lat": (-4.0, -3.2), "lon": (-40.0, -39.0), "weight": 0.20},
        "Jaguaribe": {"lat": (-6.0, -5.2), "lon": (-38.5, -37.5), "weight": 0.15},
        "Ibiapaba": {"lat": (-4.5, -3.5), "lon": (-41.0, -40.2), "weight": 0.15},
    }

    data = []
    cities = {
        "Sertão Central": ["Tauá", "Crateús", "Independência"],
        "Cariri": ["Juazeiro do Norte", "Crato", "Barbalha"],
        "Norte": ["Sobral", "Itapipoca", "Santa Quitéria"],
        "Jaguaribe": ["Jaguaribe", "Limoeiro do Norte", "Russas"],
        "Ibiapaba": ["Tianguá", "Viçosa do CE", "Ubajara"],
    }

    for _ in range(n_points):
        region = np.random.choice(list(regions.keys()), p=[r["weight"] for r in regions.values()])
        r = regions[region]

        lat = np.random.uniform(*r["lat"])
        lon = np.random.uniform(*r["lon"])

        day_offset = np.random.randint(0, 60)
        dt = datetime.now() - timedelta(days=day_offset)

        sat = np.random.choice(["AQUA_M-T", "TERRA_M-T", "NPP-375", "NOAA-20"])

        data.append({
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "datetime": dt,
            "satellite": sat,
            "municipio": np.random.choice(cities[region]),
            "bioma": "Caatinga",
            "source": "SYNTHETIC",
        })

    df = pd.DataFrame(data)
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    return df


with st.spinner("Carregando dados..."):
    if data_source == "INPE (online)":
        df = load_data_inpe(
            date_range[0].strftime("%Y-%m-%d"),
            date_range[1].strftime("%Y-%m-%d"),
        )
        if df.empty:
            st.warning(
                "API do INPE não retornou dados. "
                "Mude para 'Demonstração (sintético)' para ver o funcionamento."
            )
            df = load_data_local()
            source_note = "⚠️ Dados sintéticos (demo)"
        else:
            source_note = "✅ Dados do INPE"
    elif data_source == "Dados Locais (CSV)":
        # Verificar se existe arquivo local
        csv_files = []
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        if os.path.exists(data_dir):
            csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

        if csv_files:
            selected = st.sidebar.selectbox("Arquivo CSV", csv_files)
            df = load_local_fire_data(os.path.join(data_dir, selected))
            source_note = f"📁 {selected}"
        else:
            st.info("Nenhum CSV encontrado em data/. Usando demo.")
            df = load_data_local()
            source_note = "⚠️ Dados sintéticos (demo)"
    else:
        df = load_data_local()
        source_note = "⚠️ Dados sintéticos (demo)"


# ============================================================================
# Indicadores principais
# ============================================================================

if not df.empty:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔥 Total de Focos", len(df))

    with col2:
        if "month" in df.columns:
            month_active = df[df["month"] == datetime.now().month]
            st.metric("📅 Focos no mês atual", len(month_active))
        else:
            st.metric("📅 Focos no mês atual", "N/A")

    with col3:
        if not df.empty and "lat" in df.columns:
            coords = df[["lat", "lon"]].dropna()
            unique_locs = (coords.round(2).drop_duplicates().shape[0])
            st.metric("📍 Localizações únicas", unique_locs)
        else:
            st.metric("📍 Localizações únicas", "N/A")

    with col4:
        st.metric("📊 Fonte", source_note)

    st.divider()

    # ============================================================================
    # Abas
    # ============================================================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Mapa de Calor",
        "📈 Análise Temporal",
        "🤖 Gêmeo Digital",
        "📋 Relatório",
    ])

    # --- TAB 1: Mapa ---
    with tab1:
        st.subheader("Mapa de Calor de Queimadas no Ceará")

        if "lat" in df.columns:
            col_map1, col_map2 = st.columns([3, 1])

            with col_map2:
                st.caption("Filtros do mapa")
                if "satellite" in df.columns:
                    sats = ["Todos"] + sorted(df["satellite"].unique().tolist())
                    selected_sat = st.selectbox("Satélite", sats, key="map_sat")
                else:
                    selected_sat = "Todos"

                if "bioma" in df.columns:
                    biomas = ["Todos"] + sorted(df["bioma"].unique().tolist())
                    selected_bioma = st.selectbox("Bioma", biomas, key="map_bioma")
                else:
                    selected_bioma = "Todos"

                st.caption("Áreas Críticas Monitoradas")
                for area_name, area_info in AREAS_CRITICAS.items():
                    risco = area_info["nivel_risco"]
                    emoji = "🔴" if risco == "ALTO" else "🟡" if risco == "MÉDIO" else "🟢"
                    st.markdown(f"{emoji} **{area_name}** — {risco}")

            with col_map1:
                df_map = df.copy()
                if selected_sat != "Todos" and "satellite" in df_map.columns:
                    df_map = df_map[df_map["satellite"] == selected_sat]
                if selected_bioma != "Todos" and "bioma" in df_map.columns:
                    df_map = df_map[df_map["bioma"] == selected_bioma]

                if len(df_map) > 5000:
                    df_map = df_map.sample(5000, random_state=42)

                try:
                    import folium
                    from streamlit_folium import st_folium

                    # Centro do Ceará
                    m = folium.Map(
                        location=[-5.2, -39.5],
                        zoom_start=7,
                        tiles="CartoDB positron",
                    )

                    # Camada de calor
                    heat_data = df_map[["lat", "lon"]].dropna().values.tolist()
                    from folium.plugins import HeatMap

                    HeatMap(
                        heat_data,
                        radius=12,
                        blur=15,
                        max_zoom=1,
                        gradient={
                            0.4: "blue",
                            0.6: "yellow",
                            0.8: "orange",
                            1.0: "red",
                        },
                    ).add_to(m)

                    # Áreas críticas
                    for area_name, area_info in AREAS_CRITICAS.items():
                        lat, lon = area_info["centroid"]
                        color = "red" if area_info["nivel_risco"] == "ALTO" else "orange"
                        folium.Marker(
                            [lat, lon],
                            popup=f"<b>{area_name}</b><br>{area_info['descricao']}",
                            icon=folium.Icon(color=color, icon="warning-sign"),
                        ).add_to(m)

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
                            "color": "black",
                            "weight": 2,
                            "dashArray": "5, 5",
                        },
                    ).add_to(m)

                    st_folium(m, width=700, height=500)

                except ImportError:
                    st.info("""
                    Para ver o mapa interativo, instale:
                    ```
                    pip install folium streamlit-folium
                    ```
                    """)

                    # Fallback: gráfico de dispersão
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, 8))
                    ax.scatter(
                        df_map["lon"], df_map["lat"],
                        alpha=0.3, s=5, c="red",
                    )
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    ax.set_title("Focos de Queimadas no Ceará")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
        else:
            st.warning("Dados sem coordenadas geográficas.")

    # --- TAB 2: Análise Temporal ---
    with tab2:
        st.subheader("Análise Temporal")

        analysis = FireAnalysis(df)

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            monthly = analysis.monthly_distribution()
            if not monthly.empty:
                st.subheader("Distribuição Mensal")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ["#ff6b35" if m >= 6 else "#4ecdc4" for m in monthly["month"]]
                bars = ax.bar(monthly["month"], monthly["count"], color=colors)
                ax.set_xlabel("Mês")
                ax.set_ylabel("Número de Focos")
                ax.set_xticks(range(1, 13))
                ax.set_xticklabels([
                    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                    "Jul", "Ago", "Set", "Out", "Nov", "Dez",
                ])
                ax.grid(True, alpha=0.3, axis="y")

                # Legenda
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor="#ff6b35", label="Estação Seca (Jun-Dez)"),
                    Patch(facecolor="#4ecdc4", label="Estação Chuvosa (Jan-Mai)"),
                ]
                ax.legend(handles=legend_elements, loc="upper right")
                st.pyplot(fig)

        with col_t2:
            season = analysis.peak_season()
            if season:
                st.subheader("Sazonalidade")
                st.metric("Mês de Pico", season["peak_month"])
                st.metric("Focos na Estação Seca",
                          f"{season['dry_season_total']} ({season['dry_season_pct']}%)")
                st.metric("Focos na Estação Chuvosa", season["wet_season_total"])

                st.info(
                    f"🔍 **Interpretação**: "
                    f"{season['dry_season_pct']:.0f}% dos focos ocorrem na estação seca "
                    f"(junho a dezembro), quando a vegetação da Caatinga está mais "
                    f"suscetível ao fogo."
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

    # --- TAB 3: Gêmeo Digital ---
    with tab3:
        st.subheader("🤖 Simulação do Gêmeo Digital")

        if enable_simulation:
            col_twin1, col_twin2 = st.columns([2, 1])

            with col_twin2:
                st.caption("Configuração da Simulação")

                # Parâmetros ajustáveis
                spread_rate = st.slider("Taxa de propagação", 0.1, 2.0, 1.0, 0.1,
                                        help="Multiplicador da velocidade de propagação do fogo")
                dry_factor = st.slider("Fator de seca (vegetação)", 0.0, 1.0, 0.6, 0.1,
                                       help="Nível de seca da vegetação (0=úmida, 1=muito seca)")

                if st.button("▶️ Executar Simulação", type="primary"):
                    st.session_state["run_simulation"] = True
                else:
                    if "run_simulation" not in st.session_state:
                        st.session_state["run_simulation"] = False

            with col_twin1:
                if st.session_state.get("run_simulation"):
                    with st.spinner("Simulando propagação do fogo..."):
                        # Inicializar twin
                        twin = FireDigitalTwin(resolution=0.05)

                        # Alimentar com dados históricos
                        twin.initialize_from_history(df)

                        # Adicionar focos recentes (últimos 7 dias)
                        recent = df.copy()
                        if "datetime" in recent.columns:
                            recent = recent[
                                recent["datetime"] >= pd.Timestamp.now() - pd.Timedelta(days=7)
                            ]
                        twin.add_active_fires(recent)

                        # Executar simulação
                        history = twin.simulate(steps=sim_steps)

                        # Mostrar resultados
                        final = history[-1]
                        c1, c2, c3 = st.columns(3)
                        c1.metric("🔥 Em chamas", final["burning_cells"])
                        c2.metric("🔥 Já queimado", final["burned_cells"])
                        c3.metric("📍 Total afetado", final["total_affected"])

                        # Gráfico da evolução
                        import matplotlib.pyplot as plt

                        fig, ax = plt.subplots(figsize=(10, 4))
                        df_hist = pd.DataFrame(history)

                        ax.plot(df_hist["step"], df_hist["burning_cells"],
                                label="Em chamas", color="#ff6b35")
                        ax.plot(df_hist["step"], df_hist["burned_cells"],
                                label="Já queimado", color="#8b0000")
                        ax.plot(df_hist["step"], df_hist["total_affected"],
                                label="Total afetado", color="#2c3e50", linewidth=2)

                        ax.set_xlabel("Passo da simulação")
                        ax.set_ylabel("Número de células")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)

                        # Zonas de perigo
                        zones = twin.get_fire_danger_zones(threshold=0.5)

                        if zones:
                            st.subheader("🚨 Zonas de Alto Risco Detectadas")
                            zones_df = pd.DataFrame(zones[:5])
                            zones_df["estimated_area_km2"] = zones_df["estimated_area_km2"].round(1)
                            st.dataframe(
                                zones_df,
                                column_config={
                                    "centroid_lat": "Latitude",
                                    "centroid_lon": "Longitude",
                                    "risk_level": "Risco",
                                    "area_cells": "Células",
                                    "estimated_area_km2": "Área (km²)",
                                },
                                hide_index=True,
                                use_container_width=True,
                            )

                        # Áreas críticas
                        critical = twin.check_critical_areas()
                        st.subheader("🏞️ Status das Áreas Críticas")
                        for area in critical:
                            risco = area["nivel_risco"]
                            emoji = "🔴" if risco == "ALTO" else "🟡" if risco == "MÉDIO" else "🟢"
                            alert = "🔥 ATIVO" if area["em_chamas"] else "✅ OK"
                            st.markdown(
                                f"{emoji} **{area['area']}** ({area['bioma']}) "
                                f"— Risco: {risco} | Focos históricos: {area['focos_historicos']} | {alert}"
                            )
                else:
                    st.info(
                        '👈 Clique em **"Executar Simulação"** ao lado para iniciar '
                        "o gêmeo digital. A simulação usará os dados carregados para "
                        "modelar a propagação do fogo no território cearense."
                    )

                    # Preview do que o twin faz
                    st.markdown("""
                    **O que o Gêmeo Digital faz:**
                    1. **Carrega** focos históricos e atuais do INPE
                    2. **Calcula** densidade histórica de queimadas
                    3. **Simula** propagação do fogo usando autômato celular
                    4. **Identifica** zonas de alto risco
                    5. **Monitora** áreas críticas em tempo real

                    **Fatores considerados:**
                    - 🔥 Densidade histórica de queimadas na região
                    - 🌿 Tipo de cobertura vegetal (combustível)
                    - 🏜️ Nível de seca da vegetação
                    - 📍 Proximidade de focos ativos
                    """)
        else:
            st.info("Ative 'Ativar simulação (Digital Twin)' na barra lateral para usar esta aba.")

    # --- TAB 4: Relatório ---
    with tab4:
        st.subheader("📋 Relatório Resumido")

        analysis = FireAnalysis(df)
        report = analysis.summary_report()

        if "error" not in report:
            col_r1, col_r2 = st.columns(2)

            with col_r1:
                st.markdown("### 📊 Estatísticas Gerais")
                st.markdown(f"- **Total de focos:** {report['total_focos']}")
                if "periodo" in report:
                    st.markdown(f"- **Período:** {report['periodo'].get('inicio', 'N/A')} até {report['periodo'].get('fim', 'N/A')}")
                    st.markdown(f"- **Dias analisados:** {report['periodo'].get('dias', 'N/A')}")
                if "media_anual" in report:
                    st.markdown(f"- **Média anual:** {report['media_anual']} focos")
                if "anos" in report:
                    st.markdown(f"- **Anos com dados:** {', '.join(map(str, report['anos']))}")

            with col_r2:
                if "sazonalidade" in report:
                    s = report["sazonalidade"]
                    st.markdown("### 🌦️ Sazonalidade")
                    st.markdown(f"- **Mês de pico:** {s['peak_month']}")
                    st.markdown(f"- **Estação seca (Jun-Dez):** {s['dry_season_pct']:.0f}% dos focos")
                    st.markdown(f"- **Estação chuvosa (Jan-Mai):** {s['wet_season_total']} focos")

                if "satelites" in report:
                    st.markdown("### 🛰️ Satélites Detectores")
                    for sat in report["satelites"][:4]:
                        name = list(sat.values())[0]
                        count = list(sat.values())[1]
                        st.markdown(f"- **{name}:** {count} detecções")

            if "top_biomas" in report:
                st.markdown("### 🌿 Biomas mais afetados")
                cols = st.columns(len(report["top_biomas"]))
                for i, b in enumerate(report["top_biomas"]):
                    with cols[i]:
                        st.metric(b["bioma"], f"{b['focos']} focos")

            # Top municípios
            top_muni = analysis.top_municipios(10)
            if not top_muni.empty and "municipio" in top_muni.columns:
                st.markdown("### 🏙️ Municípios com mais focos")
                st.dataframe(top_muni, hide_index=True, use_container_width=True)
        else:
            st.warning("Não foi possível gerar o relatório. Verifique os dados carregados.")

else:
    st.warning("Nenhum dado disponível. Verifique as fontes de dados ou use o modo demonstração.")

# ============================================================================
# Footer
# ============================================================================

st.divider()
st.caption(
    "**Proposta Funcional — Digital Twin para Queimadas no Ceará** | "
    "Dados abertos: INPE, NASA FIRMS, ESA Copernicus | "
    f"Última atualização: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
)
