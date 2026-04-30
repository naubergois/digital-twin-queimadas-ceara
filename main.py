#!/usr/bin/env python3
"""
Pipeline principal — Digital Twin para Queimadas no Ceará.

Uso:
    python main.py                          # Pipeline completo (demo)
    python main.py --year 2024              # Ano específico
    python main.py --api firma              # Usar NASA FIRMS (requer API key)
    python main.py --dashboard              # Só abrir o dashboard
"""

import argparse
import json
import os
import sys
from datetime import datetime

# Adiciona src ao path
sys.path.insert(0, os.path.dirname(__file__))

from src.fire_data import (
    fetch_inpe_fire_foci,
    fetch_inpe_fire_summary,
    download_year_data,
    load_local_fire_data,
)
from src.digital_twin import FireDigitalTwin
from src.analysis import FireAnalysis


def run_pipeline(year: int | None = None, use_local_csv: str | None = None) -> dict:
    """
    Executa o pipeline completo:
      1. Coleta dados
      2. Análise exploratória
      3. Simulação Digital Twin
      4. Exporta resultados

    Returns:
        Dict com resumo da execução
    """
    print("=" * 60)
    print("  🔥 DIGITAL TWIN — QUEIMADAS CEARÁ")
    print("=" * 60)

    # ── Step 1: Coleta ─────────────────────────────────────────────────
    print("\n[1/4] Coleta de dados...")

    if use_local_csv:
        df = load_local_fire_data(use_local_csv)
    elif year:
        df = download_year_data(year, output_dir="data")
    else:
        # Demo: baixar últimos 30 dias
        df = fetch_inpe_fire_foci(state_code="23")

    if df.empty:
        print("\n⚠️  Nenhum dado encontrado via API. Gerando dados sintéticos...")
        import numpy as np
        import pandas as pd

        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            "lat": np.random.uniform(-7.5, -2.8, n),
            "lon": np.random.uniform(-41.0, -37.5, n),
            "datetime": pd.date_range(end=pd.Timestamp.now(), periods=n, freq="3h"),
            "satellite": np.random.choice(["AQUA_M-T", "TERRA_M-T", "NPP-375"], n),
            "municipio": np.random.choice(["Tauá", "Crateús", "Sobral", "Juazeiro"], n),
            "bioma": "Caatinga",
            "source": "SYNTHETIC",
        })

    # Salvar dados brutos
    os.makedirs("data", exist_ok=True)
    csv_path = "data/focos_carregados.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ {len(df)} focos salvos em {csv_path}")

    # ── Step 2: Análise ─────────────────────────────────────────────────
    print("\n[2/4] Análise exploratória...")

    analysis = FireAnalysis(df)
    report = analysis.summary_report()

    print(f"  ✓ Período: {report.get('periodo', {}).get('inicio', 'N/A')} → "
          f"{report.get('periodo', {}).get('fim', 'N/A')}")
    print(f"  ✓ Total de focos: {report['total_focos']}")

    season = analysis.peak_season()
    if season:
        print(f"  ✓ Estação seca: {season['dry_season_pct']:.0f}% dos focos")
        print(f"  ✓ Mês de pico: {season['peak_month']}")

    # ── Step 3: Digital Twin ───────────────────────────────────────────
    print("\n[3/4] Simulação Digital Twin...")

    twin = FireDigitalTwin(resolution=0.05)
    twin.initialize_from_history(df)
    twin.add_active_fires(df)  # usar todos como ativos para demo

    history = twin.simulate(steps=24)
    final = history[-1]

    print(f"  ✓ {final['total_affected']} células afetadas")
    print(f"  ✓ {final['burning_cells']} ainda em chamas")

    # ── Step 4: Exporta ─────────────────────────────────────────────────
    print("\n[4/4] Exportando resultados...")

    state_path = "data/twin_state_latest.json"
    twin.export_state(state_path)

    # Relatório consolidado
    zones = twin.get_fire_danger_zones(threshold=0.5)
    critical = twin.check_critical_areas()

    result = {
        "timestamp": datetime.now().isoformat(),
        "data": {
            "total_focos": int(report["total_focos"]),
            "fonte": csv_path,
        },
        "digital_twin": {
            "grid_shape": (twin.n_lat, twin.n_lon),
            "simulation_steps": 24,
            "burning_cells": final["burning_cells"],
            "burned_cells": final["burned_cells"],
            "total_affected": final["total_affected"],
        },
        "danger_zones": zones[:5],
        "critical_areas": critical,
    }

    result_path = "data/pipeline_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Resultados salvos em {result_path}")
    print(f"  ✓ Estado do twin em {state_path}")

    print("\n" + "=" * 60)
    print("  ✅ Pipeline concluído com sucesso!")
    print("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Digital Twin para Detecção de Queimadas no Ceará"
    )
    parser.add_argument("--year", type=int, help="Ano para download completo")
    parser.add_argument("--local", type=str, help="Carregar CSV local")
    parser.add_argument(
        "--dashboard", action="store_true", help="Abrir dashboard Streamlit"
    )
    parser.add_argument("--api", choices=["inpe", "firms"], default="inpe",
                        help="Fonte de dados")

    args = parser.parse_args()

    if args.dashboard:
        print("Abrindo dashboard...")
        os.system("streamlit run dashboard/app.py")
        return

    run_pipeline(year=args.year, use_local_csv=args.local)


if __name__ == "__main__":
    main()
