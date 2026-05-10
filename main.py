#!/usr/bin/env python3
"""
Pipeline principal — Digital Twin para Queimadas no Ceará.

Uso:
    python main.py                          # Pipeline completo (demo)
    python main.py --year 2024              # Ano específico
    python main.py --api firms              # Usar NASA FIRMS (requer API key)
    python main.py --api goes16             # Focos GOES-16 (INPE) no Ceará
    python main.py --goes-unsupervised      # Twin + predição não supervisionada em cubo GOES-16 (proxy)
    python main.py --st-hypernet          # Twin + ST-HyperNet (PyTorch) no cubo proxy + artefato em data/
    python -m src.compare_st_hypernet_days --csv data/focos.csv  # PNGs real vs ST-HyperNet por dia
    python -m src.run_st_hypernet --csv data/focos.csv  # Só treinar ST-HyperNet (CLI dedicado)
    python main.py --dashboard              # Só abrir o dashboard
"""

import argparse
import json
import os
import sys
from datetime import datetime

import pandas as pd

# Adiciona src ao path
sys.path.insert(0, os.path.dirname(__file__))

from src.fire_data import (
    fetch_inpe_fire_foci,
    fetch_goes16_fire_foci_ceara,
    fetch_inpe_fire_summary,
    download_year_data,
    load_local_fire_data,
    fetch_firms_ceara,
    merge_inpe_firms,
)
from src.digital_twin import FireDigitalTwin
from src.analysis import FireAnalysis
from src.ml_digital_twin import FireMLDigitalTwin, MLTwinConfig
from src.pyro_caatinga import PyroCaatingaConfig, run_pyro_caatinga_mvp


def run_pipeline(
    year: int | None = None,
    use_local_csv: str | None = None,
    api_source: str = "inpe",
    run_ml_validation: bool = False,
    ml_mode: str = "operational",
    goes16_unsupervised: bool = False,
    goes_proxy_netcdf: str | None = None,
    goes_unsup_blend_weight: float = 0.28,
    st_hypernet: bool = False,
    st_hypernet_blend_weight: float = 0.22,
    st_hypernet_epochs: int = 10,
    st_hypernet_out_dir: str | None = None,
    st_hypernet_device: str = "cpu",
    st_hypernet_max_days_history: int = 45,
) -> dict:
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
        # Demo: baixar últimos dias conforme fonte selecionada
        if api_source == "goes16":
            df = fetch_goes16_fire_foci_ceara()
        elif api_source == "firms":
            df = fetch_firms_ceara(days=3)
        elif api_source == "inpe_firms":
            df_inpe = fetch_inpe_fire_foci(state_code="23")
            df_firms = fetch_firms_ceara(days=3)
            df = merge_inpe_firms(df_inpe, df_firms)
        else:
            df = fetch_inpe_fire_foci(state_code="23")

    if df.empty:
        print("\n❌ Nenhum dado real encontrado (INPE/csv/ano). Sem dados sintéticos.")
        print("    Verifique rede, período ou use: python main.py --year 2024")
        print("    ou --local data/seu_arquivo.csv")
        sys.exit(1)

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

    ml_result = None
    if run_ml_validation:
        print("\n[2b/4] Validação ML + Gêmeo Digital (dados reais)...")
        mode = ml_mode if ml_mode in ("operational", "strict_cell", "low_fp") else "operational"
        if mode == "strict_cell":
            cfg = MLTwinConfig(
                grid_resolution=0.2,
                lookback_days=3,
                test_ratio=0.2,
                proba_threshold=0.5,
                twin_spread_threshold=0.45,
                auto_calibrate=True,
                min_recall_target=0.2,
                min_precision_target=0.15,
                use_deep_temporal=True,
                ensemble_weight_gb=0.6,
                optimize_metric="f1",
                tolerant_radius_cells=0,
                max_positive_rate=0.12,
                mode="strict_cell",
                use_hard_negative_mining=True,
                hnm_neg_pos_ratio=8,
            )
        elif mode == "low_fp":
            cfg = MLTwinConfig(
                grid_resolution=0.2,
                lookback_days=3,
                test_ratio=0.2,
                proba_threshold=0.45,
                twin_spread_threshold=0.55,
                auto_calibrate=True,
                min_recall_target=0.55,
                min_precision_target=0.12,
                use_deep_temporal=True,
                ensemble_weight_gb=0.6,
                optimize_metric="f1",
                tolerant_radius_cells=4,
                max_positive_rate=0.12,
                mode="operational",
                use_hard_negative_mining=True,
                hnm_neg_pos_ratio=10,
                calibration_objective="low_fp",
                positive_class_weight=12.0,
            )
        else:
            cfg = MLTwinConfig(
                grid_resolution=0.2,
                lookback_days=3,
                test_ratio=0.2,
                proba_threshold=0.35,
                twin_spread_threshold=0.45,
                auto_calibrate=True,
                min_recall_target=0.7,
                min_precision_target=0.08,
                use_deep_temporal=True,
                ensemble_weight_gb=0.6,
                optimize_metric="f1",
                tolerant_radius_cells=4,
                max_positive_rate=0.25,
                mode="operational",
                use_hard_negative_mining=True,
                hnm_neg_pos_ratio=6,
                calibration_objective="f1",
            )
        ml_twin = FireMLDigitalTwin(
            cfg
        )
        ml_result = ml_twin.validate_with_real_data(df)
        print(
            "  ✓ ML AUC: "
            f"{ml_result['ml_metrics']['roc_auc']:.3f} | "
            f"PR-AUC: {ml_result['ml_metrics']['pr_auc']:.3f} | "
            f"F1(célula): {ml_result['ml_metrics']['f1']:.3f} | "
            f"F1(diário): {ml_result['ml_metrics'].get('f1_day', float('nan')):.3f} | "
            f"F1 tolerante: {ml_result['ml_metrics'].get('f1_tolerant', float('nan')):.3f}"
        )
        print(
            "  ✓ Twin F1: "
            f"{ml_result['twin_metrics']['f1']:.3f} | "
            f"F1(diário): {ml_result['twin_metrics'].get('f1_day', float('nan')):.3f} | "
            f"F1 tolerante: {ml_result['twin_metrics'].get('f1_tolerant', float('nan')):.3f} | "
            f"IoU espacial médio: {ml_result['twin_spatial_iou_mean']:.3f}"
        )
        cmp = ml_result.get("real_data_comparison", {})
        if cmp:
            d = cmp.get("day_level_detection", {})
            print(
                "  ✓ Detecção diária (real): "
                f"ML {d.get('ml_days_detected', 0)}/{d.get('days_with_real_fire', 0)} "
                f"({100*d.get('ml_day_detection_rate', 0):.1f}%) | "
                f"Twin {d.get('twin_days_detected', 0)}/{d.get('days_with_real_fire', 0)} "
                f"({100*d.get('twin_day_detection_rate', 0):.1f}%)"
            )

    goes_unsup_report = None
    if goes16_unsupervised:
        from src.goes_unsupervised_twin import GOESUnsupervisedConfig, export_public_report, run_goes16_unsupervised_from_foci

        print("\n[2c/4] GOES-16 — predição não supervisionada (cubo imagem proxy ABI)...")
        gcfg = GOESUnsupervisedConfig()
        goes_unsup_report = run_goes16_unsupervised_from_foci(
            df,
            cfg=gcfg,
            netcdf_path=goes_proxy_netcdf,
        )
        print(
            f"  ✓ Risco satélite (Isolation Forest + resíduo ΔBT): "
            f"max={goes_unsup_report['max_risk']:.3f} | "
            f"células≥0.5={goes_unsup_report['cells_above_0.5']} | "
            f"picos={len(goes_unsup_report['top_peaks'])}"
        )

    st_report = None
    if st_hypernet:
        from src.st_hypernet import (
            STHyperNetConfig,
            export_public_st_report,
            run_st_hypernet_pipeline,
            save_st_hypernet_artifact,
            write_st_hypernet_best_params_json,
        )

        print("\n[2d/4] ST-HyperNet — campo de fundo + ruptura (treino self-supervised, PyTorch)...")
        st_dir = st_hypernet_out_dir or os.path.join("data", "st_hypernet_last")
        os.makedirs(st_dir, exist_ok=True)
        st_cfg = STHyperNetConfig(
            epochs=max(2, int(st_hypernet_epochs)),
            device=str(st_hypernet_device),
            grid_resolution=0.5,
            frame_minutes=60,
            max_days_history=max(7, int(st_hypernet_max_days_history)),
            max_patches_per_epoch=4096,
        )
        st_report = run_st_hypernet_pipeline(df, cfg=st_cfg)
        save_st_hypernet_artifact(st_report, os.path.join(st_dir, "st_hypernet.pt"))
        write_st_hypernet_best_params_json(
            st_report,
            os.path.join(st_dir, "st_hypernet_best_params.json"),
            extras={"source": "main.py pipeline", "st_dir": st_dir},
        )
        print(
            f"  ✓ ST-HyperNet: score_max={st_report['max_score']:.3f} | "
            f"loss≈{st_report.get('train_meta', {}).get('loss_last', 0):.4f} | "
            f"artefatos em {os.path.abspath(st_dir)}"
        )

    # ── Step 3: Digital Twin ───────────────────────────────────────────
    print("\n[3/4] Simulação Digital Twin...")

    twin = FireDigitalTwin(resolution=0.05)
    twin.initialize_from_history(df)
    if goes_unsup_report is not None:
        twin.blend_goes16_unsupervised_risk(goes_unsup_report, weight=goes_unsup_blend_weight)
    if st_report is not None:
        twin.blend_st_hypernet_risk(st_report, weight=st_hypernet_blend_weight)
    twin.add_active_fires(df)  # focos recentes como ignição para simulação

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

    if goes_unsup_report is not None:
        from src.goes_unsupervised_twin import export_public_report

        result["goes16_unsupervised"] = export_public_report(goes_unsup_report)

    if st_report is not None:
        from src.st_hypernet import export_public_st_report

        result["st_hypernet"] = export_public_st_report(st_report)

    if ml_result is not None:
        ml_for_bundle = {k: v for k, v in ml_result.items() if k != "best_model_params"}
        result["ml_digital_twin_validation"] = ml_for_bundle

    result_path = "data/pipeline_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if ml_result is not None:
        ml_path = "data/ml_twin_validation.json"
        ml_to_save = {k: v for k, v in ml_result.items() if k != "best_model_params"}
        with open(ml_path, "w", encoding="utf-8") as f:
            json.dump(ml_to_save, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Validação ML salva em {ml_path}")
        ml_bp = ml_result.get("best_model_params")
        if ml_bp:
            bp_path = "data/ml_twin_best_params.json"
            with open(bp_path, "w", encoding="utf-8") as f:
                json.dump(ml_bp, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Parâmetros do melhor modelo ML (config + limiares + estimator) em {bp_path}")

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
    parser.add_argument("--api", choices=["inpe", "firms", "goes16", "inpe_firms"], default="inpe",
                        help="Fonte de dados")
    parser.add_argument(
        "--ml-validate",
        action="store_true",
        help="Executa técnica inovadora de gêmeo digital com ML e validação temporal",
    )
    parser.add_argument(
        "--ml-mode",
        choices=["operational", "strict_cell", "low_fp"],
        default="operational",
        help="Modo ML: operacional, strict_cell (F1 célula) ou low_fp (menos falsos positivos, calibração por precisão)",
    )
    parser.add_argument(
        "--pyro-mvp",
        action="store_true",
        help="Executa o pipeline PYRO-Caatinga MVP (climatologia residual + destilação + loop twin)",
    )
    parser.add_argument(
        "--pyro-goes-csv",
        type=str,
        default="data/focos_CE_GOES16_2024.csv",
        help="CSV base para o PYRO-Caatinga MVP",
    )
    parser.add_argument(
        "--pyro-viirs-csv",
        type=str,
        default="",
        help="CSV VIIRS opcional para destilação cruzada",
    )
    parser.add_argument(
        "--pyro-output-dir",
        type=str,
        default="data/pyro_caatinga",
        help="Diretório de saída dos artefatos do PYRO-Caatinga",
    )
    parser.add_argument(
        "--pyro-max-days",
        type=int,
        default=7,
        help="Janela máxima (dias) para montar o cubo temporal do PYRO-Caatinga",
    )
    parser.add_argument(
        "--goes-unsupervised",
        action="store_true",
        help="Predição não supervisionada em cubo GOES-16 (proxy ABI) e fusão no risco do twin",
    )
    parser.add_argument(
        "--goes-proxy-netcdf",
        type=str,
        default="",
        help="NetCDF opcional (ex.: data/pyro_caatinga/pyro_goes_proxy_cube.nc) em vez de montar cubo do CSV",
    )
    parser.add_argument(
        "--goes-unsup-blend",
        type=float,
        default=0.28,
        help="Peso [0–1] da grade GOES não supervisionada ao mesclar com risk_grid do twin",
    )
    parser.add_argument(
        "--st-hypernet",
        action="store_true",
        help="Treina ST-HyperNet no cubo GOES-proxy, salva data/st_hypernet_last/ e mescla risco no twin",
    )
    parser.add_argument(
        "--st-hypernet-blend",
        type=float,
        default=0.22,
        help="Peso [0–1] da grade ST-HyperNet (max temporal) no risk_grid",
    )
    parser.add_argument(
        "--st-hypernet-epochs",
        type=int,
        default=10,
        help="Épocas de treino ST-HyperNet no pipeline",
    )
    parser.add_argument(
        "--st-hypernet-out",
        type=str,
        default="",
        help="Diretório para st_hypernet.pt (default: data/st_hypernet_last)",
    )
    parser.add_argument(
        "--st-hypernet-device",
        type=str,
        default="cpu",
        help="cpu ou cuda para ST-HyperNet",
    )
    parser.add_argument(
        "--st-hypernet-history-days",
        type=int,
        default=45,
        help="Janela temporal (dias) do cubo GOES-proxy para ST-HyperNet no pipeline",
    )

    args = parser.parse_args()

    if args.dashboard:
        print("Abrindo dashboard...")
        os.system("streamlit run dashboard/app.py")
        return

    if args.pyro_mvp:
        print("Executando PYRO-Caatinga MVP...")
        if not os.path.exists(args.pyro_goes_csv):
            raise FileNotFoundError(f"CSV GOES não encontrado: {args.pyro_goes_csv}")

        df_goes = pd.read_csv(args.pyro_goes_csv)
        df_viirs = pd.read_csv(args.pyro_viirs_csv) if args.pyro_viirs_csv and os.path.exists(args.pyro_viirs_csv) else None

        pyro_cfg = PyroCaatingaConfig(max_days_history=max(1, int(args.pyro_max_days)))
        report = run_pyro_caatinga_mvp(
            goes_df=df_goes,
            viirs_df=df_viirs,
            output_dir=args.pyro_output_dir,
            cfg=pyro_cfg,
        )

        print("✓ PYRO-Caatinga MVP concluído")
        print("✓ Métricas:")
        print(json.dumps(report.get("metrics", {}), ensure_ascii=False, indent=2))
        print(f"✓ Artefatos em: {args.pyro_output_dir}")
        return

    run_pipeline(
        year=args.year,
        use_local_csv=args.local,
        api_source=args.api,
        run_ml_validation=args.ml_validate,
        ml_mode=args.ml_mode,
        goes16_unsupervised=args.goes_unsupervised,
        goes_proxy_netcdf=args.goes_proxy_netcdf or None,
        goes_unsup_blend_weight=float(args.goes_unsup_blend),
        st_hypernet=args.st_hypernet,
        st_hypernet_blend_weight=float(args.st_hypernet_blend),
        st_hypernet_epochs=int(args.st_hypernet_epochs),
        st_hypernet_out_dir=args.st_hypernet_out or None,
        st_hypernet_device=str(args.st_hypernet_device),
        st_hypernet_max_days_history=int(args.st_hypernet_history_days),
    )


if __name__ == "__main__":
    main()
