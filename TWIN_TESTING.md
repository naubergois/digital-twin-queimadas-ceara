# Testes e redução de falsos positivos no gêmeo digital

Este documento descreve **o que foi testado**, **como reproduzir** e **quais técnicas** endereçam erro do modelo (especialmente falsos positivos).

## Execução

```bash
cd /path/to/digital-twin-queimadas-ceara
pip install -r requirements.txt
pytest tests/ -q --ignore-glob='*slow*'
```

Teste de ponta a ponta (treino completo, mais lento):

```bash
pytest tests/test_ml_digital_twin.py::test_validate_with_real_data_runs_on_synthetic_series -q
```

## Modos de operação (`main.py`)

| Modo | Objetivo |
|------|-----------|
| `operational` | Recall alto na validação; calibração maximiza F1 com restrições suaves. |
| `strict_cell` | F1 por célula no holdout, limiar mais restrito via validação interna. |
| `low_fp` | **Menos alarmes falsos**: `calibration_objective="low_fp"` prioriza **precisão** mantendo recall mínimo; `max_positive_rate` menor; mais negativos difíceis no mining (`hnm_neg_pos_ratio=10`). |

Uso:

```bash
python main.py --ml-validate --ml-mode low_fp --local data/seu_arquivo.csv
```

## Técnicas implementadas (iteração sobre falhas)

1. **Calibração por objetivo** (`MLTwinConfig.calibration_objective`): em `low_fp`, o limiar de probabilidade ML escolhido maximiza precisão entre candidatos que respeitam `min_recall_target` e `min_precision_target`; o limiar do twin espacial maximiza `twin_precision` com recall ≥ ~90% do alvo.
2. **Limiar do twin mais alto**: propagação só quando `risk` ≥ `twin_spread_threshold`; valores maiores cortam ignição por vizinhança fraca (menos “manchas” fantasma).
3. **Peso de custo / hard negative mining**: já existentes; no modo `low_fp` o preset aumenta `hnm_neg_pos_ratio` e `positive_class_weight` para empurrar o classificador contra FP.
4. **Testes de regressão**: `test_simulate_twin_high_threshold_reduces_spread_vs_low` garante que limiar alto não aumenta área queimada versus baixo no mesmo cenário sintético.

## Métricas a acompanhar em dados reais

- **Precisão / recall por célula-dia** (`ml_metrics`).
- **`precision_tolerant` / `recall_tolerant`**: erro de localização de até `tolerant_radius_cells` (operacional).
- **Twin**: `twin_metrics.precision`, `twin_spatial_iou_mean`, detecção diária em `real_data_comparison.day_level_detection`.

Quando o twin “erra muito”, registrar no relatório JSON: `fp`, `precision`, período de teste e comparar `operational` vs `low_fp` no mesmo CSV.

## Arquivos

- `tests/test_ml_digital_twin.py` — grade, `_simulate_twin`, calibração `low_fp`, smoke de `validate_with_real_data`.
- `tests/test_digital_twin.py` — autômato clássico, inicialização e reprodutibilidade com seed.
