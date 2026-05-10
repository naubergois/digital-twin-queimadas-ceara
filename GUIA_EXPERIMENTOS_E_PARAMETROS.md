# Diretrizes: experimentos, artefatos e melhores parâmetros

Este documento fixa **como salvar**, **onde guardar** e **como manter** os resultados de experimentos e os **melhores parâmetros** de cada modelo no repositório. Complementa o protocolo operacional gerado por `run_experiments` (`EXPERIMENT_PROTOCOL.md`).

---

## 1. Princípios

1. **Reprodutibilidade**: cada execução relevante deve deixar rastro em disco (JSON/CSV/PT) com data, comando ou nota curta na mensagem de commit.
2. **Separação de papéis**:
   - **Métricas e rankings** → resumos agregados (`*_summary.csv`, `EXPERIMENTS.md`).
   - **Estado treinado / pesos** → arquivos binários (`.pt`) ou modelos serializados.
   - **Parâmetros “melhores”** → sempre um JSON dedicado por modelo ou por run, **além** dos relatórios gerais, para inspeção rápida e versionamento.
3. **Não misturar** no mesmo arquivo JSON gigante tanto o relatório completo quanto o bloco de hiperparâmetros, quando isso prejudicar leitura ou diff: use o arquivo `*_best_params.json` como fonte canônica de parâmetros.
4. **Dados de entrada**: manter referência ao CSV (ou NetCDF) usado e, quando aplicável, cópia ou manifesto na pasta do experimento (`dataset_fires_detail.*`, `input_fire_dataset`).

---

## 2. Fluxo de experimentos ML (`run_experiments`)

| Artefato | Local | Conteúdo |
|----------|--------|-----------|
| Resumo tabular | `data/experiments/all_experiments_summary.csv` | Uma linha por técnica / status |
| Bundle agregado | `data/experiments/all_experiments_full.json` | `input_fire_dataset` + lista `experiments` (sem duplicar `best_model_params` no JSON de cada run) |
| Run individual | `data/experiments/runs/<nome>.json` | Métricas, `config` operacional, `model_selection`, etc. **Sem** o bloco `best_model_params` (extraído à parte) |
| **Melhores parâmetros do modelo escolhido** | `data/experiments/runs/<nome>_best_params.json` | `ml_twin_config`, `chosen_operating_point` (limiares calibrados), `best_model_selection`, `fitted_classifier_class` + `fitted_classifier_params` |
| Focos usados | `data/experiments/dataset_fires_detail.json` / `.csv` | Cada queimada com colunas do CSV (incl. `source`) |
| Ranking em Markdown | `EXPERIMENTS.md` | Gerado automaticamente |

**Manutenção**

- Preservar a pasta `data/experiments/runs/` em backup ou no controle de versão conforme política do projeto (dados grandes podem ir para LFS ou armazenamento externo).
- Ao comparar versões do código, comparar também os `*_best_params.json` correspondentes à mesma técnica.

**Comando**

```bash
python -m src.run_experiments --dataset data/focos_CE_GOES16_2024.csv --output-dir data/experiments
```

---

## 3. Gêmeo digital ML no pipeline principal (`main.py`)

| Artefato | Local |
|----------|--------|
| Validação (sem bloco pesado de params) | `data/ml_twin_validation.json` |
| **Melhores parâmetros** | `data/ml_twin_best_params.json` |
| Resultado agregado do pipeline | `data/pipeline_result.json` (trecho ML sem `best_model_params` duplicado) |

---

## 4. ST-HyperNet

### 4.1 Treino isolado (`run_st_hypernet`)

| Artefato | Local |
|----------|--------|
| Pesos + meta mínima | `<out-dir>/st_hypernet.pt` |
| **Melhores parâmetros / meta de treino** | `<out-dir>/st_hypernet_best_params.json` |

### 4.2 Comparação por dias (`compare_st_hypernet_days`)

| Artefato | Local |
|----------|--------|
| Checkpoint | `<out>/<prefix>_best_model.pt` |
| **Parâmetros + extras da corrida** | `<out>/<prefix>_best_params.json` (inclui `metrics_aggregate`, `cube_time_span`, `pred_threshold`, etc.) |
| Métricas por dia | `metrics_by_day.json`, `metrics_by_day.csv`, `metrics_aggregate.json` |
| Figuras | `*_real_vs_pred_*.png`, `*_ceara_map_*.png` / `.html` |

**Manutenção**

- Para cada estudo (ex.: “2024 todos os dias com foco”), usar **um diretório de saída distinto** (`--out`) em vez de sobrescrever pastas de referência sem querer.
- Registrar no commit ou no README interno: comando completo, `--year`, `--max-days-history`, `--epochs`, `--pred-threshold`.

---

## 5. GOES não supervisionado (comparação por dia)

Parâmetros principais ficam no JSON de resumo (`metrics_by_day.json` → `config`) e nas figuras. Se no futuro existir checkpoint explícito, seguir o mesmo padrão: **`<prefix>_best_params.json` + `<prefix>_best_model.*`**.

---

## 6. O que significa “melhores parâmetros” neste projeto

| Modelo | O que é salvo |
|--------|----------------|
| **ML (FireMLDigitalTwin)** | Configuração completa `MLTwinConfig`, limiares finais `proba_threshold` e `twin_spread_threshold`, identidade e `get_params()` do classificador **já ajustado** (após benchmark + HNM). |
| **ST-HyperNet** | `STHyperNetConfig` serializado, estatísticas de normalização (m7/s7/m14/s14), `loss_last` e `loss_curve`, relatório público; o `.pt` guarda `state_dict` + `meta` + `config`. |

Isso permite **reproduzir inferência** e auditar **qual ponto de operação** foi usado, sem depender só do JSON de métricas.

---

## 7. Boas práticas de manutenção

1. **Versionar decisões**: quando um conjunto de parâmetros for “oficial” (ex.: paper ou operação), copiar o `*_best_params.json` correspondente para um nome estável (`reports/baseline_2024_ml_best_params.json`) ou referenciar o hash do commit.
2. **Não apagar runs** sem backup se forem citados em publicações ou dashboards.
3. **Alinhar nomes**: manter `file_prefix` / nome do experimento coerente com pastas e figuras.
4. **Re-executar**: após mudanças em `src/ml_digital_twin.py` ou `src/st_hypernet.py`, regravar experimentos e **substituir** ou **arquivar** pastas antigas com sufixo de data (`_2026-05-10`).
5. **Privacidade**: `dataset_fires_detail` pode conter dados sensíveis; tratar como dado operacional, não necessariamente público.

---

## 8. Referência rápida de comandos

```bash
# Experimentos ML (várias técnicas)
python -m src.run_experiments --dataset data/focos_CE_GOES16_2024.csv

# ST-HyperNet (treino + JSON de parâmetros)
python -m src.run_st_hypernet --csv data/focos_CE_GOES16_2024.csv --out-dir data/st_hypernet

# ST-HyperNet vs focos por dia (+ .pt + best_params.json na pasta --out)
python -m src.compare_st_hypernet_days --csv data/focos_CE_GOES16_2024.csv --out data/st_hypernet_compare
```

---

## 9. Relação com outros documentos

- **`EXPERIMENT_PROTOCOL.md`**: gerado automaticamente ao rodar `run_experiments`; descreve procedimento e lista básica de saídas.
- **`EXPERIMENTS.md`**: ranking gerado automaticamente.
- **Este guia**: política humana de organização e de **sempre** persistir melhores parâmetros por modelo; atualizar manualmente quando novos fluxos ou artefatos forem adicionados ao código.
