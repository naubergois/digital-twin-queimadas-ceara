# Agente de explicação (DeepSeek) — caso a caso

O script `src/compare_case_explainer_agent.py` lê `metrics_by_day.json` (saída de `compare_st_hypernet_days` ou `compare_goes_unsupervised_days`) e pede ao modelo **DeepSeek** um texto em português sobre **o que aconteceu naquele dia** e **como o modelo se comportou face à realidade** (métricas na grade).

## Requisitos

1. Chave API DeepSeek (`DEEPSEEK_API_KEY`). Pode usar ficheiro `.env` na raiz do projeto (já suportado por `python-dotenv`).
2. Pacote `requests` (já em `requirements.txt`).

## Uso

```bash
export DEEPSEEK_API_KEY="sk-..."   # não commite isto

# Explicar os primeiros 5 dias listados no JSON (barato para testar)
python -m src.compare_case_explainer_agent \
  --metrics-dir data/st_hypernet_2024_all_fire_days \
  --out data/st_hypernet_2024_all_fire_days/case_explanations_sample.json \
  --max-cases 5

# Dias concretos
python -m src.compare_case_explainer_agent \
  --metrics-dir data/st_hypernet_2024_all_fire_days \
  --dates 2024-08-09,2024-11-30 \
  --out data/st_hypernet_2024_all_fire_days/case_explanations_dois_dias.json

# Todos os dias do JSON (pode ser longo e custar vários pedidos à API)
python -m src.compare_case_explainer_agent \
  --metrics-dir data/st_hypernet_2024_all_fire_days \
  --out data/st_hypernet_2024_all_fire_days/case_explanations_full.json

# Sem gastar créditos: só valida leitura e selecção
python -m src.compare_case_explainer_agent \
  --metrics-dir data/st_hypernet_2024_all_fire_days \
  --max-cases 2 \
  --dry-run
```

## Saída

Ficheiro JSON com:

- `cases[]`: por dia, `date`, `explanation_pt` (texto do modelo), `input_digest` (métricas enviadas).

O agente **não** envia imagens à API (apenas números e metadados); as figuras PNG/HTML continuam no disco para consulta humana.

## Modelo e endpoint

Por defeito: `deepseek-chat` em `https://api.deepseek.com/v1/chat/completions` (API compatível com OpenAI). Altere com `--model` se a DeepSeek documentar outro nome.

## Privacidade

O prompt contém agregados do dia (TP/FP/FN, IoU, etc.). Não inclui nomes de pessoas; pode incluir `n_focos` e datas. Avalie políticas internas antes de enviar dados sensíveis à nuvem.
