# Documentação da pesquisa e dos testes

Este ficheiro define **como registar** experiências, métricas e decisões ao longo do projeto (gêmeo digital / GOES-16 / INPE). Mantém-se um registo único e reproduzível da evolução da pesquisa.

---

## 1. O que documentar

| Tipo | O quê guardar |
|------|----------------|
| **Corrida de avaliação** | Data ISO (UTC), comando completo, commit Git, parâmetros (`--method`, `--truth-dilate`, canais, horas). |
| **Métricas** | Ficheiro JSON gerado (`--output-json`) ou cópia das linhas impressas (P, R, F1, IoU, Acc). |
| **Figuras** | PNG dos mapas (`--map-dir`) com nome que inclua o método e a data do dia avaliado. |
| **Dados de entrada** | Caminho do CSV INPE, pasta dos NetCDF GOES (`data/goes16_raw/`), versão dos produtos (CMIPF, canais). |
| **Decisão / conclusão** | 2–4 frases: o que melhorou ou piorou e porquê (hipótese). |

---

## 2. Registo por experiência (copiar e preencher)

Crie uma nova secção em **`docs/EVOLUCAO_PESQUISA.md`** (ver secção 6) ou um ficheiro por sprint em **`docs/experimentos/YYYY-MM-DD_slug.md`** usando o modelo:

```markdown
## EXP-YYYYMMDD-xxx — título curto

- **Autor / data:**  
- **Commit:** `git rev-parse --short HEAD`  
- **Objetivo:** (ex.: comparar twin vs residual com dilatação INPE)

### Entrada
- INPE: `data/inpe_focos_ce/...csv`
- GOES: download / `--skip-download` + `data/goes16_raw/`
- Dias UTC: `2024-10-31`, ...

### Comando
```bash
python -m src.unsupervised_fire_goes ...
```

### Saídas
- Métricas: `data/goes16_eval/metrics_....json`
- Mapas: `data/goes16_eval/maps/real_vs_previsto_*.png`

### Resultados (copiar do JSON ou terminal)
| método | P | R | F1 | IoU | Acc |
|--------|---|---|----|-----|-----|
| ... | | | | | |

### Notas
- ...
```

---

## 3. Artefactos do repositório (referência rápida)

| Pasta / ficheiro | Conteúdo |
|------------------|----------|
| `data/goes16_raw/` | NetCDF GOES-16 (CMIPF) |
| `data/goes16_ceara_png/` | PNG térmicos do Ceará (script PNG) |
| `data/inpe_focos_ce/` | CSV INPE agregados por ano / combinado |
| `data/goes16_eval/maps/` | Mapas real vs previsto (TP/FP/FN/TN) |
| `data/goes16_eval/*.json` | Métricas por corrida |
| `src/goes16_download.py` | Download NOAA Open Data (S3) |
| `src/goes16_ceara_image.py` | Recorte Ceará + PNG |
| `src/inpe_queimadas_download.py` | Download focos INPE |
| `src/unsupervised_fire_goes.py` | Avaliação não supervisionada + mapas |
| `src/goes_fire_digital_twin.py` | Gêmeo digital espacial (assimilação GOES) |

---

## 4. Testes automatizados (quando existirem)

- Coloque testes em `tests/` com nomes `test_<módulo>_<comportamento>.py`.
- Ao documentar uma alteração de código, indique **comando**: `pytest tests/ -q` (ou alvo específico).
- Se um teste for **lento** ou precisar de rede, marque com `@pytest.mark.network` ou documente em docstring e **não** falhe o CI por defeito sem dados.

*(Este repositório pode ainda não ter bateria completa de testes; ao adicionar, atualize esta secção.)*

---

## 5. Linha do tempo da pesquisa

### Ficheiro principal sugerido: `docs/EVOLUCAO_PESQUISA.md`

Ordem **cronológica inversa** (mais recente no topo):

```markdown
## 2026-05-10 — resumo do dia
- ...
```

Cada entrada deve ligar a **commits**, **comandos** e **ficheiros de métricas/figuras** quando existirem.

### Versões de dependências

Ao reportar um resultado importante, grave um *freeze* opcional:

```bash
pip freeze > docs/fixtures/requirements-snapshot-YYYYMMDD.txt
```

---

## 6. Criar os ficheiros auxiliares

1. **`docs/EVOLUCAO_PESQUISA.md`** — diário da pesquisa (começar com uma entrada “Estado inicial”).  
2. **`docs/experimentos/`** — opcional; um MD por experiência maior.  
3. Ao terminar cada sessão de trabalho: **uma entrada curta** em `EVOLUCAO_PESQUISA.md` + cópia ou caminho dos JSON/PNG.

---

## 7. Comandos úteis para reprodução

**Download INPE (Ceará, vários anos):**
```bash
python -m src.inpe_queimadas_download --start 2024 --end 2026
```

**Avaliação GOES vs INPE (com twin e mapas):**
```bash
python -m src.unsupervised_fire_goes \
  --inpe-csv data/inpe_focos_ce/focos_ce_INPE_2024_2026.csv \
  --dates 2024-10-31 \
  --method all \
  --output-json data/goes16_eval/metrics_run.json \
  --map-dir data/goes16_eval/maps
```

**Métricas estritas (sem dilatação INPE, sem calibração de limiar):**
```bash
python -m src.unsupervised_fire_goes ... --truth-dilate 0
# (omitir --calibrate-contamination)
```

---

## 8. Boas práticas

- Preferir **datas em UTC** para alinhar com GOES e campos `data_pas` / `data_hora_gmt`.  
- Não apagar JSON antigos: renomear com sufixo `_run_v2`, `_notwin`, etc.  
- Quando publicar figuras externamente, indicar **licença dos dados** (INPE / NOAA Open Data).  
- Separar claramente no texto o que é **método não supervisionado** do que usa **calibração com INPE** (`--calibrate-contamination`) ou **tolerância espacial** (`--truth-dilate`).

---

*Última atualização do guia: criado para padronizar documentação de testes e evolução da pesquisa neste repositório.*
