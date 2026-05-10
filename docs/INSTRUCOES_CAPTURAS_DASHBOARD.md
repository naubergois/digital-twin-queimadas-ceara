# Instruções — capturas de ecrã do dashboard (README)

Este documento descreve **como obter e manter** as imagens do painel Streamlit usadas no [README.md](../README.md) na raiz do repositório.

## Mapas do Ceará com achados (PNG da pipeline, para o README)

Além das **capturas de ecrã** do Streamlit, o README mostra **dois PNG** gerados pela comparação ST-HyperNet (focos reais vs previsto na grade):

| Ficheiro em `docs/screenshots/` | Origem típica |
|----------------------------------|----------------|
| `mapa-ceara-reais-vs-previsto.png` | `data/.../st_hypernet_ceara_map_<DATA>.png` |
| `grade-real-vs-previsto.png` | `data/.../st_hypernet_real_vs_pred_<DATA>.png` |

A pasta `data/` está no `.gitignore`; por isso estes ficheiros **copiam-se** para `docs/screenshots/` antes do commit:

```bash
python scripts/copy_sample_st_map_pngs_to_docs.py --date 2024-08-09
# ou outra pasta: --metrics-dir data/st_hypernet_compare
```

O script grava nomes estáveis usados pelo [README.md](../README.md).

---

## Figuras de experimentos ML e comparação ST (README)

Gera **barras horizontais** de ranking (`run_experiments`) e um **resumo agregado** ST-HyperNet a partir de ficheiros em **`docs/fixtures/`** (por defeito), sem depender de `data/` no git:

```bash
python scripts/generate_readme_experiment_figures.py
```

Saídas em `docs/screenshots/`:

| PNG | Conteúdo |
|-----|----------|
| `experimentos-ranking-f1-ml.png` | F1 estrito (ML) por experimento |
| `experimentos-ranking-score-operacional.png` | Score operacional por experimento |
| `comparacao-st-hypernet-agregado.png` | IoU / precisão / recall (médias + micro) |

Para reflectir uma corrida local recente, passe os paths absolutos ou relativos aos CSV/JSON gerados em `data/`.

---

## O que capturar (dashboard Streamlit)

O dashboard (`streamlit run dashboard/app.py`) tem **duas abas** principais:

| Ficheiro sugerido | Conteúdo a mostrar no ecrã |
|-------------------|-----------------------------|
| `docs/screenshots/dashboard-aba-st-hypernet.png` | Aba **«ST-HyperNet (modelo)»**: métricas agregadas, gráfico precisão/IoU, dia seleccionado, mapa reconstruído (se existir `map_real_latlon` no JSON) ou PNGs de comparação. |
| `docs/screenshots/dashboard-aba-focos.png` | Aba **«Focos no mapa»**: mapa Folium com cluster e, se possível, fonte local ou período com dados. |

Opcional (terceira imagem para documentação):

| Ficheiro | Conteúdo |
|----------|----------|
| `docs/screenshots/dashboard-sidebar.png` | Sidebar com pasta de métricas ST-HyperNet e controlos de focos (pode ser crop da mesma janela). |

## Passo a passo manual

1. **Dados para a aba ST-HyperNet**  
   Garanta que existe `data/st_hypernet_2024_all_fire_days/metrics_by_day.json` (ou outra pasta listada na sidebar) **com** os campos `map_real_latlon` / `map_pred_latlon_score` se quiser o mapa reconstruído na captura. Caso contrário, a captura mostrará só gráficos e PNGs estáticos.

2. **Arranque local** (na raiz do repo):
   ```bash
   pip install -r requirements.txt
   streamlit run dashboard/app.py
   ```
   Abra `http://localhost:8501` no browser.

3. **Tamanho da janela**  
   Use largura **≥ 1280 px** (ideal 1400–1600 px) para o layout «wide» do Streamlit não ficar espalmado de mais.

4. **Captura**  
   - macOS: `Cmd + Shift + 4` e seleccione a janela, ou ferramenta de captura integrada.  
   - Windows: `Win + Shift + S`.  
   - Linux: ferramenta do ambiente (GNOME «Capturar ecrã», etc.).

5. **Gravar**  
   Guarde os PNG em `docs/screenshots/` com os nomes da tabela acima (substitua ficheiros antigos se actualizar a UI).

6. **README**  
   O [README.md](../README.md) referencia estes caminhos relativos. Depois de adicionar ou trocar imagens, faça `git add docs/screenshots/*.png` e commit.

## Captura automática (opcional)

O Streamlit importa o mesmo código que `main.py`; o interpretador usado no script **tem de ter** as dependências do projecto (`tqdm`, `folium`, etc.). Recomenda-se um virtualenv:

```bash
python -m venv .venv-capture
. .venv-capture/bin/activate   # Windows: .venv-capture\Scripts\activate
pip install -r requirements.txt
pip install playwright
python -m playwright install chromium
python scripts/capture_dashboard_screenshots.py
```

O script arranca o Streamlit numa porta livre, espera o carregamento e grava `docs/screenshots/dashboard-aba-st-hypernet.png` e `dashboard-aba-focos.png` (muda para a segunda aba via `[role="tab"]`).

Requer **rede** se a app fizer pedidos (INPE na aba de focos). Para captura estável, deixe a fonte **«Dados Locais (CSV)»** e um período com CSV em `data/` antes de correr o script, ou aceite avisos vazios nessa aba.

Depois (opcional), copie os PNGs de mapa para o README:

```bash
python scripts/copy_sample_st_map_pngs_to_docs.py --date 2024-08-09
```

## Checklist antes de publicar

- [ ] Imagens nítidas (sem zoom browser estranho).
- [ ] Sem chaves API, tokens ou dados pessoais visíveis.
- [ ] Nomes de ficheiro exactamente como no README (`dashboard-aba-st-hypernet.png`, `dashboard-aba-focos.png`).
- [ ] **Mapas com achados:** `mapa-ceara-reais-vs-previsto.png` e `grade-real-vs-previsto.png` actualizados (`copy_sample_st_map_pngs_to_docs.py`) se mudou o dia de exemplo.
- [ ] **Experimentos / ST:** `generate_readme_experiment_figures.py` — actualizar `docs/fixtures/` se quiser mudar o exemplo versionado, depois voltar a gerar os três PNGs.
- [ ] Tamanho razoável (se > 1.5 MB cada, comprima com `pngquant` ou similar).

## Manutenção

Quando alterar `dashboard/app.py` (layout, abas, textos), **actualize as capturas** para o README não ficar desactualizado visualmente.
