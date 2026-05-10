# Instruções para criação e manutenção do README

Este guia define **como redigir e atualizar** o `README.md` do repositório, incluindo **resumo simples de como o modelo funciona** (§1.1), **referência a todos os experimentos**, **como gerar os mapas do Ceará** (PNG/HTML) e **como incluir imagens** no README, sem duplicar conteúdo pesado nem quebrar links no GitHub/GitLab.

---

## 1. Objetivo do README

- **Primeira leitura**: explicar o projeto, como instalar e rodar, e onde achar resultados.
- **Não substituir** artigos longos (`ARTICLE.md`, `ARTICLE_ACADEMIC.md`) nem o ranking detalhado de experimentos (`EXPERIMENTS.md`): o README deve **linkar** esses ficheiros e resumir só o essencial.
- **Imagens**: ilustrar fluxos ou resultados representativos; evitar dezenas de PNGs embutidos (peso de clone e renderização).

---

## 1.1 Instrução: secção «Como o modelo funciona» no `README.md`

Inclua sempre uma explicação **curta e em linguagem simples** (leitor sem formação em ML ou física de satélite).

| Regra | Detalhe |
|--------|---------|
| **Posição** | Logo **após** «Objetivo» (ou imediatamente antes de «Artigos»), título sugerido: `## Como o modelo funciona (em poucas palavras)`. |
| **Tamanho** | Cerca de **4 a 6** frases numeradas, ou até ~120 palavras, mais **uma** linha de limitação operacional. |
| **Conteúdo mínimo** | (1) focos → grelha no espaço e no tempo; (2) gêmeo digital = simulação de propagação entre células; (3) ML opcional = padrões nos dias anteriores + limiares; (4) ST-HyperNet / GOES não sup. opcionais = «normal» vs «fora do normal» num cubo simplificado a partir dos focos — **não** é produto operacional INPE nem ABI bruto. |
| **Estilo** | Frases curtas, voz activa, **sem** equações; evite siglas sem explicar (pode escrever «aprendizado de máquinas (ML)» na primeira vez). |
| **Profundidade** | Não duplicar o `ARTICLE.md`: termine com **um** link para quem quiser o detalhe técnico. |
| **Manutenção** | Ao alterar o fluxo principal (`main.py`, módulos centrais em `src/`), rever esta secção para continuar verdadeira. |

O `README.md` do repositório já contém um exemplo desta secção; mantenha-a alinhada ao código quando a lógica mudar.

---

## 2. Estrutura recomendada (alinhada ao README atual)

Sugestão de secções, nesta ordem lógica:

| Secção | Conteúdo |
|--------|-----------|
| Título + uma linha | Nome do projeto e frase de valor. |
| Objetivo | Bullet points (detecção, simulação, validação). |
| **Como o modelo funciona** | Resumo simples em 4–6 passos; ver **§1.1**. |
| Artigos / papers | Links para `ARTICLE.md`, `ARTICLE_ACADEMIC.md`. |
| Documentação técnica | Links: `EXPERIMENTS.md`, `EXPERIMENT_PROTOCOL.md`, `PYRO_CAATINGA.md`, **`GUIA_EXPERIMENTOS_E_PARAMETROS.md`**, `EXPERIMENTS.md` (ranking). |
| **Experimentos e resultados** (recomendado expandir) | Tabela ou lista curta: comando → pasta de saída → ficheiros-chave (`metrics_by_day.json`, `*_best_params.json`, figuras). Ver secção 4. |
| Dados abertos | Tabela fonte/acesso (já existente). |
| Estrutura do projeto | Árvore de pastas (atualizar quando surgirem novos módulos). |
| Como executar | Blocos `bash` com `main.py`, `run_experiments`, `compare_st_hypernet_days`, `compare_goes_unsupervised_days`, PYRO, dashboard, notebook. |
| Funcionalidades | Checklist `[x]` mantida alinhada ao código. |
| Artefatos gerados | Lista de caminhos sob `data/` (sincronizar com `GUIA_EXPERIMENTOS_E_PARAMETROS.md`). |
| **Figuras de exemplo** (opcional mas desejável) | Galeria curta com imagens (secção 5); **gerar mapas Ceará**: secção 4. |
| Dashboard | Instruções de uso no README; **capturas PNG** para o README: [docs/INSTRUCOES_CAPTURAS_DASHBOARD.md](docs/INSTRUCOES_CAPTURAS_DASHBOARD.md) e `scripts/capture_dashboard_screenshots.py`. |
| Próximos passos | Lista viva. |

---

## 3. Incluir “todos os experimentos” no README

O repositório tem **várias famílias** de experimento; o README deve mencionar **todas**, com link para o detalhe noutro ficheiro quando for longo.

### 3.1 Bateria ML (`run_experiments`)

- **Link obrigatório**: [EXPERIMENTS.md](EXPERIMENTS.md) (ranking gerado).
- **Texto curto no README**: uma frase do tipo “Dez+ configurações de `MLTwinConfig` comparadas; métricas em CSV/JSON.”
- **Caminhos**: `data/experiments/all_experiments_summary.csv`, `all_experiments_full.json`, `runs/<nome>.json`, `runs/<nome>_best_params.json`, `dataset_fires_detail.json`.
- **Não** colar tabelas enormes no README: use link ou 2–3 linhas de destaques (ex.: melhor F1 estrito) actualizadas à mão quando fizer sentido.

### 3.2 ST-HyperNet (treino + comparação por dia)

- Comandos: `python -m src.run_st_hypernet …`, `python -m src.compare_st_hypernet_days …`.
- Saídas típicas: `data/st_hypernet_compare/` ou pastas dedicadas (ex.: `data/st_hypernet_2024_all_fire_days/`) com `metrics_by_day.json`, `st_hypernet_best_model.pt`, `st_hypernet_best_params.json`, PNGs `st_hypernet_real_vs_pred_*.png`, `st_hypernet_ceara_map_*.png`/`.html`.
- Referência de política de artefatos: [GUIA_EXPERIMENTOS_E_PARAMETROS.md](GUIA_EXPERIMENTOS_E_PARAMETROS.md).

### 3.3 GOES não supervisionado (comparação por dia)

- Comando: `python -m src.compare_goes_unsupervised_days …`.
- Saídas: análogas ao ST (prefixo `goes_unsup_*`, mesmo tipo de métricas/figuras).

### 3.4 PYRO-Caatinga MVP

- Já listado em “Artefatos”; manter comandos em “Como executar”.

### 3.5 Pipeline principal (`main.py`)

- Mencionar `data/pipeline_result.json`, `data/ml_twin_validation.json`, `data/ml_twin_best_params.json`, `data/st_hypernet_last/` quando aplicável.

### Tabela modelo “Experimentos” no README

Exemplo (ajustar pastas reais após cada campanha):

| Experimento | Comando (resumo) | Saída principal |
|-------------|------------------|------------------|
| Bateria ML | `python -m src.run_experiments --dataset …` | `data/experiments/` |
| ST-HyperNet vs dias | `python -m src.compare_st_hypernet_days --csv … --out …` | `metrics_by_day.json`, PNGs, `*_best_params.json` |
| GOES unsup vs dias | `python -m src.compare_goes_unsupervised_days …` | idem, prefixo `goes_unsup` |
| PYRO MVP | `python main.py --pyro-mvp …` | `data/pyro_caatinga/` |

---

## 4. Como gerar os mapas do Ceará (PNG + HTML)

Os **mapas de todo o estado** (focos reais em vermelho, células previstas em verde, bbox do Ceará) são criados automaticamente pelos fluxos de **comparação por dia**. Não existe um comando separado só para mapas: eles são um produto da mesma execução que gera as grades `*_real_vs_pred_*.png` e as métricas.

### 4.1 Dependências

Na raiz do projeto (com o ambiente activo):

```bash
pip install -r requirements.txt
```

São necessários **matplotlib** (PNG) e **folium** (HTML interactivo). Se o HTML falhar, confirme `import folium` no Python.

### 4.2 ST-HyperNet (recomendado para mapas alinhados ao cubo ST)

Gera, **por cada dia avaliado**:

- `st_hypernet_ceara_map_<YYYY-MM-DD>.png` — vista geográfica (scatter + raster de score)
- `st_hypernet_ceara_map_<YYYY-MM-DD>.html` — mapa Folium (camadas reais / previsto)

Exemplo (Ceará, CSV de focos, ano 2024 completo, pasta de saída dedicada):

```bash
python -m src.compare_st_hypernet_days \
  --csv data/focos_CE_GOES16_2024.csv \
  --out data/st_hypernet_2024_all_fire_days \
  --year 2024 \
  --max-days 0 \
  --max-days-history 0 \
  --epochs 8 \
  --inference-stride 2 \
  --pred-threshold 0.2
```

Parâmetros úteis:

| Parâmetro | Efeito nos mapas |
|-----------|------------------|
| `--out` | Pasta onde caem **todos** os PNG/HTML desse run |
| `--year` | Restringe focos (e cubo) ao ano civil |
| `--max-days 0` | Um ficheiro de mapa **por cada dia com foco** dentro do cubo |
| `--pred-threshold` | Limiar **base**; o limiar **adaptativo por dia** (por defeito) pode subir para reduzir falsos positivos nas métricas e na legenda |
| `--no-adaptive-threshold` | Usa só o limiar fixo em toda a lógica binária |

O prefixo dos nomes de ficheiro é **`st_hypernet`** no CLI; para outro prefixo (ex.: relatório paralelo), chame em Python `compare_st_hypernet_fire_days_and_save_figures(..., file_prefix=\"meu_prefixo\")`.

Saída adicional na mesma pasta: `metrics_by_day.json`, `st_hypernet_best_model.pt`, `st_hypernet_best_params.json`, grades `st_hypernet_real_vs_pred_*.png`.

### 4.3 GOES não supervisionado (mapas com prefixo `goes_unsup`)

Mesma lógica de ficheiros, com **Isolation Forest** / cubo proxy:

```bash
python -m src.compare_goes_unsupervised_days \
  --csv data/focos_CE_GOES16_2024.csv \
  --out data/goes_unsup_2024_maps \
  --year 2024 \
  --max-days 0 \
  --max-days-history 0 \
  --pred-threshold 0.35
```

Ficheiros: `goes_unsup_ceara_map_<data>.png`, `goes_unsup_ceara_map_<data>.html` (e grades `goes_unsup_real_vs_pred_*.png`).

### 4.4 Onde está o código

- `src/compare_ceara_maps.py` — `save_ceara_map_png`, `save_ceara_folium_map`, máscara de visualização.
- `src/compare_st_hypernet_days.py` e `src/compare_goes_unsupervised_days.py` — orquestram o loop por dia e chamam os guardadores de figuras.

### 4.5 Verificar que os mapas foram gerados

```bash
ls data/st_hypernet_2024_all_fire_days/st_hypernet_ceara_map_*.png | head
ls data/st_hypernet_2024_all_fire_days/st_hypernet_ceara_map_*.html | head
```

Abrir o `.html` no navegador (duplo clique ou `open` no macOS).

### 4.6 Incluir essas imagens no README

Depois de geradas, use caminhos relativos na raiz do repo (ver secção 5). Se `data/` não for versionado no Git, copie os PNG escolhidos para `docs/readme_figures/` antes de referenciar no README (secção 8).

---

## 5. Incluir imagens no README

### 5.1 Sintaxe Markdown (GitHub)

Imagens com **caminho relativo à raiz do repositório**:

```markdown
![Legenda curta](data/st_hypernet_2024_all_fire_days/st_hypernet_ceara_map_2024-08-09.png)
```

Requisitos:

- O ficheiro PNG/JPG **tem de existir no repo** para aparecer no GitHub (não use só caminhos locais que estão no `.gitignore`, a menos que documente “gerar localmente”).
- **Legenda**: data + técnica (ex.: “ST-HyperNet — mapa Ceará — 2024-08-09”).

### 5.2 Boas práticas

1. **Poucas imagens no README** (3–6): escolher dias representativos (alto IoU, caso difícil, saturação corrigida pelo limiar adaptativo).
2. **Muitas figuras**: não commitar centenas de PNGs só para o README; opções:
   - manter galeria numa pasta `docs/figures/` com **cópias** de 6–12 imagens escolhidas a dedo, **versionadas**; ou
   - linkar para uma release/Zenodo com o pacote `data/…` comprimido.
3. **Tamanho**: preferir PNG ≤ ~500 KB cada (reduzir DPI ou recortar se necessário).
4. **Consistência**: mesma convenção de nomes que o pipeline (`st_hypernet_ceara_map_DATE.png`).

### 5.3 Exemplo de secção “Figuras de exemplo”

```markdown
## Figuras de exemplo (ST-HyperNet, 2024)

Grade 3 painéis e mapa do estado (run `compare_st_hypernet_days`, pasta `data/st_hypernet_2024_all_fire_days/`):

| Dia | Grade real vs previsto | Mapa Ceará |
|-----|-------------------------|------------|
| 2024-08-09 | ![…](data/.../st_hypernet_real_vs_pred_2024-08-09.png) | ![…](data/.../st_hypernet_ceara_map_2024-08-09.png) |
```

*(Substituir `…` pelo caminho real; em tabelas grandes considere só uma coluna de imagem por linha.)*

### 5.4 O que não fazer

- Não usar `file://` nem caminhos absolutos da máquina.
- Não embutir imagens binárias em base64 no Markdown.
- Não prometer no README imagens que **não** estão no Git por política de tamanho — nesse caso, escrever “executar o comando X gera os PNGs em `data/…`”.

---

## 6. Checklist ao atualizar o README

- [ ] A secção **«Como o modelo funciona»** continua simples, correcta e com link para `ARTICLE.md`.
- [ ] Comandos na secção “Como executar” batem com `main.py` / módulos `src.*` actuais.
- [ ] Todas as **famílias de experimento** aparecem (ML batch, ST compare, GOES compare, PYRO, pipeline).
- [ ] Links para `EXPERIMENTS.md`, `EXPERIMENT_PROTOCOL.md`, `GUIA_EXPERIMENTOS_E_PARAMETROS.md` estão correctos.
- [ ] Lista “Artefatos gerados” inclui `*_best_params.json`, `ml_twin_best_params.json`, `st_hypernet_best_params.json` quando forem parte do fluxo oficial.
- [ ] Imagens: caminhos relativos válidos; ficheiros versionados ou nota explícita de geração local.
- [ ] Estrutura de pastas (`src/`, `data/`) reflecte o estado actual do repo.

---

## 7. Relação com outros documentos

| Documento | Papel |
|-----------|--------|
| `README.md` | Porta de entrada + comandos + visão geral + poucas figuras. |
| `EXPERIMENTS.md` | Ranking automático das técnicas ML (não editar manualmente o ranking). |
| `EXPERIMENT_PROTOCOL.md` | Procedimento gerado pelo `run_experiments`. |
| `GUIA_EXPERIMENTOS_E_PARAMETROS.md` | Política de guardar experimentos e melhores parâmetros. |
| **Este ficheiro** | Como montar e manter o README em coerência com o acima. |

---

## 8. Opcional: figuras sob `docs/`

Se quiser imagens **estáveis** independentemente de apagar/regenerar `data/st_hypernet_*`:

1. Criar `docs/readme_figures/`.
2. Copiar os PNG escolhidos para lá.
3. No README: `![…](docs/readme_figures/st_2024-08-09_map.png)`.
4. Documentar no commit que essas cópias são “snapshot para documentação”.

Isto mantém o README legível mesmo quando pastas grandes em `data/` não vão para o Git.
