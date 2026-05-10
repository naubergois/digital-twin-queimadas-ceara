# Plano de implementação — PYRO-Caatinga

> Plano executável de implementação da técnica **PYRO-Caatinga**
> (climatology-residual transformer + destilação cruzada VIIRS→GOES + twin loop)
> dentro do repositório do gêmeo digital de queimadas do Ceará.
>
> Referência da proposta: [RESEARCH_GOES16.md](RESEARCH_GOES16.md)

---

## 0. Visão geral

| | |
|---|---|
| **Duração total** | 12 semanas (3 marcos de 4 semanas) |
| **Marco 1 (sem 1–4)** | MVP offline: front-end climatológico + modelo aluno treinado com VIIRS |
| **Marco 2 (sem 5–8)** | Streaming a 5 min + integração no dashboard + cabeças físicas |
| **Marco 3 (sem 9–12)** | Twin feedback loop + self-distillation online + validação no Ceará |
| **Critério de sucesso** | F1 ≥ 0.78 vs BDQueimadas no Ceará; latência ≤ 5 min; FP em meio-dia ↓ ≥ 60% vs FDC |

---

## 1. Estado atual do repositório (baseline)

```
src/
├── analysis.py        # análise estatística de focos
├── digital_twin.py    # núcleo do gêmeo digital
├── fire_data.py       # cliente FIRMS / BDQueimadas
└── satellite.py       # NASA GIBS streaming
dashboard/app.py       # Streamlit
config/ceara_config.py # AOI, paleta, parâmetros
notebooks/proposta_funcional.ipynb
```

**Já existe:** ingestão FIRMS, GIBS streaming, dashboard com auto-refresh, módulo
de gêmeo digital com lógica de spread (CA simples).

**Falta:** ingestão GOES-16 ABI L1b/L2, climatologia BT, modelo aluno, destilação,
loop de feedback com simulador.

---

## 2. Estrutura de diretórios alvo (após implementação)

```
src/
├── analysis.py
├── digital_twin.py
├── fire_data.py
├── satellite.py
├── ingestion/
│   ├── __init__.py
│   ├── goes_abi.py            # download/parse GOES-16/19 ABI L1b RadC
│   ├── viirs_firms.py         # já existe parcialmente em fire_data
│   └── era5.py                # vento + umidade do ECMWF para o twin
├── preprocess/
│   ├── __init__.py
│   ├── reproject.py           # GOES → grid local Ceará 2 km
│   ├── climatology.py         # μ̂(p, doy, hod) EWMA
│   └── viirs_softlabel.py     # reprojeção VNP14IMG → grid GOES com kernel
├── models/
│   ├── __init__.py
│   ├── pyro_caatinga.py       # backbone + cabeças
│   ├── blocks.py              # Mamba/Swin causal blocks
│   └── physics_heads.py       # FRP Stefan-Boltzmann + máscara + incerteza
├── training/
│   ├── __init__.py
│   ├── pretrain_mae.py        # masked autoencoding em BT̃
│   ├── distill_viirs.py       # KL teacher-student
│   ├── losses.py              # BCE+Dice, MAE FRP, KL
│   └── datamodule.py          # PyTorch Lightning DataModule
├── inference/
│   ├── __init__.py
│   ├── streaming.py           # janela deslizante 5 min
│   └── twin_loop.py           # auto-supervisão online
└── eval/
    ├── __init__.py
    ├── metrics.py             # F1, IoU, latência, FP rate
    └── compare_fdc.py         # benchmark vs produto oficial
config/
├── ceara_config.py
└── pyro_caatinga.yaml         # hiperparâmetros + caminhos
data/                          # gitignored
├── raw/goes16/
├── raw/viirs/
├── raw/era5/
├── processed/climatology.nc
├── processed/tiles/
└── checkpoints/
notebooks/
├── proposta_funcional.ipynb
├── 01_explore_goes.ipynb
├── 02_climatology.ipynb
├── 03_train_baseline.ipynb
└── 04_eval_ceara.ipynb
dashboard/app.py               # nova camada PYRO + barra de incerteza
tests/
├── test_climatology.py
├── test_distillation.py
├── test_twin_loop.py
└── fixtures/
```

---

## 3. Fases e tarefas

### Fase 0 — Setup (½ semana, semana 1)

**Objetivo:** preparar ambiente, dependências, AOI, credenciais.

| # | Tarefa | Saída | Estimativa |
|---|---|---|---|
| 0.1 | Atualizar `requirements.txt` com PyTorch, lightning, mamba-ssm/timm, xarray, satpy, s3fs, dvc | `requirements.txt` | 0.5 d |
| 0.2 | Criar `data/` no `.gitignore` e documentar estrutura | `.gitignore`, `data/README.md` | 0.5 d |
| 0.3 | Configurar acesso S3 público da NOAA (bucket `noaa-goes16`) | `config/pyro_caatinga.yaml` | 0.5 d |
| 0.4 | AOI Ceará em GeoJSON + grid 2 km projetado em GOES-ABI | `config/aoi_ceara_grid.geojson` | 1 d |

**Dependências novas:**
```
torch>=2.3.0
lightning>=2.2.0
satpy>=0.50.0
s3fs>=2024.0.0
mamba-ssm>=2.2.0   # opcional; alternativa: timm SwinT
einops>=0.7.0
zarr>=2.18.0
netCDF4>=1.6.0
dvc[s3]>=3.0.0     # versionar dados grandes
torchmetrics>=1.3.0
hydra-core>=1.3.0
```

---

### Fase A — Ingestão e climatologia (semana 1–2) → **Marco 1**

**Objetivo:** baixar 90 dias de GOES-16 ABI sobre Ceará e construir o cubo climatológico μ̂.

| # | Arquivo | Função | Critério de aceite |
|---|---|---|---|
| A.1 | `src/ingestion/goes_abi.py` | `download_abi(start, end, bands=[2,7,14], product='ABI-L1b-RadF', aoi)` | baixa 90 dias (~5 min cadência) e converte para NetCDF local |
| A.2 | `src/preprocess/reproject.py` | `reproject_to_local_grid(da, target_grid)` | reprojeta GOES → EPSG:31984 grid 2 km, validado no notebook 01 |
| A.3 | `src/preprocess/climatology.py` | `compute_ewma_climatology(cube, lambda_=0.05, robust='trimmed_median')` | gera `processed/climatology.nc` com dims (pixel, doy=366, hod=24, band) |
| A.4 | `notebooks/01_explore_goes.ipynb` | sanity check visual de B7/B14/ΔBT em dia conhecido com fogo BDQueimadas | gráfico antes/depois resíduo |
| A.5 | `tests/test_climatology.py` | testes unitários do EWMA, robustez à NaN, exclusão de pixels históricos com fogo | `pytest -q` verde |

**Detalhes técnicos:**
- Climatologia robusta: usar mediana truncada (10–90 percentil) por bin (doy, hod).
- Excluir pixels presentes em BDQueimadas histórico no cálculo de μ̂ (evita "absorver" fogos recorrentes).
- Atualização online: EWMA λ=0.05 quando o sistema entra em produção.
- Salvar como Zarr para acesso parcial rápido.

---

### Fase B — Pré-treino auto-supervisionado (semana 3) → **Marco 1**

**Objetivo:** modelo aluno aprende representações de BT̃ sem rótulos via masked autoencoding.

| # | Arquivo | Função | Critério de aceite |
|---|---|---|---|
| B.1 | `src/models/blocks.py` | `CausalMambaBlock`, `SwinT3DBlock` | testes de shape em batch fictício |
| B.2 | `src/models/pyro_caatinga.py` | `PyroCaatingaBackbone(in_ch=4, T=12, hidden=128, blocks=4)` | ~5M params; forward em CPU < 200ms para tile 64×64×12 |
| B.3 | `src/training/pretrain_mae.py` | mascarar 50% dos patches espaciotemporais; loss = MSE em BT̃ | curva de loss decresce em 30k steps |
| B.4 | `src/training/datamodule.py` | tiles 64×64 com janela 12 frames (1h), augmentação flip/rot | DataLoader retorna 8 amostras/s em laptop |
| B.5 | `notebooks/02_climatology.ipynb` + `03_train_baseline.ipynb` | visualização de embeddings (UMAP) de pixels com/sem fogo | clusters separáveis qualitativamente |

**Notas:**
- Backbone leve para inferência em CPU/MPS no laptop do projeto.
- Pré-treino em GPU Colab se necessário; checkpoint salvo em `data/checkpoints/mae_pretrain.ckpt`.

---

### Fase C — Destilação VIIRS→GOES (semana 4) → **Marco 1**

**Objetivo:** treinar cabeças supervised com VIIRS (375 m, NASA FIRMS VNP14IMG) como teacher.

| # | Arquivo | Função | Critério de aceite |
|---|---|---|---|
| C.1 | `src/preprocess/viirs_softlabel.py` | `viirs_to_goes_grid(viirs_pts, kernel='gauss', sigma_km=1.0)` | mapa contínuo [0,1] no grid GOES |
| C.2 | `src/models/physics_heads.py` | `MaskHead`, `FRPHead` (Stefan-Boltzmann), `UncertaintyHead` (MC-dropout) | shapes corretos; FRP > 0 |
| C.3 | `src/training/losses.py` | `combined_loss = α·BCE_hard(BDQ) + (1-α)·KL(student ∥ viirs_soft) + β·DiceLoss + γ·MAE_FRP` | unitário por componente |
| C.4 | `src/training/distill_viirs.py` | trainer Lightning; α agendado de 0.2→0.7 | F1 ≥ 0.65 no holdout em 5 epochs |
| C.5 | `tests/test_distillation.py` | KL não-NaN, gradiente flui, máscara de nuvem aplicada | verde |

**Notas:**
- VIIRS-overpass apenas ~01h e ~13h locais; nas horas sem overpass usa-se só BCE_hard contra BDQueimadas.
- Nuvem mascarada via B2 (0.64 µm) e B4 (1.378 µm cirrus) com threshold simples; pixels mascarados não contribuem para KL.
- Validação cruzada espacial (folds por mesorregião do Ceará) para evitar leakage.

---

### Marco 1 — checkpoint (fim da semana 4)

**Entrega:** modelo treinado offline, F1 ≥ 0.65 no holdout VIIRS, notebook `03_train_baseline.ipynb` reproduz o resultado.

**Demo:** notebook que carrega 1 dia de GOES sobre Ceará, aplica PYRO-Caatinga e plota máscara + FRP + incerteza vs BDQueimadas.

---

### Fase D — Streaming e cabeças físicas (semana 5–6) → **Marco 2**

**Objetivo:** rodar inferência incremental a cada 5 min e expor ao dashboard.

| # | Arquivo | Função | Critério de aceite |
|---|---|---|---|
| D.1 | `src/inference/streaming.py` | `RingBuffer(T=12)` + `predict_step(frame_t)` | latência < 1s por frame em laptop |
| D.2 | `src/inference/streaming.py` | listener S3 NOAA via `s3fs` watch | recebe novo arquivo em < 30s após publicação |
| D.3 | `src/models/physics_heads.py` | `FRPHead` calibrado: σ(B7)·A·T⁴ × correção atmosférica | erro < 30% vs FDC FRP em casos rotulados |
| D.4 | `src/inference/streaming.py` | top-K MC-dropout só nos pixels com escore > 0.3 | latência mantida; K=200 |

---

### Fase E — Integração no dashboard (semana 7) → **Marco 2**

**Objetivo:** sobrepor camada PYRO ao mapa GIBS já existente.

| # | Arquivo | Função | Critério de aceite |
|---|---|---|---|
| E.1 | `dashboard/app.py` | nova aba "PYRO-Caatinga" com camada de máscara + heatmap de incerteza | render em < 3s |
| E.2 | `dashboard/app.py` | toggle "Comparar com FDC oficial" com overlay lado a lado | diff visualmente claro |
| E.3 | `dashboard/app.py` | gráfico de barras: contagens PYRO vs FDC vs BDQueimadas última 1h | atualiza com auto-refresh existente |
| E.4 | `dashboard/app.py` | alertas para pixels com escore > τ_high e baixa incerteza | toast/notif Streamlit |

---

### Fase F — Avaliação Marco 2 (semana 8)

| # | Arquivo | Função | Critério de aceite |
|---|---|---|---|
| F.1 | `src/eval/metrics.py` | F1, IoU, latência média, FP rate por hora-do-dia | testes verdes |
| F.2 | `src/eval/compare_fdc.py` | benchmark PYRO vs FDC vs FTA num período de 30 dias | tabela markdown + gráficos em `notebooks/04_eval_ceara.ipynb` |
| F.3 | Documentar resultados em `RESULTS_M2.md` | — | revisado |

**Critério de avanço:** F1 ≥ 0.72 no Ceará e latência ≤ 5 min em produção.

---

### Fase G — Twin feedback loop (semana 9–10) → **Marco 3**

**Objetivo:** fechar o laço com o simulador CA já existente em `src/digital_twin.py`.

| # | Arquivo | Função | Critério de aceite |
|---|---|---|---|
| G.1 | `src/ingestion/era5.py` | download horário de vento u/v e umidade ERA5 sobre Ceará | NetCDF local atualizado horário |
| G.2 | `src/digital_twin.py` | aceitar `mask_t` PYRO + ERA5 e gerar `mask_t+5` por CA + vento | unit test com cenário sintético |
| G.3 | `src/inference/twin_loop.py` | comparar `ŷ_simulated_t+5` com `goes_real_t+5`; calcular erro pixel-wise | log JSON com erro por frame |
| G.4 | `src/inference/twin_loop.py` | self-distillation: se var(MC-dropout) < τ_low e erro alto, criar pseudo-rótulo | ckpt incremental salvo |
| G.5 | `tests/test_twin_loop.py` | invariantes: erro decresce ao longo de N frames sintéticos | verde |

**Notas:**
- Self-distillation conservador: só atualiza pesos das cabeças, mantém backbone congelado.
- Histerese para evitar feedback loop instável (mínimo 20 frames entre atualizações).
- Salvar todas as pseudo-rotulagens para auditoria offline.

---

### Fase H — Validação Ceará 90 dias (semana 11) → **Marco 3**

| # | Tarefa | Saída |
|---|---|---|
| H.1 | Selecionar período seco 2025 (set–nov) com fogos confirmados BDQueimadas | lista de eventos |
| H.2 | Rodar PYRO + FDC + FTA + GRU baseline no mesmo período | logs e máscaras |
| H.3 | Calcular métricas finais (Tabela 4 abaixo) e gerar `RESULTS_FINAL.md` | doc revisado |
| H.4 | Análise de erro: por mesorregião, por hora-do-dia, por tamanho de fogo | gráficos |

### Fase I — Hardening e documentação (semana 12) → **Marco 3**

| # | Tarefa | Saída |
|---|---|---|
| I.1 | Atualizar `README.md` e `ARTICLE.md` com PYRO-Caatinga | docs |
| I.2 | CI: `pytest`, lint, type-check com `mypy --strict` em `src/models` e `src/training` | GitHub Actions |
| I.3 | Dockerfile + `docker-compose` com serviço de streaming + dashboard | `docker compose up` sobe tudo |
| I.4 | Guia de operação `OPERATIONS.md`: como reagir a alertas, retreinar, monitorar drift | doc |

---

## 4. Cronograma resumido (Gantt textual)

```
Semana:        1  2  3  4  5  6  7  8  9  10 11 12
Fase 0 setup   ██
Fase A clim    ██ ██
Fase B mae        █
Fase C distill       ██
                                     ┃ Marco 1
Fase D stream             ██ ██
Fase E dash                     ██
Fase F eval2                       ██
                                     ┃ Marco 2
Fase G twin                           ██ ██
Fase H valid                                ██
Fase I hard                                    ██
                                     ┃ Marco 3
```

---

## 5. Métricas de aceitação por marco

### Marco 1 (sem 4) — MVP offline

| Métrica | Alvo |
|---|---|
| F1 holdout VIIRS (Ceará) | ≥ 0.65 |
| Recall fogos < 4 ha | ≥ 0.40 |
| Tempo de treino end-to-end | ≤ 24h em GPU única |
| Reprodutível via notebook | sim |

### Marco 2 (sem 8) — Streaming + dashboard

| Métrica | Alvo |
|---|---|
| Latência por frame | ≤ 1s CPU / ≤ 200ms GPU |
| Latência total detecção | ≤ 5 min após publicação NOAA |
| F1 vs BDQueimadas (30 dias) | ≥ 0.72 |
| FP em meio-dia (12–14h local) | ↓ ≥ 40% vs FDC |
| Camada PYRO no dashboard | sim |

### Marco 3 (sem 12) — Twin loop + validação final

| Métrica | Alvo |
|---|---|
| F1 vs BDQueimadas (90 dias seca) | ≥ 0.78 |
| FP em meio-dia | ↓ ≥ 60% vs FDC |
| Recall fogos < 4 ha | ↑ ≥ 25% vs FDC |
| Latência | ≤ 5 min |
| Erro FRP médio | ≤ 30% vs FDC FRP |
| Estabilidade self-distill (drift) | ΔF1 ≤ 2pp em 30 dias |

---

## 6. Riscos e mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|---|---|---|---|
| GOES-16 reposicionado/aposentado | baixa (já é GOES-19 East) | médio | trocar bucket S3 para `noaa-goes19`; mesmas bandas |
| VIIRS overpass nublado | alta | médio | fallback para BCE_hard; mascarar pixels nublados na KL |
| Climatologia "absorve" fogo recorrente | média | alto | mediana truncada + exclusão pixels BDQ histórico |
| Self-distillation instável | média | alto | histerese 20 frames; congelar backbone; auditoria de pseudo-rótulos |
| Latência S3 NOAA > 1 min | baixa | médio | listener event-driven + cache local |
| MC-dropout caro | média | baixo | top-K com K=200 |
| Drift sazonal (clima muda) | alta | médio | re-baseline μ̂ a cada 30 dias com EWMA |
| Falta de GPU local | média | baixo | usar Colab/Kaggle para treino; inferência em CPU |
| Rótulos BDQueimadas atrasam | média | médio | usar VIIRS como ground truth principal; BDQ como complemento |

---

## 7. Dependências externas e custos

| Recurso | Provedor | Custo |
|---|---|---|
| GOES-16/19 ABI L1b | NOAA Open Data S3 (`noaa-goes16`) | grátis |
| VIIRS VNP14IMG | NASA FIRMS API | grátis (chave já no `.env`) |
| ERA5 vento/umidade | Copernicus CDS | grátis (registrar) |
| BDQueimadas | INPE (terrabrasilis) | grátis |
| GPU para treino | Colab/Kaggle gratuito ou A10 spot | ~US$10–30 total |
| Storage local | ~50 GB para 90 dias GOES | local/disk |

---

## 8. Ordem de execução recomendada (sprint-a-sprint)

### Sprint 1 (sem 1–2): "ver o dado"
- Fase 0 + A: ingestão GOES + climatologia + notebook 01
- **Demo:** plot de B7/B14/BT̃ sobre Ceará num dia de fogo confirmado

### Sprint 2 (sem 3–4): "o modelo aprende"
- Fase B + C: pré-treino MAE + destilação VIIRS
- **Demo:** notebook 03 com F1 ≥ 0.65 no holdout

### Sprint 3 (sem 5–6): "tempo real"
- Fase D: streaming + cabeças físicas
- **Demo:** terminal mostrando inferência a cada 5 min com novo arquivo S3

### Sprint 4 (sem 7–8): "operável"
- Fase E + F: dashboard + benchmark
- **Demo:** dashboard com camada PYRO + tabela comparativa vs FDC

### Sprint 5 (sem 9–10): "fecha o laço"
- Fase G: twin feedback + self-distillation
- **Demo:** logs mostrando pseudo-rotulagens auditáveis e melhoria contínua

### Sprint 6 (sem 11–12): "valida e empacota"
- Fase H + I: validação 90 dias + docs + Docker
- **Entrega final:** `RESULTS_FINAL.md` + `docker compose up` funcional

---

## 9. Definition of Done (DoD)

Cada Pull Request deve atender:

- [ ] Código tipado (`mypy --strict` passa em arquivos novos)
- [ ] Testes unitários cobrindo o caminho feliz e ≥ 1 caso de borda
- [ ] Documentação inline mínima (docstring de função pública)
- [ ] Notebook de demo atualizado se mudou comportamento visível
- [ ] Sem credenciais/dados em commits (verificado por pre-commit hook)
- [ ] Métricas do marco vigente não regridem (CI roda benchmark leve)
- [ ] Log estruturado JSON para todas as etapas de inferência

---

## 10. Próximas ações imediatas

1. Atualizar `requirements.txt` com novas dependências (Fase 0.1).
2. Criar branch `feat/pyro-caatinga` a partir de `main`.
3. Abrir issue-tracker mapeando Fases A→I como milestones.
4. Iniciar Fase A: script de download GOES-16 ABI para 90 dias sobre Ceará.

---

## 11. Referências cruzadas

- Estado da arte e proposta detalhada: [RESEARCH_GOES16.md](RESEARCH_GOES16.md)
- Artigo do projeto: [ARTICLE.md](ARTICLE.md)
- Configuração regional: [config/ceara_config.py](config/ceara_config.py)
- Núcleo do gêmeo digital: [src/digital_twin.py](src/digital_twin.py)
- Streaming de satélite atual: [src/satellite.py](src/satellite.py)
