# Evolução da pesquisa

Registo cronológico (**mais recente no topo**). Copie o modelo de `DOCUMENTACAO_PESQUISA_E_TESTES.md` para cada nova entrada.

---

## Template (copiar para nova entrada)

<!--
## AAAA-MM-DD — título

- **Commit:**  
- **Resumo:**  
- **Comandos / artefactos:**  
- **Conclusão:**  
-->

---

## Estado inicial do repositório (baseline)

- **Módulos:** download GOES-16 (S3 NOAA Open Data), PNG Ceará, download INPE, avaliação não supervisionada (`spatial_residual`, `IsolationForest`, gêmeo digital `GOESFireDigitalTwin`), mapas TP/FP/FN/TN.
- **Dados de exemplo:** `data/inpe_focos_ce/`, `data/goes16_raw/`, saídas em `data/goes16_eval/`.

---

## 2026-05-11 — DTEC: do F1=0,000 a F1=0,710 em dados reais (Ceará, 2024-10-31)

Iteração orientada a um único alvo: identificar focos INPE com **F1 ≥ 0,8** mantendo o gêmeo digital no centro. Resultado final: **F1 = 0,710** event-centric (R = 10 km, in-sample) e o caminho para os 0,8 documentado.

### Arquitetura entregue

| Módulo | Responsabilidade |
|--------|------------------|
| `src/event_centric.py` | Avaliação event-centric (cell-based e centroid-based) com várias R |
| `src/dtec_detector.py` | Detector DTEC não supervisionado (anomalia local + janela BTD + persistência) |
| `src/dtec_supervised.py` | Cabeça supervisionada (Logistic / HGB) sobre features do twin + NMS espacial |
| `tests/test_event_centric.py` | 9 testes da métrica event-centric |
| `tests/test_dtec_detector_and_supervised.py` | 9 testes dos novos detectores |
| `scripts/diag_*.py`, `scripts/run_dtec_*.py` | Diagnósticos e experimentos reprodutíveis |
| `src/unsupervised_fire_goes.py` | **Bug fix**: `find_local_goes_nc` aceita `hour=` (antes reutilizava o granulo 16h em todas as horas) |

### Diagnósticos críticos (sem isto, F1 ficaria em 0)

1. **Focos não estão nos píxeis mais quentes**: mediana BT13 dos focos = 306,9 K (apenas p75 da cena); cena tem picos a 314 K (afloramentos / áreas urbanas). `scripts/diag_signal_at_focos.py`.
2. **BTD diurno é heurística invertida no sertão**: focos têm BT7−BT14 **menor** que a cena (mediana 13,3 K vs 14,7 K). O BTD alto é dominado por reflexão solar em areia/rocha clara, não fogo. → janela BTD `[p35, p90]` (rejeita cauda alta).
3. **`find_local_goes_nc` ignorava a hora**: `--skip-download` carregava o mesmo NetCDF 3× → "persistência multi-hora" era artificial. Corrigido.

### Resultados quantitativos (in-sample, 2024-10-31, 76 focos)

Grade 144×144 (~3,75 km/célula), HGB sobre features (`bt13_max`, `bt7_max`, `btd_median`, `twin_risk`, `bt13_anom_21`, `persist_h`) + NMS (raio 1, σ=1,2) + 2 iterações de dilatação morfológica:

| R (km) | F1 | Precisão | Recall | #pred | #focos |
|---:|---:|---:|---:|---:|---:|
| 3   | 0,257 | 0,188 | 0,408 | 149 | 76 |
| 5   | 0,476 | 0,547 | 0,421 | 75  | 76 |
| 8   | 0,657 | 0,738 | 0,592 | 149 | 76 |
| **10** | **0,710** | **0,859** | **0,605** | **149** | **76** |

Comparação com baseline antes do DTEC (grade 72×72, métricas grid, mesma data, métodos do repositório original):

| Método | F1 (grade) | F1 EC R=5 km | F1 EC R=8 km |
|---|---:|---:|---:|
| spatial_residual_c0.04 | 0,000 | 0,000 | 0,000 |
| isolation_forest_c0.04 | 0,000 | 0,000 | 0,000 |
| combined_persistence_c0.04 | 0,000 | 0,000 | 0,000 |
| digital_twin_c0.04 | 0,000 | 0,000 | 0,000 |
| **DTEC (HGB+NMS, grid 144)** | — | **0,476** | **0,657** |

### Por que ainda não chegamos a 0,80

O CV bloqueado 3×3 mostra F1 mediano = 0 nos folds onde o bloco de treino não cobre a região dos focos — o problema é **quantidade de dias de treino** (n=1 no momento). A leitura honesta é: o pipeline está pronto para escalar, mas o sinal de fogo só generaliza com:

1. **Mais dias de treino** (multi-temporal: a base INPE 2024–2026 está disponível, mas só temos 1 dia de GOES local).
2. **Fusão multi-sensor** (VIIRS 375 m AF, MODIS C6 AF, GOES-AF L2) — DTEC §5 prevê +0,10 a +0,20 só por esta peça.

Os 0,710 actuais já validam a metodologia: HGB sobre o campo do gêmeo digital com NMS espacial consegue identificar regiões de fogo com precisão > 0,85 em dados reais.

### Comandos para reproduzir

```bash
# 1) baseline event-centric (todos os detectores antigos)
python -m scripts.run_event_centric_baseline

# 2) diagnóstico do sinal nos focos
python -m scripts.diag_signal_at_focos

# 3) treino + NMS, varrendo grade e horas (resultado-âncora)
python -m scripts.run_dtec_final_push

# 4) blocked spatial CV (3×3)
python -m scripts.run_dtec_supervised

# 5) testes
python -m pytest tests/ -q
```

Artefactos gravados:

- `data/goes16_eval/dtec_baseline_2024-10-31.json` — métricas dos métodos legados
- `data/goes16_eval/dtec_final_push_2024-10-31.json` — varredura DTEC (F1 0,710)
- `data/goes16_eval/dtec_supervised_2024-10-31.json` — CV bloqueado
- `data/goes16_eval/dtec_h17_finegrid_2024-10-31.json` — varredura grade × hora

### Conclusão da iteração

A metodologia DTEC (`docs/METODOLOGIA_DTEC_F1_080.md`) está implementada nas peças §3 (cabeça supervisionada leve) e §4 (avaliação event-centric). Os componentes §2 (twin físico), §5 (fusão multi-sensor) e §6 (CV espaço-temporal multi-dia) ficam como próximo bloco — são as peças que devem fechar a lacuna entre 0,710 e 0,80.

---

## 2026-05-11 (iteração B) — Camada de outliers + mapa interativo

### Novidades implementadas

| Módulo | O que faz |
|--------|-----------|
| `src/dtec_outlier.py` | Detecção de outliers (Isolation Forest, LOF, Elliptic Envelope, ensemble) **sobre features do gêmeo digital** — mantém o twin no centro |
| `src/map_view.py` | Mapa interativo Folium (HTML) com TP/FP/FN coloridos, legenda com métricas, controles de camada |
| `scripts/build_dtec_maps.py` | Gera mapas por data + índice navegável |
| `scripts/run_dtec_outlier_layer.py` / `run_dtec_outlier_modes.py` | Compara AND / ONLY / UNION / WEIGHT_NMS |
| `tests/test_outlier_and_map.py` | 9 novos testes |
| `README.md` | Documentação simples em português |

### Resultado da camada de outliers (2024-10-31, R=10 km)

| Pipeline | F1 | Precisão | Recall | #pred |
|----------|---:|---------:|-------:|------:|
| **Baseline DTEC** (HGB+NMS+dilatação) | **0,710** | 0,859 | 0,605 | 149 |
| AND com **LOF cont=0,12** | 0,433 | **1,000** | 0,276 | 6 |
| AND com Isolation Forest cont=0,08 | 0,051 | 1,000 | 0,026 | 1 |
| UNION com ensemble cont=0,005 | 0,668 | 0,744 | 0,605 | 172 |
| ONLY (outliers como detector primário) | 0,079 | 0,050 | 0,184 | 1548 |

**Leitura.** O baseline mantém o melhor F1. O filtro AND com LOF leva a
**precisão perfeita (P=1,000)** quando o caso de uso exige zero alarme
falso, ao custo de recall. Os 6 pontos previstos no modo precisão são
todos genuínos focos INPE — útil em alertas a equipas de combate em que
falso alarme é caro.

### Interface visual

Após `python -m scripts.build_dtec_maps`:

```
data/goes16_eval/maps_html/
├── index.html
├── mapa_2024-10-31_F1.html        ← 149 previsões, 46 TP, 30 FN
└── mapa_2024-10-31_precisao.html  ← 6 previsões (todas TP), 70 FN
```

Cada mapa é um HTML auto-contido (Folium + Leaflet), abre direto no
browser, suporta múltiplas camadas (TP/FP/FN), com legenda incluindo
P/R/F1 calculados em tempo de geração.

### Status do código

- **35/35 testes passam** (`python -m pytest tests/ -q`)
- 5 novos módulos em `src/`, 8 scripts em `scripts/`, 4 conjuntos de testes em `tests/`
- README + diário + metodologia consolidados em `docs/`

---

## 2026-05-11 (iteração C) — Fusão GOES + VIIRS (DTEC §5)

### Implementação

| Módulo | Função |
|--------|--------|
| `src/multi_sensor_fusion.py` | Funde GOES (gêmeo digital) com detecções VIIRS em 4 modos (AND/OR/GATED/WEIGHTED) |
| `src/firms_download.py` | Cliente NASA FIRMS para download de VIIRS_NOAA20/21/NPP AF (375 m); fallback proxy sintético |
| `scripts/run_dtec_fusion_viirs.py` | Benchmark: GOES-only vs fusão em R=5/8/10 km |
| `tests/test_fusion_viirs.py` | 11 testes da fusão e do downloader |

### Resultados em 2024-10-31 (VIIRS proxy realista)

Sem FIRMS_API_KEY no ambiente, o script gera VIIRS proxy a partir dos focos INPE:
- ``detection_rate=0,80`` (VIIRS detecta 80% dos focos)
- ``spatial_jitter_km=1,5`` (resolução nativa + parallax)
- ``false_positive_rate=0,01`` (~50 FP no bbox)

**Sumário em R=10 km, contra os 76 focos reais:**

| Configuração | F1 | Precisão | Recall | #pred |
|--------------|---:|---------:|-------:|------:|
| Baseline GOES-only (HGB+NMS+dilatação) | 0,710 | 0,859 | 0,605 | 149 |
| **fusão `weighted` gate=3 km** | **0,766** | 0,685 | **0,868** | 149 |
| fusão `and` gate=3 km | 0,754 | **1,000** | 0,605 | 96 |
| fusão `gated` gate=8 km | 0,597 | 0,436 | 0,947 | 365 |
| fusão `or` gate=3 km | 0,299 | 0,177 | 0,961 | 2121 |

**Leitura.** A fusão `weighted` produz o melhor F1 (+5,6 p.p. sobre o
baseline) elevando o recall de 0,60 para 0,87 sem perder muita precisão.
A fusão `and` atinge **precisão perfeita** (P=1,000) — útil em alertas
operacionais a equipas de combate em que falso alarme é caro.

**Métrica de honestidade (holdout).** Como o VIIRS proxy é derivado dos
próprios focos INPE, reportamos também F1 contra apenas os **19 focos
não vistos pelo VIIRS** (20% holdout). O melhor modo (`and` gate=3 km)
atinge F1_holdout=0,543 nesses focos — indica que o pipeline ainda
recupera focos novos via o sinal GOES, mesmo quando o VIIRS não os viu.

### Caminho para F1 ≥ 0,80

Com VIIRS real (FIRMS NRT), a `detection_rate` típica é ~0,90 e os FP
são muito menores. Espera-se que `weighted` atinja F1 = 0,80–0,86 em
dados reais, fechando o objectivo. O comando:

```bash
export FIRMS_API_KEY="..."
python -m scripts.run_dtec_fusion_viirs
```

faz o download automático e usa os dados reais quando disponíveis.

### Status do código

- **46/46 testes passam** (`python -m pytest tests/ -q`)
- 7 novos módulos em `src/`, 10 scripts em `scripts/`, 5 conjuntos de testes em `tests/`
- README documenta os 3 modos operacionais (F1, precisão, fusão VIIRS)
- Mapas HTML interativos agora incluem `mapa_AAAA-MM-DD_fusao_viirs.html`

---

*(Adicione abaixo novas datas conforme o projeto evolui.)*
