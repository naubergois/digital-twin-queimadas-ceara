# 🔥 DTEC — Gêmeo Digital para Queimadas no Ceará

> **Em uma frase.** Pegamos imagens do satélite **GOES-16** e tentamos
> apontar **onde estão as queimadas** no Ceará, comparando o resultado com
> os focos reais detectados pelo **INPE**. O coração do método é um
> **gêmeo digital** que mantém um "estado de risco" do território e é
> atualizado a cada hora com novas observações.

[![tests](https://img.shields.io/badge/tests-46%20passing-brightgreen)](tests/)
[![F1@10km](https://img.shields.io/badge/F1%20%40%2010km-0.766-blue)](docs/EVOLUCAO_PESQUISA.md)
[![precision-mode](https://img.shields.io/badge/Precisão%20(modo%20outlier)-1.000-success)](docs/EVOLUCAO_PESQUISA.md)
[![fusion-precision](https://img.shields.io/badge/Precisão%20(fusão%20AND)-1.000-success)](docs/EVOLUCAO_PESQUISA.md)

---

## ❓ O que o projeto faz, em linguagem simples

Imagine que o satélite tira uma foto térmica do nordeste do Brasil de 10
em 10 minutos. Numa região seca como o sertão do Ceará, **muitos pixéis
ficam quentes** durante o dia — afloramentos de rocha, areia clara,
áreas urbanas. Só uma pequena parte desses pixéis quentes corresponde
realmente a fogo.

Para separar **fogo de pixel-quente-qualquer**, o projeto faz três
coisas:

1. **Gêmeo digital** — uma representação computacional do território que
   acumula o "risco" de fogo célula a célula, hora a hora. Quando o
   sensor vê algo anómalo, o gêmeo atualiza o risco daquela região.
2. **Classificador supervisionado leve** — uma regressão logística
   (ou Gradient Boosting) que aprende a separar fogo de não-fogo a
   partir dos focos INPE históricos, usando como entrada os **campos
   do gêmeo digital**.
3. **Filtro de outliers** — verifica se a célula prevista é
   estatisticamente "estranha" comparada ao resto da cena (Isolation
   Forest, LOF, Elliptic Envelope). Quando se quer **precisão máxima
   (modo outlier)**, isto leva a precisão de 0,86 → **1,000**.

No final é gerado um **mapa interativo** (HTML) que mostra, para cada
data, os focos reais (INPE) e as previsões do modelo, coloridos como
acerto (TP), alarme falso (FP) ou foco perdido (FN).

---

## 📊 Resultados em dados reais (Ceará, 2024-10-31, 76 focos)

| Modo | Como funciona | F1 | Precisão | Recall |
|------|---------------|---:|---------:|-------:|
| **F1-ótimo (fusão VIIRS, weighted)** | DTEC + score VIIRS ponderado | **0,766** | 0,685 | **0,868** |
| **Precisão-perfeita (fusão VIIRS, AND)** | DTEC ∩ vizinhança VIIRS | 0,754 | **1,000** | 0,605 |
| GOES-only (HGB + NMS + dilatação) | gêmeo + classificador supervisionado | 0,710 | 0,859 | 0,605 |
| Precisão-ótima (outlier LOF) | GOES + filtro LOF sobre features do gêmeo | 0,433 | **1,000** | 0,276 |
| Baseline antigo | métodos do repo antes do DTEC | 0,000 | 0,000 | 0,000 |

R = 10 km (avaliação event-centric).
A fusão usa **VIIRS-I AF 375 m** (NASA FIRMS) — em modo demo, com VIIRS
sintético realista (detection_rate=0,8, jitter 1,5 km, FP=1%).
Detalhes e protocolo em [`docs/EVOLUCAO_PESQUISA.md`](docs/EVOLUCAO_PESQUISA.md).

---

## 🚀 Como rodar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

Bibliotecas-chave: `numpy`, `pandas`, `scipy`, `scikit-learn`,
`xarray`, `netCDF4`, `folium`, `matplotlib`, `requests`.

### Reproduzir o resultado principal

Os ficheiros de exemplo já estão em `data/goes16_raw/` (1 dia, 3 horas,
3 canais) e `data/inpe_focos_ce/focos_ce_INPE_2024_2026.csv`.

```bash
# 1. Diagnóstico do sinal GOES nos focos INPE
python -m scripts.diag_signal_at_focos

# 2. Baseline event-centric (métodos antigos)
python -m scripts.run_event_centric_baseline

# 3. Resultado-âncora DTEC (F1 ≈ 0,71 em R=10 km)
python -m scripts.run_dtec_final_push

# 4. Comparação com filtro de outliers (modo precisão)
python -m scripts.run_dtec_outlier_modes

# 4b. Fusão GOES + VIIRS (DTEC §5) — F1 sobe de 0,71 para 0,77
python -m scripts.run_dtec_fusion_viirs

# 5. Mapas HTML interativos (real vs previsto + fusão)
python -m scripts.build_dtec_maps
# Abrir data/goes16_eval/maps_html/index.html no browser

# 6. Validação por blocos espaciais (CV 3x3)
python -m scripts.run_dtec_supervised

# 7. Testes
python -m pytest tests/ -q
```

---

## 🗺️ Mapa interativo

Após `python -m scripts.build_dtec_maps`:

```
data/goes16_eval/maps_html/
├── index.html                            ← abrir este no browser
├── mapa_2024-10-31_F1.html               ← GOES-only (F1 baseline)
├── mapa_2024-10-31_precisao.html         ← GOES + outlier LOF (P=1.000)
└── mapa_2024-10-31_fusao_viirs.html      ← GOES + VIIRS fusão (F1=0.766)
```

Cada mapa contém:

- 🟢 **Verde** — TP: ou foco INPE detectado, ou previsão que casa com foco
- 🟠 **Laranja** — FP: previsão sem foco INPE no raio de 10 km
- 🔴 **Vermelho** — FN: foco INPE que ninguém viu
- Legenda canto inferior esquerdo com Precision/Recall/F1
- Controles de camadas (mostrar/ocultar TP/FP/FN)
- Clique nos pontos para ver lat/lon/hora

---

## 🧠 Arquitetura DTEC (Dual-Twin Event-Centric)

```
   ┌──────────────────────────────┐    ┌──────────────────────────────┐
   │  GOES-16 ABI (CMIPF L2)      │    │  VIIRS-I AF 375 m            │
   │  canais 7 / 13 / 14 × horas  │    │  NASA FIRMS (NOAA-20/21/NPP) │
   └─────────────┬────────────────┘    └─────────────┬────────────────┘
                 │                                   │
   ┌─────────────▼─────────────────────┐             │
   │  GÊMEO DIGITAL (twin)             │             │
   │  src/goes_fire_digital_twin.py    │             │
   │   - score multi-escala T_B13      │             │
   │   - contraste BT7−BT14            │             │
   │   - persistência probabilística   │             │
   │   - campo de risco r(x)           │             │
   └───────┬──────────────────┬────────┘             │
           │                  │                      │
           ▼                  ▼                      ▼
   ┌──────────────┐  ┌──────────────────┐   ┌──────────────────────┐
   │ Features (6) │  │ Cabeça superv.HGB│   │ src/multi_sensor_    │
   │  twin_risk   │  │ + NMS espacial   │   │   fusion.py          │
   │  bt13_max    │  │ + dilatação      │   │  modos: AND / OR /   │
   │  bt7_max     │  └─────────┬────────┘   │  GATED / WEIGHTED    │
   │  btd_median  │            │            └───────────┬──────────┘
   │  bt13_anom21 │            │                        │
   │  persist_h   │  ┌─────────▼─────────────┐          │
   └──────┬───────┘  │ Outlier filter (twin) │          │
          │          │ src/dtec_outlier.py   │          │
          │          │  IsolForest / LOF /   │          │
          │          │  Elliptic Env. / ens. │          │
          │          └─────────┬─────────────┘          │
          │                    │                        │
          └─────────┬──────────┴──────────┬─────────────┘
                    ▼                     ▼
            ┌─────────────────────────────────────────┐
            │ Máscara final — 3 modos:                │
            │  • F1-ótimo (GOES+HGB+NMS+dilatação)    │
            │  • Precisão-ótima (+ filtro LOF)        │
            │  • F1-melhor (fusão weighted VIIRS)     │
            │  • Precisão-perfeita (fusão AND VIIRS)  │
            └────────────┬────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌────────────┐ ┌──────────────┐ ┌──────────────────┐
   │ grid F1    │ │ event-centric│ │ Folium HTML      │
   │ (legado)   │ │ R=3/5/8/10km │ │ TP/FP/FN + métri.│
   │            │ │ DTEC §4      │ │ src/map_view.py  │
   └────────────┘ └──────────────┘ └──────────────────┘
```

---

## 🧪 Por que o "gêmeo digital" é mais que um modelo

O `GOESFireDigitalTwin` (em [`src/goes_fire_digital_twin.py`](src/goes_fire_digital_twin.py))
implementa o ciclo **observação → assimilação → estado atualizado**
(ISO/IEC 23247) — não é apenas um detector "uma observação por vez":

1. **Memória entre horas** — o risco de cada célula persiste com
   decaimento, então um foco fraco visto em 3 horas seguidas pesa mais
   do que um único pico ruidoso.
2. **Fusão probabilística** — `r' = 1 - (1 - r·persistence)(1 - s)`
   combina risco anterior com observação nova de forma robusta.
3. **Limiar adaptativo** — usa GMM-2 ou percentil global; **as
   features do twin** são depois consumidas por DTEC.

Tudo isto faz com que o estado do gêmeo digital seja um **input
estável** para os passos seguintes (classificador supervisionado e
filtro de outliers). Sem o twin, cada hora seria um detector
independente e a precisão cai.

---

## 📁 Estrutura do repositório

```
.
├── README.md                       ← este ficheiro
├── requirements.txt
├── pytest.ini
├── config/
│   └── ceara_config.py             ← BBOX Ceará, ID IBGE
├── data/
│   ├── goes16_raw/                 ← NetCDFs GOES (ABI L2 CMIPF)
│   ├── goes16_eval/                ← métricas JSON + mapas HTML/PNG
│   └── inpe_focos_ce/              ← focos INPE (CSV)
├── docs/
│   ├── DOCUMENTACAO_PESQUISA_E_TESTES.md
│   ├── EVOLUCAO_PESQUISA.md        ← diário do projeto, métricas detalhadas
│   ├── METODOLOGIA_NOVA_PROPOSTA.md
│   └── METODOLOGIA_DTEC_F1_080.md  ← justificação do alvo F1 ≥ 0,8
├── scripts/
│   ├── diag_pred_vs_focos.py
│   ├── diag_signal_at_focos.py     ← descoberta-chave: BTD invertido no sertão
│   ├── diag_feature_separability.py
│   ├── run_event_centric_baseline.py
│   ├── run_dtec_grid_search.py
│   ├── run_dtec_supervised.py
│   ├── run_dtec_h17_fine_grid.py
│   ├── run_dtec_final_push.py
│   ├── run_dtec_outlier_layer.py
│   ├── run_dtec_outlier_modes.py
│   └── build_dtec_maps.py          ← gera HTML interativo
├── src/
│   ├── goes16_download.py
│   ├── goes16_ceara_image.py
│   ├── inpe_queimadas_download.py
│   ├── goes_fire_digital_twin.py   ← gêmeo digital (núcleo)
│   ├── goes_fire_method_v2.py      ← combined_persistence (legado)
│   ├── unsupervised_fire_goes.py   ← pipeline original + fix bug horário
│   ├── event_centric.py            ← métrica event-centric (DTEC §4)
│   ├── dtec_detector.py            ← detector não-supervisionado v2
│   ├── dtec_supervised.py          ← cabeça HGB + NMS (DTEC §3)
│   ├── dtec_outlier.py             ← filtro de outliers sobre o twin
│   ├── multi_sensor_fusion.py      ← fusão GOES + VIIRS (DTEC §5)
│   ├── firms_download.py           ← downloader NASA FIRMS (VIIRS AF)
│   └── map_view.py                 ← Folium → HTML interativo
└── tests/
    ├── test_dtec_detector_and_supervised.py
    ├── test_event_centric.py
    ├── test_fire_metrics_real.py
    ├── test_fire_metrics_synthetic.py
    ├── test_fusion_viirs.py
    └── test_outlier_and_map.py
```

---

## 🛰️ Fusão GOES + VIIRS (DTEC §5)

A peça que mais melhora F1 é a fusão com **VIIRS-I AF (375 m)** —
detecções de fogo de muito alta resolução publicadas pelo NASA FIRMS.
O módulo [`src/multi_sensor_fusion.py`](src/multi_sensor_fusion.py)
implementa 4 modos:

| Modo | Regra | Quando usar |
|------|-------|-------------|
| `AND` | GOES ∩ vizinhança VIIRS | Alerta de combate (zero falso alarme) |
| `OR` | GOES ∪ VIIRS | Cobertura máxima (recall) |
| `GATED` | GOES só conta perto de VIIRS; VIIRS sozinha já vale | Compromisso |
| `WEIGHTED` | score = 0,6·P_GOES + 0,4·indicador_VIIRS | F1-ótimo |

Resultado em 2024-10-31 (R=10 km, VIIRS proxy detection_rate=0,8):

| Modo | F1 | Precisão | Recall |
|---|---:|---:|---:|
| GOES-only baseline | 0,710 | 0,859 | 0,605 |
| **AND gate=3 km** | 0,754 | **1,000** | 0,605 |
| **WEIGHTED gate=3 km** | **0,766** | 0,685 | 0,868 |

Para usar VIIRS real (FIRMS NRT):

```bash
export FIRMS_API_KEY="$(cat ~/.firms_api_key)"
python -m scripts.run_dtec_fusion_viirs    # auto-download
```

---

## 🛠️ Personalização rápida

**Trocar a data alvo nos mapas:** editar `DATES` em
[`scripts/build_dtec_maps.py`](scripts/build_dtec_maps.py).

**Mudar o raio R da avaliação event-centric:** parâmetro `radius_km` em
qualquer chamada a `evaluate_event_centric(...)`.

**Mudar a sensibilidade da camada de outliers:**

```python
from src.dtec_outlier import OutlierConfig, filter_predictions_by_outlier

cfg = OutlierConfig(
    method="local_outlier_factor",  # ou isolation_forest, elliptic_envelope, ensemble
    contamination=0.12,             # subir = mais permissivo
)
pred_precise = filter_predictions_by_outlier(pred_base, feats, valid_bins, cfg=cfg)
```

---

## 📈 Caminho para F1 ≥ 0,80

A iteração actual:

| Etapa | F1 (R=10 km) | Status |
|-------|-------------:|--------|
| Métodos legados (spatial_residual / IF / twin / combined_persistence) | 0,000 | ✅ baseline |
| DTEC GOES-only (HGB + NMS + dilatação) | 0,710 | ✅ implementado |
| **DTEC + fusão VIIRS (weighted)** | **0,766** | ✅ implementado (modo demo) |
| DTEC + fusão VIIRS real (FIRMS NOAA-20/21/NPP) | est. **0,80–0,86** | 🔄 esperado |
| + CV espaço-temporal multi-dia | est. 0,82–0,88 | ⏳ próximo |
| + Twin físico (Rothermel-lite + ERA5) | est. 0,84–0,90 | ⏳ futuro |

Para activar a fusão com VIIRS real, basta exportar a API key da NASA
FIRMS e voltar a correr:

```bash
export FIRMS_API_KEY="sua-chave-firms"
python -m scripts.run_dtec_fusion_viirs
```

A chave é gratuita em https://firms.modaps.eosdis.nasa.gov/api/.

---

## 📚 Documentação adicional

- **Metodologia detalhada:** [`docs/METODOLOGIA_DTEC_F1_080.md`](docs/METODOLOGIA_DTEC_F1_080.md)
- **Diário de pesquisa:** [`docs/EVOLUCAO_PESQUISA.md`](docs/EVOLUCAO_PESQUISA.md)
- **Padrão de documentação:** [`docs/DOCUMENTACAO_PESQUISA_E_TESTES.md`](docs/DOCUMENTACAO_PESQUISA_E_TESTES.md)
- **Diagnóstico do tecto baixo:** [`docs/METODOLOGIA_NOVA_PROPOSTA.md`](docs/METODOLOGIA_NOVA_PROPOSTA.md)

---

## 📄 Dados e licenças

- **GOES-16 ABI L2 CMIPF** — NOAA Open Data Dissemination (público).
- **Focos INPE** — Programa Queimadas/CPTEC (público).

Este código é uma pesquisa académica; **não é um sistema operacional de
combate a incêndios**. Para alertas reais, consulte o INPE/CEMADEN.
