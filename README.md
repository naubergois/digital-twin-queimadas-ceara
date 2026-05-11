# 🔥 DTEC — Gêmeo Digital para Queimadas no Ceará

> **Em uma frase.** Pegamos imagens do satélite **GOES-16** e tentamos
> apontar **onde estão as queimadas** no Ceará, comparando o resultado com
> os focos reais detectados pelo **INPE**. O coração do método é um
> **gêmeo digital** que mantém um "estado de risco" do território e é
> atualizado a cada hora com novas observações.

[![tests](https://img.shields.io/badge/tests-35%20passing-brightgreen)](tests/)
[![F1@10km](https://img.shields.io/badge/F1%20%40%2010km-0.710-blue)](docs/EVOLUCAO_PESQUISA.md)
[![precision-mode](https://img.shields.io/badge/Precisão%20(modo%20outlier)-1.000-success)](docs/EVOLUCAO_PESQUISA.md)

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
| **F1-ótimo** | gêmeo + HGB + NMS + dilatação | **0,710** | 0,859 | 0,605 |
| **Precisão-ótima** | acima + filtro LOF sobre features do gêmeo | 0,433 | **1,000** | 0,276 |
| Baseline antigo | métodos do repo antes do DTEC | 0,000 | 0,000 | 0,000 |

R = 10 km (avaliação event-centric).
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

# 5. Mapas HTML interativos (real vs previsto)
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
├── index.html                       ← abrir este no browser
├── mapa_2024-10-31_F1.html          ← modo F1-ótimo
└── mapa_2024-10-31_precisao.html    ← modo precisão (outlier filter)
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
                   ┌──────────────────────────────────┐
                   │     GOES-16 ABI (CMIPF L2)       │
                   │   canais 7 / 13 / 14 × horas     │
                   └────────────────┬─────────────────┘
                                    │
                ┌───────────────────▼───────────────────┐
                │      GÊMEO DIGITAL (twin)             │
                │  src/goes_fire_digital_twin.py        │
                │   - score multi-escala em T_B13       │
                │   - contraste BT7−BT14                │
                │   - persistência probabilística       │
                │   - campo de risco contínuo r(x)      │
                └────────────────┬──────────────────────┘
                                 │
            ┌────────────────────┼────────────────────────┐
            ▼                    ▼                        ▼
    ┌──────────────┐  ┌────────────────────┐   ┌──────────────────┐
    │ Features (6) │  │ Cabeça superv. HGB │   │ Outlier filter   │
    │ bt13_max     │  │ src/dtec_supervised│   │ src/dtec_outlier │
    │ bt7_max      │  │  → P(fogo|x)       │   │  - IsolForest    │
    │ btd_median   │  │ + NMS espacial     │   │  - LOF           │
    │ twin_risk    │  │ + dilatação 2 it.  │   │  - Elliptic Env. │
    │ bt13_anom_21 │  └──────────┬─────────┘   │  - ensemble (≥2) │
    │ persist_h    │             │              └─────────┬────────┘
    └──────────────┘             ▼                        ▼
                          ┌────────────────────────────────────┐
                          │  Máscara de previsão (modo F1)     │
                          │  ou (modo PRECISÃO = ∩ outliers)   │
                          └─────────────────┬──────────────────┘
                                            │
                  ┌─────────────────────────┼──────────────────────────┐
                  ▼                         ▼                          ▼
           ┌─────────────────┐   ┌──────────────────────┐    ┌──────────────────┐
           │ avaliação grid  │   │ avaliação event-     │    │ mapa interativo  │
           │ (legado)        │   │ centric (DTEC §4)    │    │ Folium HTML      │
           │                 │   │  R = 3/5/8/10 km     │    │ TP/FP/FN coloridos│
           │ confusion(...)  │   │ src/event_centric.py │    │ src/map_view.py  │
           └─────────────────┘   └──────────────────────┘    └──────────────────┘
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
│   └── map_view.py                 ← Folium → HTML interativo
└── tests/
    ├── test_dtec_detector_and_supervised.py
    ├── test_event_centric.py
    ├── test_fire_metrics_real.py
    ├── test_fire_metrics_synthetic.py
    └── test_outlier_and_map.py
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

O resultado atual (0,710) usa **um único dia** de GOES local. A
metodologia DTEC ([docs/METODOLOGIA_DTEC_F1_080.md](docs/METODOLOGIA_DTEC_F1_080.md))
prevê o resto:

| Componente | Ganho previsto em F1 |
|------------|----------------------|
| Mais dias de GOES no treino + CV espaço-temporal | +0,05 a +0,10 |
| Fusão multi-sensor (VIIRS 375 m AF, MODIS, GOES-AF L2) | +0,10 a +0,20 |
| Twin físico (Rothermel-lite + vento ERA5) | +0,02 a +0,05 |

Total esperado: 0,86 a 1,05 → **alvo 0,80 alcançável** com fusão
multi-sensor (próximo bloco de implementação).

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
