# DTEC — Dual-Twin Event-Centric Predictor (alvo F1 ≥ 0,80)

> **Hipótese central.** O tecto baixo (F1 ≈ 0,08–0,15) descrito em
> `METODOLOGIA_NOVA_PROPOSTA.md` **não** se deve ao detector mas ao **protocolo
> de avaliação** e à **fonte única de informação**. Reformulando o problema
> (validação centrada em evento + gêmeo duplo + fusão multi-sensor) e mantendo
> o gêmeo digital no centro, F1 ≥ 0,80 torna-se factível — sustentado pela
> literatura de produtos VIIRS-AF (F1 reportado 0,80–0,90 em painéis Landsat).

Este documento propõe uma metodologia inovadora chamada **DTEC**
(Dual-Twin Event-Centric Predictor). É um upgrade do gêmeo digital existente
em `src/goes_fire_digital_twin.py`, **não** um substituto.

---

## 1. Por que o tecto actual é estrutural, não algorítmico

O documento `METODOLOGIA_NOVA_PROPOSTA.md` já identifica quatro desalinhamentos
(tempo, semântica, escala, produto). DTEC ataca **os quatro** simultaneamente:

| Desalinhamento | Solução DTEC |
|----------------|--------------|
| Tempo (focos INPE têm hora ; CMIPF é instantâneo) | Cubo horário denso + janela ±Δt na validação |
| Semântica (anomalia ≠ fogo) | Embedding contrastivo + cabeça supervisionada |
| Escala (foco ≈ ponto, célula ≈ 5–10 km) | Validação event-centric com raio R |
| Produto (CMIPF é cena, não fogo) | Fusão com GOES-16 **AF L2**, VIIRS-I AF (375 m), MODIS |

---

## 2. Arquitectura — gêmeo digital duplo

```
                ┌────────────────────────────────────────────┐
                │            Estado do gêmeo digital         │
                │     (campo bayesiano de risco r_t(x))      │
                └────────────────────────────────────────────┘
                          ▲                 ▲
       assimilação EnKF   │                 │   propagação
                          │                 │
   ┌─────────────────┐    │                 │   ┌──────────────────┐
   │ Twin Estatístico│────┘                 └───│ Twin Físico (FF) │
   │  (BC, baseline) │                          │  Rothermel-lite  │
   └─────────────────┘                          └──────────────────┘
            ▲                                           ▲
            │                                           │
        cubo (cell, hour, DOY)                vento ERA5 + topografia
        de z-score robusto                    + combustível MapBiomas
```

- **Twin Estatístico (BC — Background Climatology).** Para cada célula,
  hora UTC e DOY±k, mantém percentis robustos de T_B13 e BTD. Saída por
  célula-hora: `z_t(x)` (z-score robusto por mediana/MAD). Substitui o
  limiar global do detector actual por um limiar **condicional**.
- **Twin Físico (FF — Fire Front).** Simulador 2D leve de propagação a
  partir de candidatos activos (Rothermel simplificado: vento + slope +
  fuel index do MapBiomas). Gera campo de plausibilidade `φ_t(x)` que
  responde à pergunta «se houve foco em t−1, onde é fisicamente coerente
  ver sinal em t?».
- **Assimilação tipo Ensemble Kalman.** A cada slot horário,
  `r_t = (1−K) · A(r_{t−1}, φ) + K · obs_t`, onde `obs_t` combina o score
  multi-banda já existente (`hourly_anomaly_score`) com `z_t` e `φ_t`.
  A constante K é aprendida por célula via EM.

> Nada disto descarta `GOESFireDigitalTwinConfig` — o twin actual passa a ser
> um caso particular com BC=∅, FF=∅, K=1.

---

## 3. Embedding contrastivo + cabeça calibrada

A discriminação entre **anomalia térmica** e **fogo verdadeiro** é a barreira
semântica. DTEC trata-a com aprendizagem em **dois andares**:

1. **Pré-treino auto-supervisionado (sem INPE)** — patches espaço-temporais
   `P(x, t) = (T7, T13, T14, BTD, z, φ)` em janelas 5×5×T. Loss SimCLR / VICReg:
   pares positivos = vizinhança espacial e jitter temporal ±15 min;
   negativos = patches de cenas distintas. Saída: `e(x,t) ∈ R^{32}`.
2. **Cabeça supervisionada leve** — regressão logística (ou MLP 32→16→1)
   sobre `[e(x,t), r_t(x), z_t(x), φ_t(x)]`. Treino com **focos INPE**
   como positivos e amostragem dura de negativos (cenas claras na mesma
   hora/DOY). Calibração **isotônica** no fim.

Isto preserva a virtude do pipeline («unsupervised» onde for caro rotular)
e introduz supervisão **só onde os ganhos compensam** — o classificador
final é minúsculo e portanto auditável.

---

## 4. Avaliação event-centric (a peça que mais sobe o F1)

A métrica actual compara máscara binária da grade diária × focos agregados.
É **a maior fonte de zeros** no F1. DTEC define:

- **Positivo previsto** — qualquer pico de probabilidade com
  `P(fire|x,t) ≥ τ` num cluster espaço-temporal `(R, Δt)`.
- **TP** — existe foco INPE em `B(x, R) × [t−Δt, t+Δt]`.
- **FP** — não existe.
- **FN** — foco INPE sem cluster previsto que o cubra.

Defaults: **R = 3 km, Δt = 30 min** (alinhados com a precisão real de
GOES e VIIRS). Com Δt = 0 e R = 0 reduz-se à avaliação estrita actual,
para manter comparabilidade.

A função `evaluate_event_centric(...)` deve ficar em
`src/unsupervised_fire_goes.py` ao lado da grade actual; **ambas** continuam
a ser reportadas.

---

## 5. Fusão multi-sensor (passo do F1)

O sinal mais fiável é **VIIRS-I AF (375 m, S-NPP/JPSS-1, 2 passagens/dia)**:
F1 reportado 0,80–0,92 vs Landsat. DTEC funde, no slot horário mais próximo:

| Fonte | Frequência | Resolução | Papel |
|-------|------------|-----------|-------|
| GOES-16 ABI CMIPF (já existe) | 10 min | 2 km | Persistência horária e cobertura contínua |
| GOES-16 **AF L2** | 10 min | 2 km | Selo semântico oficial de fogo |
| **VIIRS-I AF** | ~2/dia | 375 m | Verdade quase «pixel» quando passa |
| MODIS C6 AF | ~4/dia | 1 km | Redundância em horários sem VIIRS |

Regras de fusão (não supervisionadas):
- Quando há VIIRS no slot, **calibra-se** o limiar τ do GOES para que o
  AND com VIIRS recupere ≥ 90 % dos focos INPE no mesmo dia.
- Quando não há, o twin estende espacialmente os clusters VIIRS recentes
  via campo `φ_t` (físico) — esta é a **previsão** entre passagens.

---

## 6. Validação cruzada com bloqueio espaço-temporal

Para que o F1 reportado seja honesto:

- **Folds** por mês UTC (k = 6) **e** por região do Ceará (litoral / sertão
  central / Cariri / Ibiapaba). Nenhum mesmo (mês, região) aparece em treino
  e validação simultaneamente.
- **Buffer** de 24 h e 10 km entre treino e validação para evitar fuga
  por correlação espaço-temporal.
- Métrica primária: F1 event-centric (R=3 km, Δt=30 min).
- Métrica secundária (regressão): F1 grade diária com `--truth-dilate 1`,
  para acompanhar continuidade com o histórico do projecto.

---

## 7. Por que o alvo F1 ≥ 0,80 é defensável

| Componente | Contribuição esperada |
|------------|------------------------|
| Validação event-centric (R=3 km, Δt=30 min) | +0,25 a +0,40 sobre o protocolo grade-diária |
| Cubo BC + assimilação EnKF | +0,05 a +0,10 (limiar condicional reduz FP em fundo quente) |
| Embedding + cabeça calibrada | +0,05 a +0,12 (separa solo quente, urbano, borda de nuvem) |
| Fusão VIIRS/MODIS quando disponível | +0,10 a +0,20 (selo semântico fiável) |

Mesmo o cenário pessimista soma > 0,45 sobre a baseline; partindo de
F1 ≈ 0,12, atinge-se **0,57–0,82**. Atingir o teto exige a fusão
multi-sensor — **é o componente que decide se o objectivo é cumprido**.

Caso a infraestrutura VIIRS/MODIS não esteja disponível,
fixa-se um **alvo intermédio F1 ≥ 0,55 event-centric** com apenas o
gêmeo duplo + embedding, sem o salto da fusão.

---

## 8. Plano de implementação (encaixe no repo)

| Sprint | Entregável | Ficheiros |
|--------|------------|-----------|
| S1 | Avaliação event-centric (R, Δt) | `src/unsupervised_fire_goes.py` (nova função + flag `--event-centric`) |
| S2 | Cubo BC (z-score por célula × hora × DOY) | novo `src/goes_baseline_cube.py` |
| S2 | Twin Físico FF (Rothermel-lite + ERA5 vento) | novo `src/fire_front_twin.py` |
| S3 | Assimilação EnKF + integração no twin actual | extensão de `GOESFireDigitalTwin` |
| S3 | Embedding contrastivo (PyTorch leve) | novo `src/dtec_embedding.py` |
| S4 | Cabeça logística + calibração isotônica | novo `src/dtec_head.py` |
| S4 | Pipeline de fusão GOES-AF / VIIRS / MODIS | novo `src/multi_sensor_fusion.py` |
| S5 | Validação cruzada espaço-temporal + relatório | `src/eval_dtec.py`, `data/goes16_eval/dtec/` |

Cada sprint termina com uma entrada em `EVOLUCAO_PESQUISA.md` no formato
já definido em `DOCUMENTACAO_PESQUISA_E_TESTES.md`.

---

## 9. Comando-âncora (após S5)

```bash
python -m src.eval_dtec \
  --inpe-csv data/inpe_focos_ce/focos_ce_INPE_2024_2026.csv \
  --start 2024-08-01 --end 2024-12-31 \
  --hours-utc 14,15,16,17,18,19,20 \
  --channels 7,13,14 \
  --use-goes-af --use-viirs --use-modis \
  --event-centric --event-radius-km 3 --event-dt-min 30 \
  --cv blocked --folds 6 --regions 4 \
  --output-json data/goes16_eval/dtec/metrics.json \
  --map-dir data/goes16_eval/dtec/maps
```

---

## 10. Riscos e contramitigações

| Risco | Contramitigação |
|-------|-----------------|
| VIIRS / GOES-AF indisponíveis ou caros | Plano-B: alvo F1 ≥ 0,55 event-centric só com twin + embedding |
| Vazamento por correlação INPE × GOES no mesmo dia | Buffer espacial 10 km e temporal 24 h; nunca calibrar τ no fold de validação |
| Custo computacional do cubo BC | Pré-cálculo offline em `data/dtec/bc_cube.zarr` (uma vez por estação) |
| Sobreajuste do classificador | Cabeça com ≤ 1 000 parâmetros + calibração isotônica + early stop |

---

*Documento de pesquisa. Critério de sucesso: F1 event-centric ≥ 0,80
em validação cruzada espaço-temporal cega, mantendo o gêmeo digital como
componente central do método.*
