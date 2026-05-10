# Detecção de queimadas em GOES-16: estado da arte e proposta inovadora

> Pesquisa de papers recentes (2022–2025) sobre técnicas de deep learning aplicadas
> a imagens GOES-16/19 ABI para detecção de queimadas, com proposta de uma
> arquitetura nova adaptada ao bioma Caatinga e ao gêmeo digital do Ceará.

---

## 1. Contexto

- **Sensor:** GOES-16 ABI — substituído operacionalmente por GOES-19 em 07/04/2025,
  mesmo instrumento e mesmas bandas, então toda a literatura aqui se aplica.
- **Bandas-chave para fogo:** B2 (0.64 µm, visível), B7 (3.9 µm, MIR), B14 (11.2 µm, TIR).
- **Cadência:** 5 min CONUS / 10 min full-disk a 2 km de resolução.
- **Produto operacional de referência:** GOES-R FDC (Fire/Hot Spot Characterization)
  — máscara, temperatura, área e FRP sub-pixel.

### Desafios específicos do Ceará / Caatinga
- Solo nu pode atingir 50–60 °C ao meio-dia, gerando falsos positivos em
  thresholds contextuais e CNNs sem normalização adequada.
- Fogos pequenos (< 4 ha) são frequentes e ficam abaixo da resolução de 2 km do GOES.
- Disponibilidade limitada de rótulos regionais; principais fontes são
  BDQueimadas/INPE e VIIRS/FIRMS (NASA).

---

## 2. Estado da arte: técnicas recentes (2022–2025)

### 2.1 Séries temporais com RNN (GRU/LSTM)
**Zhao & Ban (2022)** — *GOES-R Time Series for Early Detection of Wildfires with Deep GRU-Network*
- Séries densas de B7 e B14 alimentam uma GRU empilhada pixel-a-pixel.
- Detecção mais cedo que VIIRS active-fire e mais precisa que GOES-R FDC em
  latitudes médias/baixas — relevante para o Nordeste.
- GRU tem ~metade dos parâmetros de uma LSTM com desempenho similar; viável em
  inferência quase em tempo real.
- Link: https://www.mdpi.com/2072-4292/14/17/4347

### 2.2 CNN + redução de latência
**Hong et al. (2022)** — *A deep learning model using geostationary satellite data for forest fire detection with reduced detection latency* (GIScience & Remote Sensing)
- Combina contexto espacial (CNN) com cadência geoestacionária para reduzir a
  latência típica do FDC (que precisa acumular varreduras antes de confirmar).
- Link: https://www.tandfonline.com/doi/full/10.1080/15481603.2022.2143872

### 2.3 Auto-supervisão multi-sensor (GOES + TEMPO)
**arXiv:2510.09845 (out/2025)** — *Harnessing Self-Supervised Deep Learning and Geostationary Remote Sensing for Advancing Wildfire and Associated Air Quality Monitoring*
- Pré-treino auto-supervisionado em radiâncias de GOES-18 ABI + TEMPO; aprende
  representações sem máscaras rotuladas.
- Diferencia plumas de fumaça vs. nuvens e mapeia frentes de fogo near-real-time.
- Reporta concordância forte entre máscaras de sensores diferentes e melhoria
  sobre produtos operacionais.
- Link: https://arxiv.org/abs/2510.09845

### 2.4 Datasets multi-tarefa (benchmark)
**TS-SatFire (Nature Scientific Data, 2025)** — *A Multi-Task Satellite Image Time-Series Dataset for Wildfire Detection and Prediction*
- Séries temporais públicas para detecção, segmentação ativa e predição de
  propagação. Bom ponto de partida para fine-tuning regional.
- Link: https://www.nature.com/articles/s41597-025-06271-3

### 2.5 Algoritmo clássico aprimorado — baseline forte
**Li et al. (2021)** — Fire Thermal Anomaly (FTA) sobre GOES-16 ABI
- Não é deep learning, mas é o baseline a bater: thresholding contextual
  multi-espectral em B2, B7 e B14 com refinamento sub-pixel para FRP.
- Cobertura América inteira, validado no Brasil.
- Link: https://www.sciencedirect.com/science/article/pii/S2666017221000031

### 2.6 Vision Transformers e híbridos CNN+ViT
- **arXiv:2504.13776 (2025)** — ViTs competem com CNNs (UNet customizada chegou
  a IoU 93.58% em Landsat-8); ViTs treinam mais barato.
  https://arxiv.org/abs/2504.13776
- **CN2VF-Net (MDPI Fire, 2025)** — híbrido CNN+ViT multi-escala; útil para
  fogos pequenos sub-pixel.
  https://www.mdpi.com/2571-6255/8/6/211

### 2.7 Aplicação no Brasil (Cerrado)
**SEMFOGO-DF (Springer, 2025)** — *A fire management intelligent system for the Brazilian cerrado biome based on a deep learning two phase detection method*
- Sistema operacional two-phase de detecção de fumaça via CV; não usa GOES-16
  diretamente, mas mostra o pipeline operacional típico para o Brasil.
- Link: https://link.springer.com/article/10.1007/s40860-025-00244-4

### 2.8 Reconstrução generativa de progressão
**arXiv:2506.10404 (2025)** — *Generative Algorithms for Wildfire Progression Reconstruction from Multi-Modal Satellite Active Fire Measurements and Terrain Height*
- Combina GOES com outros sensores via modelos generativos para reconstruir
  frentes de fogo entre varreduras.
- Link: https://arxiv.org/html/2506.10404

### 2.9 Predição de propagação
**arXiv:2505.17556 (2025)** — *Wildfire spread forecasting with Deep Learning*
- U-Net 3D / ViT em datacube Mesogeos (Mediterrâneo); janela temporal 10 dias.
- Não usa GOES, mas a arquitetura U-Net 3D é transferível.
- Métricas: Dice 53.6%, IoU 36.6%, Precisão 59.6%, Recall 48.8%.
- Link: https://arxiv.org/html/2505.17556v1

---

## 3. Tabela comparativa do estado da arte

| # | Técnica / Paper | Sensor | Arquitetura | Bandas usadas | Métrica reportada | Latência | Aplicável Caatinga? |
|---|---|---|---|---|---|---|---|
| 1 | GRU GOES-R (Zhao & Ban 2022) | GOES-R ABI | GRU empilhada | B7, B14 | detecção mais cedo que VIIRS, mais precisa que FDC | ~5 min | parcial (sofre com BT alto de solo) |
| 2 | CNN geoestacionário (Hong 2022) | GOES geo | CNN espacial | B7, B14, B2 | redução de latência vs FDC | < FDC | parcial |
| 3 | Self-supervised GOES+TEMPO (2510.09845) | GOES-18 + TEMPO | encoder auto-supervisionado | radiâncias multi-banda | concordância forte entre sensores; melhoria sobre produtos operacionais | horária | sim (não exige rótulo) |
| 4 | TS-SatFire (Nature 2025) | multi-sensor | benchmark multi-tarefa | série temporal | dataset, não modelo único | n/a | dataset US/Canada — precisa fine-tuning |
| 5 | FTA contextual (Li 2021) | GOES-16 ABI | thresholding multi-espectral | B2, B7, B14 | melhoria sobre FDC; FRP sub-pixel | ~5–10 min | baseline; muitos FP em solo quente |
| 6 | ViT (2504.13776) | Landsat-8 | ViT vs CNN | RGB+NIR | UNet IoU 93.58% (Landsat) | n/a (não-geo) | adaptável |
| 7 | CN2VF-Net (MDPI Fire 2025) | imagens RGB/IR | CNN+ViT híbrido | multi-escala | F1 alto em fogos pequenos | n/a | adaptável |
| 8 | SEMFOGO-DF (Springer 2025) | câmeras + CV | CNN two-phase | RGB | sistema operacional Cerrado | tempo real | conceitual |
| 9 | Generativo (2506.10404) | GOES + multi | modelo generativo | MIR/TIR + DEM | reconstrução de frente entre varreduras | retrospectivo | sim |
| 10 | U-Net3D Mesogeos (2505.17556) | sensores diversos | U-Net 3D / ViT | datacube 10d | Dice 53.6% / IoU 36.6% | retrospectivo | sim, requer adaptação |

### Lacunas identificadas (gap analysis)
1. **Nenhum trabalho remove explicitamente o fundo térmico do bioma** (BT sazonal-diurna).
2. **Nenhum** combina destilação VIIRS→GOES (375 m → 2 km) como *teacher–student* multi-resolução.
3. **Nenhum** fecha o laço com um simulador de propagação para auto-supervisão online.
4. Aplicações brasileiras existentes (SEMFOGO-DF) usam câmeras locais, não GOES.

---

## 4. Proposta inovadora: **PYRO-Caatinga**

> *Climatology-Residual Spatiotemporal Transformer com destilação cruzada VIIRS→GOES
> e laço de gêmeo digital para detecção sub-pixel de fogo no semi-árido.*

### 4.1 Pitch
PYRO-Caatinga ataca simultaneamente os três gaps acima: **(a)** remove o viés de
solo quente via resíduo climatológico por pixel/hora, **(b)** destila a precisão
do VIIRS (375 m, 2× ao dia) para um aluno GOES (2 km, 5 min), e **(c)** acopla a
saída ao simulador de propagação do gêmeo digital, transformando o erro de
predição em sinal de auto-supervisão online.

### 4.2 Arquitetura em 4 blocos

```
                ┌────────────────────────────────────────────────────────┐
                │ 1. CLIMATOLOGY-RESIDUAL FRONT-END                      │
GOES-16 ABI ──► │   BT̃(p,t) = BT(p,t) − μ̂(p, doy, hod)                  │
B2,B6,B7,B14    │   μ̂ = média EWMA pixel-a-pixel por dia-do-ano          │
+ GLM raios     │   e hora-do-dia (λ=0.05); robust median, excluindo     │
                │   pixels do BDQueimadas histórico                      │
                └────────────────────────────────────────────────────────┘
                                          │
                ┌─────────────────────────▼──────────────────────────────┐
                │ 2. SPATIOTEMPORAL CAUSAL MAMBA / SWIN-T                │
                │   janela 12 frames × 64×64 px (≈1h de histórico)       │
                │   atenção causal → streaming a 5 min                   │
                └─────────────────────────┬──────────────────────────────┘
                                          │
                ┌─────────────────────────▼──────────────────────────────┐
                │ 3. CABEÇAS FÍSICAS                                     │
                │   • máscara de fogo sub-pixel (BCE + Dice)             │
                │   • FRP regressão (MAE) consistente com Stefan-        │
                │     Boltzmann em B7+B14                                │
                │   • incerteza epistêmica via MC-dropout (T=20)         │
                └─────────────────────────┬──────────────────────────────┘
                                          │
                ┌─────────────────────────▼──────────────────────────────┐
                │ 4. DIGITAL-TWIN FEEDBACK LOOP                          │
                │   ŷ(t+5) → simulador CA + vento ERA5 → compara com     │
                │   GOES real em t+5 → erro vira pseudo-rótulo;          │
                │   self-distillation se incerteza < τ                   │
                └────────────────────────────────────────────────────────┘
```

### 4.3 Receita de treino (MVP em 4 semanas)

| Fase | Duração | Entrega |
|---|---|---|
| **A — Climatologia** | 1 sem | Cubo NetCDF μ̂(p, doy, hod) sobre 90 dias de GOES-16 no Ceará, B7/B14/ΔBT |
| **B — Pré-treino auto-supervisionado** | 1 sem | Masked autoencoding sobre BT̃; pretexto: prever frame mascarado dado contexto de 1h |
| **C — Destilação VIIRS→GOES** | 1 sem | Loss = α·BCE_hard(BDQueimadas) + (1−α)·KL(student ∥ VIIRS_soft); soft-labels VNP14IMG reprojetadas em kernel gaussiano para grid GOES |
| **D — Twin loop** | 1 sem | Acoplamento ao módulo CA de spread; self-distillation ativo quando var(MC-dropout) < τ |

### 4.4 Métricas-alvo vs FDC oficial GOES-16

| Métrica | FDC baseline (Caatinga) | PYRO-Caatinga (alvo) | Ganho |
|---|---|---|---|
| Latência de detecção | ~10 min | ≤ 5 min (1 frame) | −50% |
| Falsos positivos em meio-dia | alta | ↓ ≥ 60% | climatology-residual |
| Recall em fogos < 4 ha | baixa (≈ sub-pixel) | ↑ ≥ 25% | destilação VIIRS sub-pixel |
| F1 vs BDQueimadas/INPE no Ceará | ~0.55 | ≥ 0.78 | combinação dos três |

### 4.5 Riscos e mitigações
- **VIIRS nublado quando GOES não está** → mascarar pixels com nuvem (B2 + cirrus B4) antes de aplicar a *soft label*.
- **Climatologia pode "absorver" fogos recorrentes** → mediana truncada robusta + exclusão de pixels do histórico BDQueimadas no cálculo de μ̂.
- **Custo do MC-dropout** → executar apenas no top-K de pixels candidatos por escore inicial.

---

## 5. PYRO-Caatinga vs. estado da arte

| Componente da proposta | Inspirado em | O que está faltando na referência | O que PYRO-Caatinga adiciona |
|---|---|---|---|
| Front-end climatology-residual | GRU GOES-R (Zhao & Ban 2022) | usa BT bruto, sofre em solo quente | resíduo BT̃ por pixel/hora — remove viés de fundo da Caatinga |
| Backbone causal Mamba/Swin-T | ViT/CN2VF-Net | sem dimensão temporal causal de streaming | janela deslizante de 1h com inferência incremental a 5 min |
| Destilação VIIRS→GOES | self-supervised arXiv:2510.09845 | só pré-treino em radiâncias, sem multi-resolução | *teacher* VIIRS 375 m → *student* GOES 2 km via KL em soft-labels |
| Cabeças físicas (FRP Stefan-Boltzmann) | FTA (Li 2021) | thresholds fixos, sem incerteza | regressão consistente + MC-dropout para incerteza epistêmica |
| Loop com simulador (CA + ERA5) | TS-SatFire / Mesogeos | só treino offline, sem feedback do twin | erro de previsão t+5 vira pseudo-rótulo (self-distillation online) |
| Aplicação regional Brasil | SEMFOGO-DF | usa câmeras locais, não GOES | GOES + VIIRS + ERA5 + BDQueimadas, foco Caatinga |

### Tabela final: PYRO-Caatinga vs. principais técnicas

| Aspecto | FDC oficial | GRU GOES-R | Self-Sup GOES+TEMPO | CN2VF-Net | **PYRO-Caatinga** |
|---|---|---|---|---|---|
| Remove fundo térmico do bioma | não | não | não | não | **sim (EWMA por pixel/hora)** |
| Multi-resolução teacher–student | não | não | parcial | não | **sim (VIIRS→GOES)** |
| Streaming 5 min | sim | sim | horário | n/a | **sim (causal)** |
| Saída física (FRP) | sim | não | parcial | não | **sim (Stefan-Boltzmann)** |
| Incerteza calibrada | não | não | não | não | **sim (MC-dropout)** |
| Loop com simulador | não | não | não | não | **sim (twin feedback)** |
| Adaptado à Caatinga | não | não | não | não | **sim (rótulos BDQueimadas + fine-tuning regional)** |

---

## 6. Próximos passos no repositório

1. Esqueleto do cubo de climatologia μ̂ em `src/preprocess/climatology.py` (90 dias de GOES-16 sobre Ceará).
2. Módulo de destilação `src/training/distill_viirs.py` com VNP14IMG do FIRMS reprojetado.
3. Backbone Mamba pequeno (~5M params) em `src/models/pyro_caatinga.py`, treinável em CPU.
4. Acoplamento ao módulo CA de spread já existente — endpoint para receber ŷ(t+5) e devolver erro.
5. Dashboard: nova camada com máscara PYRO + barra de incerteza, sobreposta ao GIBS já integrado.

---

## 7. Referências

- [GOES-R Time Series for Early Detection of Wildfires with Deep GRU-Network (MDPI 2022)](https://www.mdpi.com/2072-4292/14/17/4347)
- [Harnessing Self-Supervised Deep Learning and Geostationary Remote Sensing — arXiv:2510.09845](https://arxiv.org/abs/2510.09845)
- [TS-SatFire: A Multi-Task Satellite Image Time-Series Dataset (Nature Scientific Data 2025)](https://www.nature.com/articles/s41597-025-06271-3)
- [A deep learning model using geostationary satellite data for forest fire detection with reduced detection latency (GIScience & RS 2022)](https://www.tandfonline.com/doi/full/10.1080/15481603.2022.2143872)
- [Improvements in high-temporal resolution active fire detection — GOES-16 ABI FTA](https://www.sciencedirect.com/science/article/pii/S2666017221000031)
- [Fighting Fires from Space: Vision Transformers for Wildfire Detection — arXiv:2504.13776](https://arxiv.org/abs/2504.13776)
- [CN2VF-Net: Hybrid CNN+ViT for Multi-Scale Fire Detection (MDPI Fire 2025)](https://www.mdpi.com/2571-6255/8/6/211)
- [SEMFOGO-DF: Fire management system for the Brazilian Cerrado (Springer 2025)](https://link.springer.com/article/10.1007/s40860-025-00244-4)
- [Wildfire spread forecasting with Deep Learning — arXiv:2505.17556](https://arxiv.org/html/2505.17556v1)
- [Generative Algorithms for Wildfire Progression Reconstruction — arXiv:2506.10404](https://arxiv.org/html/2506.10404)
- [Deep Learning Approaches for Wildland Fires Using Satellite Remote Sensing (MDPI Fire 2023)](https://www.mdpi.com/2571-6255/6/5/192)
- [GOES-16 ABI L2 Fire/Hot Spot Characterization product (NOAA)](https://www.goes-r.gov/products/baseline-fire-hot-spot.html)
