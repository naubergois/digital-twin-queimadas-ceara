# PYRO-Caatinga MVP

## Visao Geral
PYRO-Caatinga e uma proposta para deteccao de queimadas no GOES-16 com quatro blocos integrados:
1. Climatology-residual front-end
2. Modelo espaco-temporal causal (no MVP: cabeca simplificada em score residual)
3. Cabecas fisicas (mascara e proxy de FRP)
4. Loop de feedback com o gemeo digital (pseudo-rotulos em t+5)

O objetivo principal e reduzir falso positivo em solo quente da Caatinga e melhorar deteccao operacional.

## O que foi implementado neste repositorio

Arquivo principal: src/pyro_caatinga.py

### Bloco 1 - Climatologia residual online
- Classe: ClimatologyResidualFrontEnd
- Formula: BT~(p,t) = BT(p,t) - mu(p, doy, hod)
- Atualizacao de mu por EWMA (lambda configuravel, default 0.05)

### Bloco 2 - Destilacao cruzada VIIRS -> GOES
- Classe: ViirsGoesDistiller
- Construcao de soft labels por reprojecao de pontos VIIRS para grade GOES
- Suavizacao gaussiana para representar incerteza sub-pixel

### Bloco 3 - Cabeca fisica simplificada
- Probabilidade baseline a partir de BT bruto
- Probabilidade residual a partir de BT residual
- Proxy FRP com base em energia termica relativa entre bandas residuais

### Bloco 4 - Loop de feedback do gemeo digital
- Classe: DigitalTwinFeedbackLoop
- Gera pseudo-rotulos confiaveis com base em:
  - incerteza epistêmica proxy (variancia MC-dropout)
  - consistencia entre previsao do twin e observacao em t+5
- Ativacao com limiar de incerteza tau

## Runner CLI

Arquivo: src/run_pyro_caatinga.py

Comando recomendado:

```bash
python -m src.run_pyro_caatinga \
  --goes-csv data/focos_CE_GOES16_2024.csv \
  --output-dir data/pyro_caatinga \
  --max-days 7
```

Comando equivalente via main:

```bash
python main.py \
  --pyro-mvp \
  --pyro-goes-csv data/focos_CE_GOES16_2024.csv \
  --pyro-output-dir data/pyro_caatinga \
  --pyro-max-days 7
```

## Artefatos de saida

- data/pyro_caatinga/pyro_goes_proxy_cube.nc
- data/pyro_caatinga/pyro_residual_cube.nc
- data/pyro_caatinga/pyro_viirs_soft_labels.npy
- data/pyro_caatinga/pyro_twin_pseudo_labels.npy
- data/pyro_caatinga/pyro_frp_proxy.npy
- data/pyro_caatinga/pyro_caatinga_report.json

## Limites do MVP atual

- O cubo GOES e proxy (derivado de pontos) e nao ingestao ABI/GLM nativa.
- O bloco espaco-temporal usa uma formulacao simplificada em vez de Transformer/Mamba completo.
- O loop com twin e auto-supervisao inicial (pseudo-rotulos), sem treino iterativo de rede profunda nesta versao.
- Para garantir execucao diaria estavel, o cubo temporal e montado com janela recente (max_days, default 7).

## Proximo passo recomendado

1. Ingestao real de bandas ABI (B2, B6, B7, B14) e GLM por frame de 5 minutos.
2. Treino do aluno causal (Swin-T/Mamba pequeno) com distilacao KL + BCE hard.
3. Integracao de vento ERA5/Open-Meteo no simulador do twin para feedback fisico mais forte.
