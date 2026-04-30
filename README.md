# Digital Twin para Detecção de Queimadas no Ceará

**Proposta Funcional** — Monitoramento e predição de queimadas usando dados abertos de satélite e gêmeos digitais.

## Objetivo

Construir um gêmeo digital (digital twin) do estado do Ceará para:
- **Detectar** focos de queimadas em tempo real via satélites (INPE/NASA FIRMS)
- **Simular** a propagação do fogo baseado em condições ambientais
- **Visualizar** em um dashboard interativo 2D/3D
- **Validar** o modelo comparando predições com observações reais

## Dados Abertos Utilizados

| Dado | Fonte | Acesso |
|------|-------|--------|
| Focos de calor (MODIS/VIIRS) | INPE BDQueimadas | bdqueimadas.dpi.inpe.br |
| Imagens multiespectrais | ESA Copernicus Sentinel-2 | scihub.copernicus.eu |
| Meteorologia | FUNCEME / ERA5 | funceme.br |
| Vegetação/Cobertura | MapBiomas | mapbiomas.org |
| Topografia (SRTM) | USGS | earthexplorer.usgs.gov |
| Limites municipais | IBGE | ibge.gov.br |

## Estrutura do Projeto

```
digital-twin-queimadas-ceara/
├── README.md              ← Esta proposta
├── requirements.txt       ← Dependências Python
├── config/
│   └── ceara_config.py    ← Configurações do Ceará (municípios, áreas críticas)
├── data/
│   └── (dados baixados aqui - gitignored)
├── src/
│   ├── fire_data.py       ← Download e parsing de focos de calor
│   ├── digital_twin.py    ← Motor do gêmeo digital (predição de propagação)
│   └── analysis.py        ← Análise temporal e estatística
├── notebooks/
│   └── proposta_funcional.ipynb  ← Notebook principal com demo
└── dashboard/
    └── app.py             ← Dashboard interativo (Streamlit)
```

## Como Executar

```bash
# Instalar dependências
pip install -r requirements.txt

# Executar pipeline completa
python -m src.fire_data --state CE --year 2025

# Executar dashboard
streamlit run dashboard/app.py

# Executar notebook
jupyter notebook notebooks/proposta_funcional.ipynb
```

## Funcionalidades Implementadas

- [x] Download automático de focos de calor do INPE para o Ceará
- [x] Mapeamento de focos por município e bioma
- [x] Modelo simplificado de propagação de fogo (algoritmo de cell automaton)
- [x] Dashboard interativo com mapa de calor
- [x] Análise temporal (sazonalidade, tendência anual)
- [ ] Integração com Sentinel-2 para validação de cicatrizes de queimadas
- [ ] Integração com dados meteorológicos da FUNCEME
- [ ] Modelo de machine learning para predição de risco

## Próximos Passos

1. Expandir o modelo de propagação com dados reais de vento e umidade
2. Adicionar validação visual usando imagens Sentinel-2 pré/pós-fogo
3. Publicar como serviço web com autenticação
4. Integrar alertas em tempo real via Telegram/WhatsApp
