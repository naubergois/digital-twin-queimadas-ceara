# 🔥 Digital Twin para Detecção de Queimadas no Ceará

## Uma Plataforma Open-Source de Monitoramento, Simulação e Predição de Incêndios Florestais

---

**Autor:** Francisco Nauber Bernardo Gois  
**Repositório:** [github.com/naubergois/digital-twin-queimadas-ceara](https://github.com/naubergois/digital-twin-queimadas-ceara)  
**Licença:** MIT  
**Stack:** Python 3, Streamlit, Folium, Pandas, NumPy, NASA APIs, Open-Meteo

---

## Sumário

1. [Contexto e Motivação](#1-contexto-e-motivação)
2. [O Problema](#2-o-problema)
3. [Arquitetura da Solução](#3-arquitetura-da-solução)
4. [Módulo 1 — Coleta de Dados (fire_data.py)](#4-módulo-1--coleta-de-dados)
5. [Módulo 2 — Gêmeo Digital (digital_twin.py)](#5-módulo-2--gêmeo-digital)
6. [Módulo 3 — Análise Estatística (analysis.py)](#6-módulo-3--análise-estatística)
7. [Módulo 4 — Imagens de Satélite (satellite.py)](#7-módulo-4--imagens-de-satélite)
8. [Dashboard Interativo (app.py)](#8-dashboard-interativo)
9. [Pipeline Principal (main.py)](#9-pipeline-principal)
10. [Resultados e Validação](#10-resultados-e-validação)
11. [Como Executar](#11-como-executar)
12. [Próximos Passos](#12-próximos-passos)
13. [Referências](#13-referências)

---

## 1. Contexto e Motivação

O estado do Ceará, localizado no Nordeste brasileiro, possui aproximadamente **95% do seu território ocupado pelo bioma Caatinga** — uma das florestas tropicais secas mais biodiversas do mundo e também uma das mais ameaçadas. A cada ano, entre junho e dezembro (estação seca), o estado registra **milhares de focos de calor** detectados por satélites de monitoramento.

Dados do INPE (Instituto Nacional de Pesquisas Espaciais) mostram que o Ceará frequentemente figura entre os estados com maior número de queimadas no Nordeste, com picos históricos ultrapassando **15.000 focos anuais**. As consequências são severas:

- Perda de biodiversidade na Caatinga
- Emissão massiva de CO₂ e material particulado
- Problemas respiratórios na população
- Degradação do solo e desertificação
- Prejuízos econômicos na agricultura e pecuária

**A pergunta central deste projeto é:** como podemos usar tecnologia de gêmeos digitais, dados abertos de satélite e inteligência computacional para monitorar, simular e — quem sabe — antecipar a propagação do fogo no território cearense?

---

## 2. O Problema

Os sistemas atuais de monitoramento de queimadas no Brasil (BDQueimadas/INPE, NASA FIRMS) fornecem dados brutos de focos de calor, mas apresentam limitações significativas:

| Limitação | Impacto |
|---|---|
| Dados reativos (focos já detectados) | Sem capacidade preditiva |
| Visualização em mapas 2D estáticos | Difícil entender propagação |
| Sem modelagem de propagação | Não simula "para onde o fogo vai" |
| APIs fragmentadas (INPE, NASA, ESA) | Integração manual e trabalhosa |
| Sem análise de risco integrada | Decisão baseada apenas em intuição |

**Nossa solução** ataca esses quatro pontos simultaneamente através de um pipeline integrado que coleta dados de múltiplas fontes, constrói um gêmeo digital do território, simula a propagação do fogo e apresenta tudo em um dashboard interativo com imagens de satélite ao vivo.

---

## 3. Arquitetura da Solução

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DIGITAL TWIN                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Coleta      │    │   Análise    │    │  Simulação   │       │
│  │   de Dados    │───▶│  Estatística  │───▶│  Digital Twin│       │
│  │              │    │              │    │              │       │
│  │ • INPE API   │    │ • Sazonalidade│   │ • Autômato   │       │
│  │ • NASA FIRMS │    │ • Tendências  │    │   Celular 2D │       │
│  │ • NASA GIBS  │    │ • Top Munic.  │    │ • Propagação │       │
│  │ • Open-Meteo │    │ • Anomalias   │    │ • Zonas de   │       │
│  │ • Sentinel-2 │    │ • KDE Clusters│    │   Risco      │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                  │                   │               │
│         └──────────────────┴───────────────────┘               │
│                              │                                  │
│                     ┌────────▼────────┐                        │
│                     │   Dashboard      │                        │
│                     │   Streamlit      │                        │
│                     │                  │                        │
│                     │ • Mapas Folium   │                        │
│                     │ • Gráficos Matpl │                        │
│                     │ • Satélite ao    │                        │
│                     │   Vivo           │                        │
│                     │ • Auto-Refresh   │                        │
│                     └─────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### Fluxo de Dados

1. **Entrada**: APIs públicas (INPE, NASA, Open-Meteo) ou dados locais (CSV)
2. **Processamento**: Pandas DataFrames → análise temporal → modelo de autômato celular
3. **Saída**: Dashboard interativo + estado exportável do twin (JSON)
4. **Fallback**: Dados sintéticos realistas quando APIs estão indisponíveis

---

## 4. Módulo 1 — Coleta de Dados

**Arquivo:** `src/fire_data.py` (272 linhas)

### 4.1 INPE BDQueimadas API

O INPE mantém uma API REST pública que retorna focos de calor detectados pelos satélites de referência (AQUA, TERRA, NPP, NOAA-20).

**Endpoint principal:**
```
https://terrabrasilis.dpi.inpe.br/queimadas/api/focos
```

**Nossa implementação:**
```python
def fetch_inpe_fire_foci(state_code="23", date_from=None, date_to=None):
    """
    Busca focos de calor da API do INPE.

    Args:
        state_code: Código IBGE do estado (23 = Ceará)
        date_from: Data inicial (YYYY-MM-DD)
        date_to: Data final (YYYY-MM-DD)

    Returns:
        DataFrame com colunas: lat, lon, datetime, satellite, municipio, bioma
    """
```

**Características:**
- ✅ Sem necessidade de API key
- ✅ Dados de todos os satélites (AQUA, TERRA, NPP, NOAA-20)
- ✅ Resolução temporal horária
- ✅ Cobertura de todo o Brasil
- ⚠️ Atraso de ~3-6 horas para dados mais recentes

### 4.2 NASA FIRMS API (Opcional)

Fonte complementar com resolução mais fina (375m vs 1km):

```python
def fetch_firms_fires(api_key, source="VIIRS_SNPP_NRT", day_range=3, bbox=None):
    """
    Busca focos ativos da NASA FIRMS.

    Args:
        api_key: Token gratuito (https://firms.modaps.eosdis.nasa.gov)
        source: VIIRS_SNPP_NRT | MODIS_NRT | VIIRS_NOAA20_NRT
        day_range: 1-10 dias para trás
        bbox: (min_lon, min_lat, max_lon, max_lat)
    """
```

### 4.3 Download de Dados Históricos

```python
def download_year_data(year, output_dir="data"):
    """
    Baixa um ano completo de dados em blocos mensais.
    Usa a API do INPE com paginação automática.
    """
```

### 4.4 Fallback Sintético

Quando as APIs estão indisponíveis (manutenção, rede, limites de requisição), o sistema gera dados sintéticos **realistas** baseados na distribuição geográfica real dos focos no Ceará:

- **Regiões simuladas:** Sertão Central (35%), Cariri (15%), Norte (20%), Jaguaribe (15%), Ibiapaba (15%)
- **Sazonalidade:** Concentração na estação seca (jun-dez)
- **Satélites:** Distribuição realista dos detectores

---

## 5. Módulo 2 — Gêmeo Digital

**Arquivo:** `src/digital_twin.py` (324 linhas)

O coração do projeto. Implementa um **autômato celular 2D** que modela a paisagem do Ceará em uma grade regular e simula a propagação do fogo célula a célula.

### 5.1 Conceito

Um autômato celular é um modelo computacional onde:

- O espaço é discretizado em uma **grade regular de células**
- Cada célula possui um **estado** (vazio, vegetação, queimando, queimado)
- A evolução ocorre em **passos discretos**
- O estado futuro de cada célula depende do **estado atual dela e de suas vizinhas**

### 5.2 Classe FireDigitalTwin

```python
class FireDigitalTwin:
    def __init__(self, resolution=0.05):
        """
        resolution: tamanho da célula em graus decimais (~5.5km)
        """

    def initialize_from_history(self, df):
        """
        Constrói o grid de vegetação e a matriz de densidade
        histórica de queimadas a partir dos dados carregados.
        """

    def add_active_fires(self, df):
        """
        Marca células como "em chamas" com base nos focos ativos.
        """

    def simulate(self, steps=24):
        """
        Executa a simulação por N passos.
        Em cada passo:
          1. Células queimando propagam para vizinhas (regra de transição)
          2. Probabilidade depende de: densidade histórica + ruído
          3. Células queimam por tempo limitado e viram "queimadas"
        """
```

### 5.3 Regras de Propagação

A probabilidade de uma célula pegar fogo em cada passo é:

```
P(ignição) = α · D_histórica + β · N_vizinhas_queimando + γ · ε
```

Onde:
- **D_histórica**: Densidade histórica de queimadas na célula (0-1)
- **N_vizinhas_queimando**: Fração das 8 células vizinhas em chamas
- **ε**: Ruído aleatório (para modelar eventos estocásticos)
- **α, β, γ**: Pesos configuráveis (default: 0.3, 0.5, 0.2)

### 5.4 Zonas de Risco

```python
def get_fire_danger_zones(self, threshold=0.5):
    """
    Identifica clusters de células com alta densidade histórica.
    Usa agrupamento geográfico para encontrar regiões de risco.
    """
```

### 5.5 Áreas Críticas

```python
def check_critical_areas(self):
    """
    Monitora áreas pré-definidas de importância ecológica:
    - Chapada do Araripe (ALTO risco)
    - Serra de Baturité (MÉDIO risco)
    - Parque Nacional de Ubajara (MÉDIO risco)
    - Serra da Ibiapaba (MÉDIO risco)
    """
```

### 5.6 Exportação de Estado

```python
def export_state(self, path):
    """
    Exporta o estado completo do twin para JSON.
    Inclui: grid de vegetação, focos ativos, densidade histórica.
    """
```

---

## 6. Módulo 3 — Análise Estatística

**Arquivo:** `src/analysis.py` (284 linhas)

### 6.1 Classe FireAnalysis

```python
class FireAnalysis:
    def __init__(self, df):
        """Recebe um DataFrame com focos de calor."""
```

### 6.2 Distribuição Temporal

```python
def monthly_distribution(self):
    """Agrupa focos por mês para identificar padrão sazonal."""

def yearly_trend(self):
    """Calcula total de focos por ano para tendência de longo prazo."""

def peak_season(self):
    """
    Analisa sazonalidade: estações seca (jun-dez) vs chuvosa (jan-mai).
    Retorna mês de pico, percentual na seca, etc.
    """
```

### 6.3 Distribuição Geográfica

```python
def top_municipios(self, n=10):
    """Lista os N municípios com mais focos."""

def top_biomas(self, n=5):
    """Biomas mais afetados."""

def spatial_clusters(self, bandwidth=0.1):
    """
    Detecta clusters espaciais usando KDE (Kernel Density Estimation).
    Identifica regiões com densidade anormalmente alta de focos.
    """
```

### 6.4 Relatório Consolidado

```python
def summary_report(self):
    """
    Gera dicionário completo com:
    - Total de focos, período, média anual
    - Sazonalidade (mês de pico, % estação seca)
    - Top satélites detectores
    - Top biomas
    """
```

---

## 7. Módulo 4 — Imagens de Satélite

**Arquivo:** `src/satellite.py` (412 linhas)

### 7.1 NASA GIBS (Global Imagery Browse Services)

A NASA disponibiliza tiles de imagens de satélite processadas **gratuitamente** sem necessidade de autenticação:

| Camada | Resolução | Delay | Descrição |
|---|---|---|---|
| MODIS Terra True Color | 250m | ~1 dia | Imagem óptica visível |
| MODIS Aqua True Color | 250m | ~1 dia | Complemento do Terra |
| VIIRS True Color | 250m | <1 dia | Mais recente |
| MODIS Termal (Band 31) | 1km | ~1 dia | Temperatura de brilho |
| VIIRS Anomalias Térmicas | 375m | <1 dia | Focos de alta resolução |

**Implementação:**

```python
def gibs_tile_url(layer_key, date=None):
    """
    Gera URL de template para tiles NASA GIBS.
    Uso com Folium: folium.TileLayer(url_template)
    """
```

### 7.2 Tiles de Satélite Comerciais

```python
def satellite_layer_for_folium(layer_key):
    """
    Retorna (url, attribution, options) para camadas:
    - esri_satellite: ESRI World Imagery (máx zoom 19)
    - google_satellite: Google Satellite (máx zoom 20)
    - bing_satellite: Bing Satellite
    - gibs_*: Camadas NASA GIBS
    - osm: OpenStreetMap (fallback escuro)
    """
```

### 7.3 Índice de Perigo Meteorológico

Usa a API gratuita do **Open-Meteo** (sem API key):

```python
def fire_danger_index(weather_data):
    """
    Calcula índice composto (0-100) baseado em:
    - Temperatura máxima (peso 30)
    - Velocidade do vento (peso 20)
    - Umidade relativa baixa (peso 25)
    - Ausência de precipitação (peso 25)
    """
```

---

## 8. Dashboard Interativo

**Arquivo:** `dashboard/app.py` (731 linhas)

### 8.1 Stack

| Componente | Tecnologia | Função |
|---|---|---|
| Web framework | **Streamlit** | Interface interativa em Python puro |
| Mapas | **Folium** + **streamlit-folium** | Mapas Leaflet com overlays |
| Heatmaps | **Folium HeatMap** | Densidade de focos |
| Gráficos | **Matplotlib** | Séries temporais, barras |
| Dados | **Pandas** / **NumPy** | Processamento em memória |
| Imagens satélite | **NASA GIBS / ESRI** | Base do mapa |

### 8.2 Abas do Dashboard

#### 🛰️ **Satélite Ao Vivo** (Nova!)

A principal inovação desta versão. Uma tela de **streaming** que mostra:

- Imagem de satélite do Ceará em tempo real (ESRI, Google ou NASA GIBS)
- Focos de calor sobrepostos como círculos vermelhos
- Controle de **auto-refresh** (10-120 segundos) com contagem regressiva
- Seletor de camada de satélite (troque e veja instantaneamente)
- **Indicador de risco de fogo hoje** (temperatura, vento, umidade)
- **Linha do tempo animada** da evolução diária dos focos
- Previsão de risco para os próximos 3 dias (🔴🟡🟢)

#### 🗺️ **Mapa de Calor**

- Heatmap de densidade de focos com gradiente azul→amarelo→vermelho
- Filtros por satélite e bioma
- Marcadores das áreas críticas monitoradas
- Contorno do estado do Ceará

#### 📈 **Análise Temporal**

- Distribuição mensal com cores sazonais (seca vs chuvosa)
- Sazonalidade com indicadores numéricos
- Tendência anual com curva de evolução
- Interpretação automática dos padrões

#### 🤖 **Gêmeo Digital**

- Parâmetros ajustáveis (taxa de propagação, seca da vegetação)
- Simulação sob demanda
- Métricas de saída: células em chamas, queimadas, total afetado
- Gráfico de evolução temporal da simulação
- Zonas de alto risco detectadas
- Status das áreas críticas em tempo real

#### 📋 **Relatório**

- Estatísticas gerais (total focos, período, média)
- Sazonalidade detalhada
- Top municípios mais afetados
- Top biomas

### 8.3 Auto-Refresh

```python
# No final do dashboard
if auto_refresh:
    elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
    if elapsed >= refresh_interval:
        st.cache_data.clear()
        st.session_state.df = pd.DataFrame()
        st.rerun()
```

O sistema:
1. Limpa o cache de dados a cada ciclo
2. Recarrega dados frescos da API
3. Re-renderiza o mapa e os indicadores
4. Mantém o estado da sessão entre refreshes

---

## 9. Pipeline Principal

**Arquivo:** `main.py` (172 linhas)

O pipeline orquestra os 4 módulos em sequência:

```bash
python main.py                    # Pipeline completo (demo)
python main.py --year 2024        # Ano específico
python main.py --local data.csv   # Dados locais
python main.py --dashboard        # Só o dashboard
python main.py --api firms        # Usar NASA FIRMS
```

### Etapas:

1. **Coleta**: Busca dados da fonte selecionada (API INPE, FIRMS, CSV ou sintético)
2. **Análise**: Estatísticas descritivas, sazonalidade, clusters
3. **Simulação**: Inicializa o twin, executa N passos, coleta histórico
4. **Exportação**: Salva resultados em JSON (`data/pipeline_result.json`)
5. **Dashboard**: Abre o Streamlit (modo `--dashboard`)

---

## 10. Resultados e Validação

### 10.1 Validação do Modelo de Propagação

O autômato celular foi calibrado usando dados históricos do INPE:

| Métrica | Valor | Nota |
|---|---|---|
| Cobertura geográfica | 100% do Ceará | Grid de 0.05° (~5.5km) |
| Resolução temporal | Passos de ~1h | 24 passos = ~1 dia simulado |
| Detecção de clusters | KDE bandwidth=0.1 | Agrupamento automático |
| Áreas críticas | 4 monitoradas | Araripe, Baturité, Ubajara, Ibiapaba |
| Fallback | Sintético realista | Distribuição geográfica real |

### 10.2 Sazonalidade Observada

Os padrões históricos do Ceará mostram:

- **Estação seca (Jun-Dez)**: 70-85% dos focos anuais
- **Mês de pico**: Setembro-Outubro (pico da seca)
- **Regiões mais afetadas**: Sertão Central, Norte, Ibiapaba
- **Bioma mais atingido**: Caatinga (>90% dos focos)

### 10.3 Limitações Conhecidas

- **Resolução**: Grid de 5.5km não captura micro-topografia
- **Vento**: Não modelado diretamente (via fator estocástico)
- **Tipo de vegetação**: Modelo simplificado (combustível homogêneo)
- **APIs**: Dependem de disponibilidade externa
- **Validação**: Comparação quantitativa com dados reais é trabalho futuro

---

## 11. Como Executar

### Pré-requisitos

```bash
# Python 3.10+
python3 --version

# Instalar dependências
pip install -r requirements.txt
```

### Pipeline Completo

```bash
cd /Volumes/NAUBER/digital-twin-queimadas-ceara

# Modo demonstração (dados sintéticos)
python main.py

# Com ano específico (requer internet)
python main.py --year 2024

# Com dados do NASA FIRMS (requer token grátis)
export FIRMS_API_KEY="seu_token_aqui"
python main.py --api firms

# Apenas abrir o dashboard
streamlit run dashboard/app.py
```

### Opções da Linha de Comando

| Flag | Descrição |
|---|---|
| `--year 2024` | Baixa dados do ano completo |
| `--local data.csv` | Carrega CSV local |
| `--dashboard` | Abre o Streamlit diretamente |
| `--api inpe|firms` | Fonte de dados |

### Variáveis de Ambiente

| Variável | Obrigatória? | Descrição |
|---|---|---|
| `FIRMS_API_KEY` | Não | Token NASA FIRMS (gratuito) |
| `SH_CLIENT_ID` | Não | Sentinel Hub OAuth |
| `SH_CLIENT_SECRET` | Não | Sentinel Hub OAuth |

---

## 12. Próximos Passos

### Curto Prazo

- [ ] Integrar **Google Earth Engine** para dados históricos de satélite
- [ ] Adicionar **camadas de vento** no modelo de propagação (WRF-SFIRE)
- [ ] Melhorar a validação quantitativa contra dados reais

### Médio Prazo

- [ ] Implementar **modelo de combustível** baseado em NDVI (índice de vegetação)
- [ ] Adicionar **previsão climática** (GFS/ECMWF) ao índice de perigo
- [ ] Dashboard com **notificações push** para áreas de alto risco

### Longo Prazo

- [ ] **Graph Neural Networks** para modelagem de propagação (PyTorch Geometric)
- [ ] **Reinforcement Learning** para alocação otimizada de recursos de combate
- [ ] Integração com **sistemas de alerta** da Defesa Civil
- [ ] **Mobile app** para brigadistas com mapa offline

---

## 13. Referências

### Dados

- **INPE BDQueimadas**: [terrabrasilis.dpi.inpe.br/queimadas](https://terrabrasilis.dpi.inpe.br/queimadas)
- **NASA FIRMS**: [firms.modaps.eosdis.nasa.gov](https://firms.modaps.eosdis.nasa.gov)
- **NASA GIBS**: [gibs.earthdata.nasa.gov](https://gibs.earthdata.nasa.gov)
- **Open-Meteo**: [open-meteo.com](https://open-meteo.com)
- **Sentinel Hub**: [sentinel-hub.com](https://www.sentinel-hub.com)

### Tecnologias

- **Streamlit**: [streamlit.io](https://streamlit.io)
- **Folium**: [python-visualization.github.io/folium](https://python-visualization.github.io/folium)
- **Leaflet**: [leafletjs.com](https://leafletjs.com)
- **Pandas**: [pandas.pydata.org](https://pandas.pydata.org)
- **Open-Meteo API**: [open-meteo.com/en/docs](https://open-meteo.com/en/docs)

### Leitura Complementar

- **Cellular Automata for Wildfire Spread**: [doi.org/10.1016/j.envsoft.2022.105482](https://doi.org/10.1016/j.envsoft.2022.105482)
- **WIFIRE Project** (UC San Diego): [wifire.ucsd.edu](https://wifire.ucsd.edu)
- **FARSITE Fire Growth Model**: [firelab.org](https://firelab.org)

---

## Licença

Este projeto é distribuído sob licença MIT. Sinta-se livre para usar, modificar e contribuir.

---

*"O fogo não conhece fronteiras — mas com dados abertos e gêmeos digitais, podemos aprender a antecipá-lo."*

---

⌨️ **Código fonte:** [github.com/naubergois/digital-twin-queimadas-ceara](https://github.com/naubergois/digital-twin-queimadas-ceara)  
📧 **Contato:** [naubergois@gmail.com](mailto:naubergois@gmail.com)
