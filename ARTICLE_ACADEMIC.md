# Hybrid ML-DL Digital Twin for Wildfire Hotspot Detection in Ceara (Brazil)

## Abstract
This paper presents an open, reproducible digital twin for wildfire hotspot detection and short-horizon spatial forecasting in Ceara, Brazil. The system integrates multi-source hotspot ingestion (INPE, NASA FIRMS, KML fallback), temporal machine learning, deep temporal modeling, and cellular spread dynamics. We evaluate strict cell-level and operational metrics under temporal holdout validation. Results show that strict cell-level F1 remains challenging in sparse geospatial labeling, while operational performance is strong: daily fire-event F1 around 0.97 and spatially tolerant F1 above 0.80. We discuss architecture, calibration, model-selection strategy, and deployment implications for public-sector wildfire monitoring.

## 1. Introduction
Wildfire monitoring in semi-arid ecosystems requires both rapid detection and practical forecast support for response teams. In Ceara, seasonal fire patterns and sparse but high-impact hotspot activity create a difficult prediction setting: class imbalance is extreme, source reliability can vary, and strict geospatial matching penalizes near-miss predictions.

This work proposes a hybrid digital twin that combines data-driven risk estimation with explicit spatial dynamics. The objective is twofold:
1. Detect hotspots from real data in near operational settings.
2. Compare predictions against real observations with rigorous temporal validation.

## 2. Related Work
Operational wildfire intelligence typically combines satellite hotspot streams and meteorological risk indicators. Purely reactive dashboards often lack temporal generalization and explicit spread simulation. On the other hand, purely physical spread simulators can be difficult to calibrate in data-sparse contexts.

Hybrid approaches, where ML estimates ignition likelihood and a cellular layer enforces spatial dynamics, offer a practical compromise for regional deployments.

Recent methods still underexplore a joint strategy that combines thermal-background residualization, cross-sensor distillation (VIIRS to GOES), and online digital-twin feedback. This gap motivated the PYRO-Caatinga extension implemented as an MVP in this repository.

## 3. Study Area and Data
### 3.1 Area
Ceara, northeastern Brazil, represented by a fixed geographic bounding box.

### 3.2 Data Sources
- INPE/TerraBrasilis hotspots
- NASA FIRMS active fire data
- INPE daily KML fallback (GOES continuity)
- Local historical CSV archives for reproducible experiments

### 3.3 Data Engineering
- Coordinate normalization and quality filtering
- Daily temporal aggregation
- Deduplication by rounded lat/lon + timestamp + satellite
- Daily local persistence for incremental updates

## 4. Methodology
### 4.1 Spatial Grid and Targets
The state space is discretized into a lat/lon grid over Ceara. Each sample is a cell-day binary target (fire/no fire).

### 4.2 Features
- Multi-day lag stack
- Neighborhood-smoothed activity
- Temporal trend over lag window
- Seasonal cyclic encodings (day/month)
- Static hotspot prior
- Normalized grid position features

### 4.3 Hybrid Modeling
Candidate learners are benchmarked and constrained by precision/prevalence filters:
- HistGradientBoosting
- Random Forest
- Extra Trees
- Temporal MLP
- Logistic Regression

A hard-negative-mining refinement stage can be applied after initial fit.

### 4.4 Digital Twin Dynamics
Predicted risk maps are propagated through a cellular spread mechanism:
- neighborhood pressure
- ignition thresholding
- cooldown memory to reduce unrealistic immediate re-ignition

### 4.5 Calibration
Thresholds are selected on temporal validation folds (not on test), with configurable objective profiles:
- operational profile (event reliability)
- strict-cell profile (exact spatial match emphasis)

## 5. Evaluation Protocol
Temporal split:
- fit segment
- validation segment for model/threshold selection
- holdout test segment for final reporting

Metrics:
- strict cell-level: precision, recall, F1
- ranking: ROC-AUC, PR-AUC
- daily event detection F1
- spatially tolerant precision/recall/F1 (radius-based)
- twin IoU and municipality/season diagnostics

## 6. Results Summary
Across tested settings, strict cell-level F1 is significantly lower than operational metrics due to sparse positives and geospatial boundary effects. In contrast, operational metrics are strong:
- daily event F1 approximately 0.97
- tolerant spatial F1 above 0.80 in operational configurations

A strict-grid search improved strict cell F1 to a moderate range but did not reach 0.80, indicating structural limits in exact-cell labeling under real data noise.

In addition, an MVP extension called PYRO-Caatinga was implemented with four modules: climatology-residual front-end, VIIRS to GOES soft-label distillation, physical proxy heads, and twin-feedback pseudo-labeling. This MVP is designed as a transition layer toward full ABI/GLM-native deep training while preserving operational reproducibility.

## 7. Discussion
### 7.1 Why Strict Cell F1 Is Low
- Extreme class imbalance at cell-day level
- Spatial quantization and geolocation uncertainty
- Near-neighbor misses counted as full errors

### 7.2 Why Operational Metrics Are High
- Event-level and tolerant spatial metrics better reflect field utility
- Hybrid twin captures temporal persistence and neighborhood propagation

### 7.3 Practical Recommendation
Use a dual-reporting policy:
- strict metrics for research comparability
- operational metrics for decision support

## 8. Reproducibility
Typical run:

python main.py --local data/focos_CE_GOES16_2024.csv --ml-validate --ml-mode operational

Strict mode:

python main.py --local data/focos_CE_GOES16_2024.csv --ml-validate --ml-mode strict_cell

Main outputs:
- data/ml_twin_validation.json
- data/pipeline_result.json
- data/twin_state_latest.json

Experiment registry:

python -m src.run_experiments --dataset data/focos_CE_GOES16_2024.csv --output-dir data/experiments

PYRO-Caatinga MVP:

python main.py --pyro-mvp --pyro-goes-csv data/focos_CE_GOES16_2024.csv --pyro-output-dir data/pyro_caatinga

PYRO outputs:
- data/pyro_caatinga/pyro_goes_proxy_cube.nc
- data/pyro_caatinga/pyro_residual_cube.nc
- data/pyro_caatinga/pyro_viirs_soft_labels.npy
- data/pyro_caatinga/pyro_twin_pseudo_labels.npy
- data/pyro_caatinga/pyro_frp_proxy.npy
- data/pyro_caatinga/pyro_caatinga_report.json

## 9. Limitations
- Limited direct meteorological assimilation in the current runbook
- Endpoint volatility in public APIs requires fallback strategies
- Strict exact-cell objective may understate practical quality
- The current PYRO-Caatinga block uses a proxy GOES cube and simplified heads; a full causal transformer/Mamba stack over native ABI/GLM remains future work

## 10. Conclusion
The proposed hybrid digital twin demonstrates that open wildfire intelligence systems can achieve high operational reliability while maintaining transparent validation against real data. Although strict cell-level F1 remains difficult in sparse geospatial settings, the framework provides reproducible, decision-relevant performance and a strong base for future meteorology-aware and uncertainty-aware extensions.
