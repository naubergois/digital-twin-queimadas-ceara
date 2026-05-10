# Hybrid Digital Twin for Wildfire Detection and Forecasting in Ceara, Brazil

## A Data-Driven, Multi-Source, Open Implementation with Real-World Validation

Author: Francisco Nauber Bernardo Gois  
Repository: https://github.com/naubergois/digital-twin-queimadas-ceara  
License: MIT

## Abstract
Wildfire monitoring systems are often reactive: they detect hotspots after ignition but provide limited forward simulation and weak operational explainability. This work presents an open hybrid digital twin for Ceara (Brazil), integrating official hotspot data, satellite layers, machine learning, deep temporal modeling, and cellular spatial dynamics.

The system combines:
- Real hotspot ingestion (INPE, NASA FIRMS, KML/CSV fallback)
- A physics-inspired cellular digital twin for fire spread
- A hybrid ML+DL risk model (tree models + temporal neural model)
- Automatic threshold calibration
- Multi-metric validation against real observations

A key result is that strict cell-level F1 remains limited under strong spatial sparsity and geolocation uncertainty, while operational metrics become robust:
- Daily event detection F1 around 0.97
- Spatially tolerant F1 above 0.80 (radius-based tolerance)

This paper details the architecture, modeling decisions, experiment tracking, and practical trade-offs for real deployments.

## 1. Problem Statement
Ceara has recurrent wildfire activity, especially in dry-season months, with severe ecological and health impacts. Public APIs provide hotspot points, but incident response needs:
- near real-time ingestion,
- risk forecasting,
- spatial spread simulation,
- interpretable validation versus real observations.

The main research question is:
How far can an open hybrid digital twin improve real wildfire hotspot detection and forecast quality in operational conditions?

## 2. Data Sources and Ingestion
The pipeline merges multiple open sources.

### 2.1 Primary Sources
- INPE/TerraBrasilis wildfire hotspots
- NASA FIRMS active fire points
- INPE daily KML fallback for GOES family when endpoints change
- Local historical CSV snapshots for reproducible experiments

### 2.2 GOES Continuity Strategy
Operational endpoints may change over time and satellite references can migrate (for example, GOES-16 operational replacement contexts). The ingestion layer therefore includes:
- multi-endpoint retrieval,
- naming normalization for satellite tags,
- robust local filtering,
- KML fallback parsing.

### 2.3 Daily Update Capability
A daily updater maintains an incremental local wildfire database:
- append new records,
- deduplicate by lat/lon/time/satellite key,
- preserve historical continuity for model retraining.

## 3. System Architecture
The solution is modular and reproducible.

- src/fire_data.py
  - source connectors
  - normalization and fallback
  - daily update persistence
- src/analysis.py
  - temporal and seasonal statistics
- src/digital_twin.py
  - cellular spread simulation
- src/ml_digital_twin.py
  - hybrid ML+DL model
  - benchmark and calibration
  - real-data comparison metrics
- dashboard/app.py
  - online detection interface
  - satellite overlays
  - digital twin simulation and reporting
- main.py
  - CLI orchestration and export of consolidated outputs

## 4. Hybrid Digital Twin Method

### 4.1 Spatial Grid Representation
The state is discretized into a latitude-longitude grid over Ceara. Each cell stores:
- historical hotspot density,
- dynamic ignition state,
- spread/cooldown state,
- model-estimated risk.

### 4.2 Features for Learning
For each cell-day sample, the model builds temporal-spatial predictors:
- lag stack (multi-day hotspot history)
- neighborhood-smoothed activity
- trend between oldest and newest lag
- seasonal encodings (day-of-year and month sin/cos)
- static hotspot prior
- normalized grid coordinates

### 4.3 Candidate Models
The framework benchmarks multiple algorithms and chooses one automatically with constraints:
- HistGradientBoosting
- RandomForest
- ExtraTrees
- Temporal MLP
- Logistic Regression

Model selection is constrained by:
- minimum precision
- maximum positive prediction rate
- target optimization metric (strict F1 or operational profile)

### 4.4 Hard Negative Mining
After initial fitting, the training stage can refine on hard negatives:
- keep positives,
- rank negatives by predicted probability,
- refit on hardest confusing negatives.

This reduces trivial overprediction and improves practical precision in sparse settings.

### 4.5 Digital Twin Spread Layer
Predicted risk maps are coupled with a cellular spread process:
- neighborhood pressure from active cells,
- ignition threshold,
- cooldown logic to avoid unrealistic immediate re-ignition.

This layer yields a dynamic fire-front estimate rather than static point classification.

## 5. Evaluation Strategy
Validation is temporal and leakage-safe:
- train-fit split,
- validation split for threshold calibration,
- final holdout test period.

Metrics include:
- strict cell metrics: precision, recall, F1
- ranking quality: ROC-AUC, PR-AUC
- daily event metrics: F1 over day-has-fire vs day-no-fire
- tolerant spatial metrics: F1 with neighborhood tolerance radius
- twin spread quality: spatial IoU and detection consistency
- municipality-level error analysis
- dry vs wet season breakdown

## 6. Why Strict Cell F1 Is Hard
Strict cell F1 is structurally constrained by:
- extreme class imbalance (few positive cells per day),
- geolocation uncertainty and grid discretization,
- timing offsets between source detection and model state transitions,
- near-neighbor miss penalty (adjacent cell counts as full error).

Therefore, operationally useful models can show modest strict cell F1 while achieving high daily detection and high tolerant spatial F1.

## 7. Experimental Results

### 7.1 Strong Strict-Cell Search
A direct strict-cell search over grid resolution and temporal lookback was executed.
Best observed strict-cell setting in that search:
- grid resolution: 0.5
- lookback: 5
- ML strict F1: 0.435
- precision: 0.402
- recall: 0.474

This is substantially better than many strict-cell baselines, but still below 0.8.

### 7.2 Operationally Tuned Hybrid Twin
With operational calibration and spatial tolerance radius:
- ML strict cell F1: 0.165
- ML daily-event F1: 0.972
- ML tolerant F1: 0.818
- Twin tolerant F1: 0.805
- Twin daily-event F1: 0.972
- Day-level detection rate (real-fire days): 100%

### 7.3 Interpretation
- If the target is strict exact-cell hit only, performance is limited by label geometry and sparsity.
- If the target is operational hotspot awareness (right place neighborhood + right day), performance is robust and deployment-ready.

## 8. Internet-Guided Technique Exploration
Beyond initial implementation, the project expanded with methods commonly recommended in modern forecasting stacks:
- model ensembling (tree + neural temporal model)
- hard negative mining for sparse positives
- constrained model selection (precision and prevalence caps)
- threshold calibration by temporal validation
- tolerance-aware spatial metrics for geospatial uncertainty

These changes produced measurable improvements in practical wildfire detection quality.

## 9. PYRO-Caatinga Extension (Implemented MVP)
To address semi-arid thermal background bias in Ceara, the repository now includes a first implementation of the proposed PYRO-Caatinga workflow:

- Climatology-residual front-end with online EWMA per pixel/day-of-year/hour
- Cross-distillation VIIRS to GOES soft-label generation on the GOES grid
- Physical heads proxy outputs (fire mask probability and FRP-like proxy)
- Digital twin feedback pseudo-labeling in $t+5$ with uncertainty gating

Current implementation scope (MVP):
- file: `src/pyro_caatinga.py`
- CLI runner: `src/run_pyro_caatinga.py`
- output package under `data/pyro_caatinga/`

The MVP already produces reproducible artifacts for residual cubes, soft labels, pseudo-labels, and summary metrics, enabling iterative migration to full ABI/GLM-native training.

## 10. Practical Deployment Guidance

### 9.1 Recommended Modes
- Operational mode
  - objective: high incident awareness and low missed days
  - monitor: daily F1 and tolerant F1
- Strict research mode
  - objective: exact-cell comparison studies
  - monitor: strict cell F1 and confusion trends

### 9.2 Daily Operations
- Run daily update ingestion.
- Recompute validation snapshots periodically.
- Watch for endpoint changes in public providers.

### 9.3 Monitoring Checklist
- data freshness,
- source health (INPE/FIRMS/KML fallback),
- threshold drift,
- municipality-level residual errors,
- dry-season behavior shifts.

## 11. Limitations
- No direct wind field assimilation yet
- No explicit fuel moisture field beyond proxy dynamics
- Cell-level strict target remains sensitive to geolocation uncertainty
- Some source APIs may change or be intermittently unavailable

## 12. Future Work
- Integrate meteorological reanalysis features (wind, humidity, precipitation anomalies)
- Add uncertainty quantification per prediction map
- Couple spread with terrain slope and land-cover friction
- Evaluate ConvLSTM or transformer variants on compact tensors
- Formalize experiment registry and article-ready reproducibility package

## 13. Reproducibility
Typical command for full pipeline validation on historical real data:

python main.py --local data/focos_CE_GOES16_2024.csv --ml-validate --ml-mode operational

Strict mode for cell-level study:

python main.py --local data/focos_CE_GOES16_2024.csv --ml-validate --ml-mode strict_cell

Outputs:
- data/pipeline_result.json
- data/ml_twin_validation.json
- data/twin_state_latest.json

Experiment registry run:

python -m src.run_experiments --dataset data/focos_CE_GOES16_2024.csv --output-dir data/experiments

PYRO-Caatinga MVP run:

python main.py --pyro-mvp --pyro-goes-csv data/focos_CE_GOES16_2024.csv --pyro-output-dir data/pyro_caatinga

## 14. Conclusion
This project demonstrates that an open hybrid digital twin can move wildfire monitoring from reactive point plotting toward validated, forecast-aware operations. While strict cell-level F1 remains inherently challenging in sparse geospatial detection tasks, the system achieves strong operational behavior through daily-event reliability and tolerant spatial agreement, making it suitable for real monitoring workflows in Ceara.
