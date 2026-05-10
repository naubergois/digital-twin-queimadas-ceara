# Experiment Protocol

## Objective
Improve wildfire focus detection metrics in Ceara and compare multiple techniques under a reproducible setup.

## Dataset
- Input file: data/focos_CE_GOES16_2024.csv
- Expected columns: datetime, lat, lon (optional municipio)

## Procedure
1. Keep a fixed temporal split (train/validation/test) inside the same validation pipeline.
2. Run all experiments with deterministic seeds.
3. Save one JSON per run and aggregate summary in CSV and JSON.
4. Compare two views:
   - Strict cell-level quality (ml_f1)
   - Operational quality (weighted combination of ml_f1_tolerant and ml_f1_day)

## Tested Technique Families
- Cost-sensitive learning for class imbalance.
- Recency-aware sample weighting.
- Hard-negative mining.
- Ensemble model selection with multiple candidate classifiers.
- Metric-aware optimization (strict and tolerant criteria).

## Output Artifacts
- data/experiments/all_experiments_summary.csv
- data/experiments/all_experiments_full.json
- data/experiments/runs/*.json
- EXPERIMENTS.md

## Re-run Command
python -m src.run_experiments --dataset data/focos_CE_GOES16_2024.csv
