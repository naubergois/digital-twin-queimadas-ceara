# Experiment Comparison

## Best Results

- Best strict F1: **strict_no_hnm** (F1=0.2512, model=mlp_temporal)
- Best operational score: **operational_day_focus** (score=0.6873, F1_tol=0.4549, F1_day=0.9714)

## Ranking by Strict F1

1. strict_no_hnm - F1=0.2512, P=0.2917, R=0.2206, model=mlp_temporal
2. strict_cost_sensitive - F1=0.0698, P=0.0432, R=0.1816, model=mlp_temporal
3. strict_tolerant_optimization - F1=0.0685, P=0.0356, R=0.9296, model=mlp_temporal
4. strict_no_deep - F1=0.0685, P=0.0356, R=0.9296, model=mlp_temporal
5. operational_day_focus - F1=0.0685, P=0.0356, R=0.9296, model=mlp_temporal
6. operational_tolerant_focus - F1=0.0685, P=0.0356, R=0.9296, model=mlp_temporal
7. strict_long_memory - F1=0.0676, P=0.0350, R=1.0000, model=mlp_temporal
8. strict_high_positive_weight - F1=0.0671, P=0.0347, R=1.0000, model=mlp_temporal
9. strict_spatial_richer - F1=0.0538, P=0.0276, R=0.9823, model=mlp_temporal
10. baseline_strict_legacy - F1=0.0434, P=0.0227, R=0.4708, model=gradient_boosting

## Ranking by Operational Score

1. operational_day_focus - Score=0.6873, F1_tol=0.4549, F1_day=0.9714
2. operational_tolerant_focus - Score=0.6873, F1_tol=0.4549, F1_day=0.9714
3. strict_tolerant_optimization - Score=0.5878, F1_tol=0.2739, F1_day=0.9714
4. strict_no_hnm - Score=0.4951, F1_tol=0.2512, F1_day=0.7931
5. strict_cost_sensitive - Score=0.4757, F1_tol=0.0698, F1_day=0.9718
6. strict_no_deep - Score=0.4748, F1_tol=0.0685, F1_day=0.9714
7. strict_long_memory - Score=0.4743, F1_tol=0.0676, F1_day=0.9714
8. strict_high_positive_weight - Score=0.4742, F1_tol=0.0671, F1_day=0.9718
9. strict_spatial_richer - Score=0.4667, F1_tol=0.0538, F1_day=0.9714
10. baseline_strict_legacy - Score=0.4612, F1_tol=0.0434, F1_day=0.9718

## Notes on Tested Techniques

- Cost-sensitive learning: aumenta o peso de classes positivas em cenarios desbalanceados.
- Recency weighting: prioriza amostras recentes para adaptacao temporal.
- Hard-negative mining: foca negativos dificeis para reduzir falsos positivos.
- Soft voting ensemble: combina arvores e modelo linear para robustez.
- Metric-aware optimization: alterna objetivo entre F1 estrito, F1 tolerante e F1 diario.
