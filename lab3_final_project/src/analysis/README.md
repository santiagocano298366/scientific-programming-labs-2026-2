# analysis

| File | Content |
|---|---|
| `curves.py` | `SoilRetentionCurve`, `Integrator`, `Trapezoidal`, `Derivative`, `ForwardDiff`, `CentralDiff`, `van_genuchten`, `VanGenuchtenModel`. |
| `features.py` | `FeatureExtractor` and its three child classes, `build_feature_table`. |
| `group_stats.py` | Summary by group, outliers, group tests, Fisher ratio and AUC. |
| `metrics.py` | `se`, `sp`, `ppv`, `f1`, `confusion_counts`, `per_class_metrics`, `roc_points`, `auc_trapezoid`. |
| `classification.py` | `split_by_site`, `NearestCentroidClassifier`, `evaluate_features`, `repeated_evaluation`. |
| `matrix_ops.py` | Covariance matrix of the curves (matrix product and loops) and eigenvalues. |
| `optimized.py` | `roc_points_fast`, `fisher_ratio_fast`, `FastRetentionLevelFeatures`. |
| `complexity_analysis.ipynb` | Complexity analysis, optimization and times. |

The optimized functions give the same result as the first versions (see `tests/test_analysis.py`).
