"""Runs the whole project."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import GROUPS, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR
from src.preprocessing.loading import load_raw
from src.preprocessing.cleaning import (clean_points, group_textures, build_curve_table,
                                        select_usable_curves, flag_upward_jumps)
from src.preprocessing.optimized import build_curve_data_groupby
from src.analysis.curves import SoilRetentionCurve, VanGenuchtenModel, CentralDiff
from src.analysis.features import (RetentionLevelFeatures, ShapeFeatures, VanGenuchtenFeatures,
                                   build_feature_table)
from src.analysis.group_stats import summary_by_group, find_outliers, test_across_groups, separability_table
from src.analysis.metrics import confusion_counts, per_class_metrics, auc_trapezoid
from src.analysis.optimized import roc_points_fast
from src.analysis.classification import split_by_site, NearestCentroidClassifier, repeated_evaluation
from src.analysis.matrix_ops import curve_matrix, covariance_matrix, explained_variance
from src.visualization import plots

for folder in [PROCESSED_DIR, TABLES_DIR, FIGURES_DIR]:
    os.makedirs(folder, exist_ok=True)


def save_table(table, name):
    """Save a table in results/tables."""
    table.to_csv(os.path.join(TABLES_DIR, name))


def save_figure(fig, name):
    """Save a figure in results/figures."""
    fig.savefig(os.path.join(FIGURES_DIR, name), dpi=120)
    plt.close(fig)


raw = load_raw()
points, steps = clean_points(raw)
labelled = group_textures(points)
curve_table = build_curve_table(labelled)
usable, selected_points = select_usable_curves(curve_table, labelled)
usable = flag_upward_jumps(usable, selected_points)
curve_data = build_curve_data_groupby(selected_points)

steps.to_csv(os.path.join(PROCESSED_DIR, "cleaning_steps.csv"), index=False)
usable.to_csv(os.path.join(PROCESSED_DIR, "usable_curves.csv"))
selected_points[["layer_id", "lab_head_m", "lab_wrc", "texture_group"]].to_csv(
    os.path.join(PROCESSED_DIR, "selected_points.csv"), index=False)
save_table(steps.set_index("step"), "cleaning_steps.csv")
print(steps)

extractors = [RetentionLevelFeatures(), ShapeFeatures(), VanGenuchtenFeatures()]
feature_table = build_feature_table(curve_data, extractors)
domains = {}
features = []
for extractor in extractors:
    domains[extractor.domain] = extractor.names
    features += extractor.names
dataset = feature_table.join(usable[["texture_group", "profile_id", "reference", "has_upward_jump"]])
dataset.to_csv(os.path.join(PROCESSED_DIR, "features.csv"))
print("curves x columns:", dataset.shape)

for name, table in summary_by_group(dataset, features, GROUPS).items():
    save_table(table, "summary_" + name + ".csv")
is_outlier = find_outliers(dataset, features, GROUPS)
save_table(pd.DataFrame({"outliers": is_outlier.sum(), "share (%)": 100 * is_outlier.mean()}), "outliers.csv")
tests, alpha_bonferroni = test_across_groups(dataset, features, GROUPS)
save_table(tests, "group_tests.csv")
separability = separability_table(dataset, features, GROUPS)
save_table(separability, "separability.csv")

test_mask = split_by_site(dataset, 0.3, 42)
train, test = dataset[~test_mask], dataset[test_mask]
model = NearestCentroidClassifier().fit(train[features].values, train["texture_group"].values, GROUPS)
y_test = test["texture_group"].values
y_pred = model.predict(test[features].values)
matrix = confusion_counts(y_test, y_pred, GROUPS)
class_table = per_class_metrics(matrix, GROUPS)
scores = model.scores(test[features].values)

roc_curves = {}
for i, group in enumerate(GROUPS):
    fpr, tpr = roc_points_fast(scores[:, i], y_test == group)
    roc_curves[group] = (fpr, tpr, auc_trapezoid(fpr, tpr))
accuracy = np.trace(matrix) / matrix.sum()
macro_auc = np.mean([roc_curves[g][2] for g in GROUPS])
print("accuracy:", round(accuracy, 3), "| macro F1:", round(class_table["F1"].mean(), 3), "| macro AUC:", round(macro_auc, 3))
save_table(pd.DataFrame(matrix, index=GROUPS, columns=GROUPS), "confusion_matrix.csv")
save_table(class_table, "per_class_metrics.csv")

feature_sets = dict(domains)
feature_sets["all features"] = features
repeat = repeated_evaluation(dataset, feature_sets, GROUPS)
save_table(repeat, "repeated_splits.csv")
summary = repeat.groupby("feature set")[["accuracy", "macro F1", "majority baseline"]].agg(["mean", "std"])
summary = summary.loc[list(feature_sets.keys())]
print(summary.round(3))

rng = np.random.RandomState(42)
many_points = [i for i in usable.index if usable.loc[i, "n_points"] >= 6]
chosen = rng.choice(many_points, size=40, replace=False)
psi_grid = [0.1, 0.5, 1, 3.3, 10, 33, 100, 150]
cov = covariance_matrix(curve_matrix(curve_data, chosen, psi_grid))
eigenvalues, shares = explained_variance(cov)
print("variance explained by the first eigenvalue:", round(shares[0], 3))
save_table(pd.DataFrame({"eigenvalue": eigenvalues, "share": shares}), "covariance_eigenvalues.csv")

save_figure(plots.plot_group_counts_and_curves(usable, curve_data, GROUPS), "curves_by_group.png")
for name, columns in domains.items():
    save_figure(plots.plot_feature_densities(dataset, name, columns, GROUPS), "density_" + name.replace(" ", "_") + ".png")
save_figure(plots.plot_confusion_and_roc(matrix, roc_curves, GROUPS), "confusion_and_roc.png")
save_figure(plots.plot_feature_set_comparison(summary, repeat["majority baseline"].mean()), "feature_set_comparison.png")
example_id = dataset[dataset["texture_group"] == "medium"].index[0]
example = SoilRetentionCurve(example_id, *curve_data[example_id])
fitted = VanGenuchtenModel().fit(example)
save_figure(plots.plot_fit_and_derivative(example, fitted, CentralDiff(example.theta_at_log, h=0.05)), "fit_and_derivative.png")
save_figure(plots.plot_covariance(cov, psi_grid), "covariance.png")
print("done")
