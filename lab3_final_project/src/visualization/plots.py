"""Figures of the project."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import COLORS


def plot_group_counts_and_curves(usable, curve_data, groups):
    """Curves of each group."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    counts = usable["texture_group"].value_counts()[groups]
    axes[0].bar(counts.index, counts.values, color=[COLORS[g] for g in counts.index])
    axes[0].set_ylabel("Number of curves")
    axes[0].set_title("Curves per texture group")

    for layer_id, (psi, theta) in curve_data.items():
        group = usable.loc[layer_id, "texture_group"]
        axes[1].plot(psi, theta, color=COLORS[group], alpha=0.12)
    for group in groups:
        axes[1].plot([], [], color=COLORS[group], label=group)
    axes[1].set_xscale("symlog", linthresh=0.1)
    axes[1].set_xlabel("Matric potential psi (m)")
    axes[1].set_ylabel("Water content theta (m3/m3)")
    axes[1].set_title("All the curves")
    axes[1].legend()
    plt.tight_layout()
    return fig


def plot_feature_densities(dataset, domain_name, columns, groups):
    """Density of the features."""
    n_rows = (len(columns) + 2) // 3
    fig = plt.figure(figsize=(14, 3.6 * n_rows))
    for i, feature in enumerate(columns):
        plt.subplot(n_rows, 3, i + 1)
        sns.kdeplot(data=dataset, x=feature, hue="texture_group", hue_order=groups, palette=COLORS,
                    common_norm=False, fill=True, alpha=0.3)
        plt.title(feature)
    fig.suptitle("Density by texture group: " + domain_name)
    plt.tight_layout()
    return fig


def plot_confusion_and_roc(matrix, roc_curves, groups):
    """Confusion matrix and ROC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].imshow(matrix, cmap="Blues")
    axes[0].set_xticks(range(len(groups)))
    axes[0].set_xticklabels(groups)
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels(groups)
    axes[0].set_xlabel("Predicted group")
    axes[0].set_ylabel("True group")
    axes[0].set_title("Confusion matrix")
    for i in range(len(groups)):
        for j in range(len(groups)):
            axes[0].text(j, i, str(matrix[i, j]), ha="center", va="center")

    for group in groups:
        fpr, tpr, auc = roc_curves[group]
        axes[1].plot(fpr, tpr, color=COLORS[group], label=group + " (AUC " + str(round(auc, 3)) + ")")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[1].set_xlabel("False positive rate (1 - specificity)")
    axes[1].set_ylabel("True positive rate (sensitivity)")
    axes[1].set_title("ROC curves")
    axes[1].legend()
    plt.tight_layout()
    return fig


def plot_feature_set_comparison(summary, baseline):
    """Compare the sets of features."""
    means = summary.xs("mean", axis=1, level=1)[["accuracy", "macro F1"]]
    stds = summary.xs("std", axis=1, level=1)[["accuracy", "macro F1"]]
    positions = np.arange(len(means))
    width = 0.35

    fig = plt.figure(figsize=(9, 4.5))
    plt.bar(positions - width / 2, means["accuracy"], width, yerr=stds["accuracy"], label="accuracy")
    plt.bar(positions + width / 2, means["macro F1"], width, yerr=stds["macro F1"], label="macro F1")
    plt.axhline(baseline, color="black", linestyle="--", label="baseline")
    plt.xticks(positions, means.index, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("Test score")
    plt.title("Texture recognition by set of features")
    plt.legend()
    plt.tight_layout()
    return fig


def plot_fit_and_derivative(curve, model, derivative):
    """Curve, fit and slope."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    psi = np.logspace(-1, 3, 200)
    axes[0].plot(curve.psi, curve.theta, "ko", label="measured")
    axes[0].plot(psi, model(psi), "r", label="van Genuchten fit")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Matric potential psi (m)")
    axes[0].set_ylabel("theta (m3/m3)")
    axes[0].set_title("Curve " + str(curve.layer_id))
    axes[0].legend()

    log_psi = np.linspace(0.05, 1.95, 191)
    axes[1].plot(log_psi, derivative(log_psi))
    axes[1].set_xlabel("log10(psi)")
    axes[1].set_ylabel("d theta / d log10(psi)")
    axes[1].set_title("Slope of the curve")
    plt.tight_layout()
    return fig


def plot_covariance(cov, psi_grid):
    """Covariance plot."""
    fig = plt.figure(figsize=(5.5, 4.5))
    plt.imshow(cov, cmap="viridis")
    plt.colorbar()
    plt.xticks(range(len(psi_grid)), psi_grid)
    plt.yticks(range(len(psi_grid)), psi_grid)
    plt.xlabel("psi (m)")
    plt.ylabel("psi (m)")
    plt.title("Covariance of theta between potentials")
    plt.tight_layout()
    return fig
