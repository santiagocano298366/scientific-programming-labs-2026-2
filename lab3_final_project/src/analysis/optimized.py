"""Optimized versions."""
import numpy as np

from src.analysis.curves import Trapezoidal
from src.analysis.features import RetentionLevelFeatures


def roc_points_fast(scores, is_positive):
    """ROC curve, fast version."""
    order = np.argsort(-scores)
    scores = scores[order]
    is_positive = is_positive[order]
    n_pos = is_positive.sum()
    n_neg = len(is_positive) - n_pos

    tp = 0
    fp = 0
    tpr = [0]
    fpr = [0]
    for i in range(len(scores)):
        if is_positive[i]:
            tp += 1
        else:
            fp += 1
        if i == len(scores) - 1 or scores[i + 1] != scores[i]:
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)
    return np.array(fpr), np.array(tpr)


def fisher_ratio_fast(values, labels, groups):
    """Fisher ratio, fast version."""
    stats = values.groupby(labels).agg(["count", "mean", "var"])
    stats = stats.loc[groups]
    n = stats["count"]
    variance = stats["var"] * (n - 1) / n
    overall_mean = values.mean()
    between = (n * (stats["mean"] - overall_mean) ** 2).sum()
    within = (n * variance).sum()
    return between / within


class FastRetentionLevelFeatures(RetentionLevelFeatures):
    """Same features, faster."""

    def __init__(self):
        """Build the integrator."""
        self.integrator = Trapezoidal(0, 2, 201)

    def extract(self, curve):
        """Water content and area."""
        values = list(curve.theta_at(self.psi_reference))
        area = self.integrator.integrate(curve.theta_at_log)
        return dict(zip(self.names, values + [area]))
