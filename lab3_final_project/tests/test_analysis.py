"""Simple tests."""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.analysis.curves import SoilRetentionCurve, Trapezoidal, van_genuchten
from src.analysis.metrics import roc_points, auc_trapezoid, confusion_counts
from src.analysis.optimized import roc_points_fast, fisher_ratio_fast
from src.analysis.group_stats import fisher_ratio
from src.analysis.matrix_ops import covariance_matrix, covariance_loop


def test_trapezoidal():
    """The integral of x between 0 and 2 is 2."""
    value = Trapezoidal(0, 2, 201).integrate(lambda x: x)
    assert abs(value - 2) < 1e-9


def test_curve_is_sorted():
    """The curve sorts psi and theta together."""
    curve = SoilRetentionCurve("a", [10, 1, 100], [0.2, 0.4, 0.1])
    assert list(curve.psi) == [1, 10, 100]
    assert list(curve.theta) == [0.4, 0.2, 0.1]


def test_van_genuchten():
    """Limits of the model."""
    assert abs(van_genuchten(0.0, 1.0, 2.0, 0.1, 0.5) - 0.5) < 1e-9
    assert abs(van_genuchten(1e8, 1.0, 2.0, 0.1, 0.5) - 0.1) < 1e-3


def test_confusion_counts():
    """A small example of the confusion matrix."""
    y_true = np.array(["a", "a", "b", "b"])
    y_pred = np.array(["a", "b", "b", "b"])
    matrix = confusion_counts(y_true, y_pred, ["a", "b"])
    assert matrix.tolist() == [[1, 1], [0, 2]]


def test_optimized_roc():
    """Same ROC as the first version."""
    rng = np.random.RandomState(0)
    scores = np.round(rng.normal(size=300), 1)
    is_positive = rng.rand(300) < 0.4
    fpr, tpr = roc_points(scores, is_positive)
    fpr_fast, tpr_fast = roc_points_fast(scores, is_positive)
    assert np.allclose(fpr, fpr_fast) and np.allclose(tpr, tpr_fast)
    assert abs(auc_trapezoid(fpr, tpr) - auc_trapezoid(fpr_fast, tpr_fast)) < 1e-12


def test_optimized_fisher():
    """Same Fisher ratio as the first version."""
    rng = np.random.RandomState(1)
    groups = ["a", "b", "c"]
    labels = pd.Series(rng.choice(groups, 200))
    values = pd.Series(rng.normal(size=200))
    assert abs(fisher_ratio(values, labels, groups) - fisher_ratio_fast(values, labels, groups)) < 1e-9


def test_covariance():
    """Same covariance with both methods."""
    matrix = np.random.RandomState(2).normal(size=(20, 4))
    assert np.allclose(covariance_matrix(matrix), covariance_loop(matrix))


test_trapezoidal()
test_curve_is_sorted()
test_van_genuchten()
test_confusion_counts()
test_optimized_roc()
test_optimized_fisher()
test_covariance()
print("all tests passed")
