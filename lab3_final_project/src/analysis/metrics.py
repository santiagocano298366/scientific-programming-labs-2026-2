"""Performance measures written by hand."""
import numpy as np
import pandas as pd


def se(tp, fn):
    """Sensitivity."""
    return tp / (tp + fn)


def sp(tn, fp):
    """Specificity."""
    return tn / (tn + fp)


def ppv(tp, fp):
    """Precision."""
    return tp / (tp + fp)


def f1(tp, fn, fp):
    """F1 score."""
    precision = ppv(tp, fp)
    recall = se(tp, fn)
    return 2 * precision * recall / (precision + recall)


def confusion_counts(y_true, y_pred, classes):
    """Confusion matrix."""
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    return matrix


def per_class_metrics(matrix, classes):
    """Metrics of each class."""
    total = matrix.sum()
    table = {"support": [], "recall (Se)": [], "specificity (Sp)": [], "precision (PPV)": [], "F1": []}
    for i in range(len(classes)):
        tp = matrix[i, i]
        fn = matrix[i].sum() - tp
        fp = matrix[:, i].sum() - tp
        tn = total - tp - fn - fp
        table["support"].append(tp + fn)
        table["recall (Se)"].append(se(tp, fn))
        table["specificity (Sp)"].append(sp(tn, fp))
        table["precision (PPV)"].append(ppv(tp, fp))
        table["F1"].append(f1(tp, fn, fp))
    return pd.DataFrame(table, index=classes)


def roc_points(scores, is_positive):
    """ROC curve, first version."""
    thresholds = np.concatenate(([np.inf], np.unique(scores)[::-1]))
    tpr = []
    fpr = []
    for threshold in thresholds:
        predicted = scores >= threshold
        tpr.append(np.sum(predicted & is_positive) / np.sum(is_positive))
        fpr.append(np.sum(predicted & ~is_positive) / np.sum(~is_positive))
    return np.array(fpr), np.array(tpr)


def auc_trapezoid(fpr, tpr):
    """Area under the ROC curve."""
    auc = 0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    return auc
