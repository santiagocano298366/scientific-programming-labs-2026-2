"""Split by site, nearest centroid classifier and evaluation."""
import numpy as np
import pandas as pd

from src.analysis.metrics import confusion_counts, per_class_metrics


def split_by_site(table, test_fraction, seed):
    """Mask of the test curves, split by site."""
    rng = np.random.RandomState(seed)
    sites = table["profile_id"].unique()
    mixed = rng.permutation(sites)
    test_sites = mixed[:int(round(test_fraction * len(sites)))]
    return table["profile_id"].isin(test_sites).values


class NearestCentroidClassifier:
    """Nearest centroid classifier."""

    def fit(self, X, y, classes):
        """Compute the centroids."""
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        Z = (X - self.mean) / self.std
        self.classes = np.array(classes)
        self.centroids = np.array([Z[y == c].mean(axis=0) for c in self.classes])
        return self

    def scores(self, X):
        """Score of each class."""
        Z = (X - self.mean) / self.std
        distances = np.zeros((len(Z), len(self.centroids)))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.sqrt(((Z - centroid) ** 2).sum(axis=1))
        return -distances

    def predict(self, X):
        """Class of the closest centroid."""
        return self.classes[np.argmax(self.scores(X), axis=1)]


def evaluate_features(dataset, columns, test_mask, groups):
    """Train, test and return the metrics."""
    train = dataset[~test_mask]
    test = dataset[test_mask]
    model = NearestCentroidClassifier().fit(train[columns].values, train["texture_group"].values, groups)
    y_true = test["texture_group"].values
    y_pred = model.predict(test[columns].values)

    matrix = confusion_counts(y_true, y_pred, groups)
    table = per_class_metrics(matrix, groups)
    accuracy = np.trace(matrix) / matrix.sum()
    baseline = max(np.mean(y_true == c) for c in groups)
    return accuracy, table["F1"].mean(), baseline


def repeated_evaluation(dataset, feature_sets, groups, n_splits=30):
    """Repeat the evaluation with different splits."""
    results = {"feature set": [], "seed": [], "accuracy": [], "macro F1": [], "majority baseline": []}
    for name, columns in feature_sets.items():
        for seed in range(n_splits):
            mask = split_by_site(dataset, 0.3, seed)
            accuracy, macro_f1, baseline = evaluate_features(dataset, columns, mask, groups)
            results["feature set"].append(name)
            results["seed"].append(seed)
            results["accuracy"].append(accuracy)
            results["macro F1"].append(macro_f1)
            results["majority baseline"].append(baseline)
    return pd.DataFrame(results)
