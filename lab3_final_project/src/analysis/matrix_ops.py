"""Covariance of the curves."""
import numpy as np
import scipy.linalg

from src.analysis.curves import SoilRetentionCurve


def curve_matrix(curve_data, layer_ids, psi_grid):
    """Matrix of the curves."""
    rows = []
    for layer_id in layer_ids:
        psi, theta = curve_data[layer_id]
        curve = SoilRetentionCurve(layer_id, psi, theta)
        rows.append(curve.theta_at(psi_grid))
    return np.array(rows)


def covariance_matrix(matrix):
    """Covariance with a matrix product."""
    centered = matrix - matrix.mean(axis=0)
    return centered.T @ centered / (len(matrix) - 1)


def covariance_loop(matrix):
    """Covariance with loops."""
    n_rows, n_cols = matrix.shape
    means = [matrix[:, j].mean() for j in range(n_cols)]
    cov = np.zeros((n_cols, n_cols))
    for a in range(n_cols):
        for b in range(n_cols):
            total = 0
            for i in range(n_rows):
                total += (matrix[i, a] - means[a]) * (matrix[i, b] - means[b])
            cov[a, b] = total / (n_rows - 1)
    return cov


def explained_variance(cov):
    """Eigenvalues and explained variance."""
    eigenvalues, _ = scipy.linalg.eig(cov)
    eigenvalues = np.sort(eigenvalues.real)[::-1]
    return eigenvalues, eigenvalues / eigenvalues.sum()
