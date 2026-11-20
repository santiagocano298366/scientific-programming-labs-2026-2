"""Features of the curves."""
import numpy as np
import pandas as pd

from src.analysis.curves import SoilRetentionCurve, Trapezoidal, CentralDiff, VanGenuchtenModel


class FeatureExtractor:
    """Base class of the extractors."""
    domain = "base"
    names = []

    def extract(self, curve):
        """Features of one curve."""
        pass


class RetentionLevelFeatures(FeatureExtractor):
    """Water content and area features."""
    domain = "retention level"
    psi_reference = [1, 3.3, 10, 33, 100]
    names = ["theta_1m", "theta_3p3m", "theta_10m", "theta_33m", "theta_100m", "area_theta_logpsi"]

    def extract(self, curve):
        """Water content and area."""
        values = list(curve.theta_at(self.psi_reference))
        area = Trapezoidal(0, 2, 201).integrate(curve.theta_at_log)
        return dict(zip(self.names, values + [area]))


class ShapeFeatures(FeatureExtractor):
    """Slope of the curve."""
    domain = "curve shape"
    names = ["max_abs_slope", "log_psi_at_max_slope", "mean_abs_slope_dry"]

    def extract(self, curve):
        """Slope features."""
        slope = CentralDiff(curve.theta_at_log, h=0.05)
        grid = np.linspace(0.05, 1.95, 191)
        steep = np.abs(slope(grid))
        return {"max_abs_slope": steep.max(),
                "log_psi_at_max_slope": grid[steep.argmax()],
                "mean_abs_slope_dry": steep[grid >= 1].mean()}


class VanGenuchtenFeatures(FeatureExtractor):
    """Parameters of the van Genuchten model."""
    domain = "van Genuchten parameters"
    names = ["log10_alpha", "n", "theta_r", "theta_s"]

    def extract(self, curve):
        """Fit parameters and error."""
        model = VanGenuchtenModel().fit(curve)
        alpha, n, theta_r, theta_s = model.params
        return {"log10_alpha": np.log10(alpha), "n": n, "theta_r": theta_r, "theta_s": theta_s,
                "vg_fit_rmse": model.rmse(curve)}


def build_feature_table(curve_data, extractors):
    """Feature table of all the curves."""
    table = {}
    for layer_id, (psi, theta) in curve_data.items():
        curve = SoilRetentionCurve(layer_id, psi, theta)
        values = {}
        for extractor in extractors:
            values.update(extractor.extract(curve))
        table[layer_id] = values
    feature_table = pd.DataFrame(table).T
    feature_table.index.name = "layer_id"
    return feature_table
