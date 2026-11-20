"""Optimized version of the preprocessing."""


def build_curve_data_groupby(selected_points):
    """Dictionary of the curves, fast."""
    curve_data = {}
    for layer_id, sub in selected_points.groupby("layer_id"):
        curve_data[layer_id] = (sub["lab_head_m"].values, sub["lab_wrc"].values)
    return curve_data
