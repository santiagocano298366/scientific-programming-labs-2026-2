"""Cleaning of the data."""
import pandas as pd

from src.config import TEXTURE_MAP


def clean_points(raw):
    """Clean the points."""
    steps = [["raw file", len(raw), raw["layer_id"].nunique()]]

    points = raw.drop_duplicates()
    steps.append(["without duplicates", len(points), points["layer_id"].nunique()])

    points = points[points["data_flag"] == "good quality estimate"]
    steps.append(["good quality fits", len(points), points["layer_id"].nunique()])

    points = points.dropna(subset=["lab_wrc", "lab_head_m"])
    steps.append(["without missing values", len(points), points["layer_id"].nunique()])

    points = points[points["lab_head_m"] <= 100000]
    steps.append(["psi <= 100000 m", len(points), points["layer_id"].nunique()])

    ok = (points["lab_wrc"] >= 0) & (points["lab_wrc"] <= 1) & (points["lab_head_m"] >= 0)
    points = points[ok]
    steps.append(["physical range", len(points), points["layer_id"].nunique()])

    points = points[points["climate_classes"] == "Tropical"]
    steps.append(["tropical climate", len(points), points["layer_id"].nunique()])

    steps = pd.DataFrame(steps, columns=["step", "rows", "curves"])
    return points, steps


def group_textures(points):
    """Add the texture group."""
    labelled = points.copy()
    labelled["texture_group"] = labelled["tex_psda"].map(TEXTURE_MAP)
    labelled = labelled.dropna(subset=["texture_group"])
    return labelled


def build_curve_table(labelled):
    """Table of the curves."""
    table = labelled.groupby("layer_id").agg(
        n_points=("lab_head_m", "size"),
        psi_min=("lab_head_m", "min"),
        psi_max=("lab_head_m", "max"),
        texture_group=("texture_group", "first"),
        profile_id=("profile_id", "first"),
        reference=("reference", "first"),
        alpha_csv=("alpha", "first"),
        n_csv=("n", "first"),
    )
    return table


def select_usable_curves(curve_table, labelled):
    """Keep the usable curves."""
    usable = curve_table[(curve_table["n_points"] >= 6) & (curve_table["psi_min"] <= 1) & (curve_table["psi_max"] >= 100)]
    selected_points = labelled[labelled["layer_id"].isin(usable.index)]
    return usable, selected_points


def flag_upward_jumps(usable, selected_points):
    """Mark the curves that go up."""
    selected_points = selected_points.sort_values(["layer_id", "lab_head_m"])
    change = selected_points.groupby("layer_id")["lab_wrc"].diff()
    jump = (change > 0.05).groupby(selected_points["layer_id"]).any()
    usable = usable.copy()
    usable["has_upward_jump"] = jump
    return usable


def build_curve_data(selected_points, layer_ids):
    """Dictionary of the curves."""
    curve_data = {}
    for layer_id in layer_ids:
        sub = selected_points[selected_points["layer_id"] == layer_id]
        curve_data[layer_id] = (sub["lab_head_m"].values, sub["lab_wrc"].values)
    return curve_data
