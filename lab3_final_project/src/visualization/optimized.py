"""Fast plot of the curves."""
import numpy as np
import matplotlib.pyplot as plt

from src.config import COLORS


def plot_all_curves_fast(usable, curve_data, groups):
    """Plot all the curves, fast."""
    fig = plt.figure(figsize=(6, 4.5))
    for group in groups:
        psi_list = []
        theta_list = []
        for layer_id, (psi, theta) in curve_data.items():
            if usable.loc[layer_id, "texture_group"] == group:
                psi_list.append(np.append(psi, np.nan))
                theta_list.append(np.append(theta, np.nan))
        plt.plot(np.concatenate(psi_list), np.concatenate(theta_list), color=COLORS[group], alpha=0.12, label=group)
    plt.xscale("symlog", linthresh=0.1)
    plt.xlabel("Matric potential psi (m)")
    plt.ylabel("Water content theta (m3/m3)")
    plt.legend()
    return fig
