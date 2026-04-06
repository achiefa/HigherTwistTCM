"""
Plotting utilities for TCM analysis visualization.

This module provides functions to visualize:
- Posterior distributions of power correction coefficients
- Covariance matrix heatmaps
- Validation scatter plots
- Comparison plots between different fits
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable

from validphys.theorycovariance.output import matrix_plot_labels

# Enable LaTeX rendering for publication-quality plots
rc("text", usetex=True)


# Plot specifications for each power correction type
PLOT_SPECS = {
    "f2p": {
        "ylabel": r"$H \left( F_2^p \right) \; [\textrm{GeV}^2]$",
        "xlabel": r"$x$",
        "title": r"Proton $F_2$ higher twist",
    },
    "f2d": {
        "ylabel": r"$H \left( F_2^d \right) \; [\textrm{GeV}^2]$",
        "xlabel": r"$x$",
        "title": r"Deuteron $F_2$ higher twist",
    },
    "dis_cc": {
        "ylabel": r"$H \left( \sigma_{CC} \right) \; [\textrm{GeV}^2]$",
        "xlabel": r"$x$",
        "title": r"Charged current DIS higher twist",
    },
    "Hj": {
        "ylabel": r"$H \left( \sigma_j \right) \; [\textrm{GeV}]$",
        "xlabel": r"$y$",
        "title": r"Inclusive jet power correction",
    },
    "H2j_ystar": {
        "ylabel": r"$H \left( \sigma_{2j}^{y^*} \right) \; [\textrm{GeV}]$",
        "xlabel": r"$y^*$",
        "title": r"dijet power correction",
    },
    "H2j_ymax": {
        "ylabel": r"$H \left( \sigma_{2j}^{|y|_{\rm max}} \right) \; [\textrm{GeV}]$",
        "xlabel": r"$|y|_{\rm max}$",
        "title": r"dijet power correction",
    },
    "H2j_yb": {
        "ylabel": r"$H \left( \sigma_{2j}^{y_b} \right) \; [\textrm{GeV}]$",
        "xlabel": r"$y_b$",
        "title": r"dijet power correction",
    },
}

ALIASES_PLOT_SPECS = {
    "H2j_ATLAS": "H2j_ystar",
    "H2j_CMS": "H2j_ymax",
}
DIJET_LEGACY = {v: k for k, v in ALIASES_PLOT_SPECS.items()}

def plot_posterior(
    mean: pd.Series,
    std: pd.Series,
    nodes: Dict[str, List[float]],
    pc_type: str,
    color: str = "red",
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    hatch: Optional[str] = None,
    log_scale: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the posterior distribution for a power correction type.

    Parameters
    ----------
    mean : pd.Series
        Posterior mean values indexed by (HT, nodes).
    std : pd.Series
        Posterior standard deviations.
    nodes : dict
        Dictionary mapping PC types to their x-axis node positions.
    pc_type : str
        Type of power correction (e.g., 'f2p', 'Hj').
    color : str, optional
        Plot color. Default 'red'.
    ax : plt.Axes, optional
        Existing axes to plot on. Creates new figure if None.
    label : str, optional
        Legend label for this curve.
    hatch : str, optional
        Hatch pattern for uncertainty band.
    log_scale : bool, optional
        Use logarithmic x-axis. Default True.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    ax : plt.Axes
        The axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.get_figure()

    spec = PLOT_SPECS.get(pc_type, {"xlabel": "x", "ylabel": "H"})
    xaxis = nodes[pc_type]

    # Extract values for this PC type
    try:
        central = mean.xs(level="HT", key=pc_type).to_numpy()
        uncertainty = std.xs(level="HT", key=pc_type).to_numpy()
    except KeyError:
        central = mean.xs(level="HT", key=DIJET_LEGACY.get(pc_type, pc_type)).to_numpy()
        uncertainty = std.xs(level="HT", key=DIJET_LEGACY.get(pc_type, pc_type)).to_numpy()
    except KeyError:
        raise KeyError(f"Power correction type '{pc_type}' not found in data")

    # Plot central value and uncertainty band
    (line,) = ax.plot(xaxis, central, ls="-", lw=1.5, color=color, label=label)
    ax.scatter(xaxis, central, color=color, s=30, zorder=5)

    ax.fill_between(
        xaxis,
        central - uncertainty,
        central + uncertainty,
        color=color,
        alpha=0.3,
        hatch=hatch,
    )

    # Reference line at zero
    ax.axhline(y=0, ls="--", lw=2, color="black", alpha=0.6)

    # Labels and styling
    ax.set_xlabel(spec["xlabel"], fontsize=24)
    ax.set_ylabel(spec["ylabel"], fontsize=24)
    ax.tick_params(axis="both", labelsize=18)

    if log_scale and pc_type in ["f2p", "f2d", "dis_cc"]:
        ax.set_xscale("log")

    return fig, ax


def plot_covariance_heatmap(
    covmat: pd.DataFrame,
    title: str = "",
    figsize: Tuple[int, int] = (15, 15),
) -> plt.Figure:
    """
    Create a heatmap visualization of a covariance matrix.

    Uses symmetric log normalization to handle both positive and
    negative values with a wide dynamic range.

    Parameters
    ----------
    covmat : pd.DataFrame
        Covariance matrix to plot.
    title : str, optional
        Plot title.
    figsize : tuple, optional
        Figure size. Default (15, 15).

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    matrix = covmat.values
    max_val = np.abs(matrix).max()

    # Use symmetric log scale for visualization
    im = ax.matshow(
        matrix,
        cmap=cm.Spectral_r,
        norm=mcolors.SymLogNorm(
            linthresh=1e-5, linscale=1, vmin=-max_val, vmax=max_val
        ),
    )

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(label=r"$\widetilde{P}$", fontsize=20)
    cbar.ax.tick_params(labelsize=16)

    # Dataset labels
    ticklocs, ticklabels, startlocs = matrix_plot_labels(covmat)
    ax.set_xticks(ticklocs)
    ax.set_xticklabels(ticklabels, rotation=30, ha="right", fontsize=16)
    ax.xaxis.tick_bottom()
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(ticklabels, fontsize=16)

    # Grid lines between datasets
    startlocs_lines = [x - 0.5 for x in startlocs]
    ax.vlines(startlocs_lines, -0.5, len(matrix) - 0.5, linestyles="dashed", alpha=0.7)
    ax.hlines(startlocs_lines, -0.5, len(matrix) - 0.5, linestyles="dashed", alpha=0.7)

    ax.margins(x=0, y=0)
    if title:
        ax.set_title(title, fontsize=20)

    return fig


def plot_scatter_validation(
    central_values: np.ndarray,
    replica_mean: np.ndarray,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Create a scatter plot comparing central values to replica means.

    This is a validation plot: if the TCM analysis is self-consistent,
    the replica means should scatter around the central values with
    the expected uncertainty.

    Parameters
    ----------
    central_values : np.ndarray
        Original central values.
    replica_mean : np.ndarray
        Mean of fluctuated replicas.
    figsize : tuple, optional
        Figure size. Default (8, 8).

    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(central_values, replica_mean, alpha=0.6, s=20)

    # Diagonal reference line
    lims = [
        min(central_values.min(), replica_mean.min()),
        max(central_values.max(), replica_mean.max()),
    ]
    margin = 0.1 * (lims[1] - lims[0])
    lims = [lims[0] - margin, lims[1] + margin]

    ax.plot(lims, lims, "r-", alpha=0.5, lw=2, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Central values", fontsize=16)
    ax.set_ylabel("Replica mean", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14)

    return fig


def plot_comparison(
    results: Dict[str, Tuple[pd.Series, pd.Series]],
    nodes: Dict[str, Dict[str, List[float]]],
    pc_type: str,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    hatches: Optional[List[str]] = None,
    log_scale: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compare posterior distributions from multiple fits.

    Parameters
    ----------
    results : dict
        Dictionary mapping fit names to (mean, std) tuples.
    nodes : dict
        Dictionary mapping fit names to their node dictionaries.
    pc_type : str
        Power correction type to compare.
    labels : list, optional
        Display labels for each fit.
    colors : list, optional
        Colors for each fit.
    hatches : list, optional
        Hatch patterns for each fit.
    log_scale : bool, optional
        Use logarithmic x-axis. Default True.

    Returns
    -------
    fig : plt.Figure
        The figure object.
    ax : plt.Axes
        The axes object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    fitnames = list(results.keys())
    n_fits = len(fitnames)

    if labels is None:
        labels = fitnames
    if colors is None:
        colors = [f"C{i}" for i in range(n_fits)]
    if hatches is None:
        hatches = [None] * n_fits

    legends = []
    legend_labels = []

    for i, fitname in enumerate(fitnames):
        mean, std = results[fitname]
        fit_nodes = nodes[fitname]

        plot_posterior(
            mean,
            std,
            fit_nodes,
            pc_type,
            color=colors[i],
            ax=ax,
            hatch=hatches[i],
            log_scale=log_scale,
        )

        # Collect legend handles
        line = plt.Line2D([0], [0], color=colors[i], lw=2)
        patch = plt.Rectangle((0, 0), 1, 1, fc=colors[i], alpha=0.3)
        legends.append((line, patch))
        legend_labels.append(labels[i])

    ax.legend(legends, legend_labels, loc="best", fontsize=16)

    return fig, ax


def save_all_posteriors(
    mean: pd.Series,
    std: pd.Series,
    nodes: Dict[str, List[float]],
    output_dir: Union[str, Path],
    prefix: str = "",
) -> None:
    """
    Generate and save posterior plots for all power correction types.

    Creates both log-scale and linear-scale versions for DIS observables.

    Parameters
    ----------
    mean : pd.Series
        Posterior mean values.
    std : pd.Series
        Posterior standard deviations.
    nodes : dict
        Node positions for each PC type.
    output_dir : Path
        Directory to save plots.
    prefix : str, optional
        Prefix for filenames.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get available PC types from the data
    available_types = mean.index.get_level_values("HT").unique()

    for pc_type in available_types:
        if pc_type not in nodes:
            continue

        # Log scale plot
        fig, ax = plot_posterior(mean, std, nodes, pc_type, log_scale=True)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}{pc_type}_log_scale.png", dpi=150)
        plt.close(fig)

        # Linear scale plot
        fig, ax = plot_posterior(mean, std, nodes, pc_type, log_scale=False)
        ax.set_xscale("linear")
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}{pc_type}_linear_scale.png", dpi=150)
        plt.close(fig)
