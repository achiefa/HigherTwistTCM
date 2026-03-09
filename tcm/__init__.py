"""
TCM: Theory Covariance Method for Power Corrections
====================================================

A Python framework for extracting power corrections from global parton
distribution function (PDF) fits using Bayesian inference with theory
covariance matrices.

This package implements the methodology described in:

    R. D. Ball, A. Chiefa, R. Stegeman,
    "Parton distributions with higher twist and jet power corrections",
    arXiv:2511.14387 (2025)

Quick Start
-----------
>>> from tcm import load_fit_config, compute_posteriors
>>> config = load_fit_config("my_nnpdf_fit")
>>> posteriors, covariance = compute_posteriors(config, C, S, predictions, data)

Modules
-------
core
    Bayesian inference algorithms for posterior extraction.
io
    Data loading and serialization utilities.
plotting
    Publication-quality visualization functions.
cfactors
    C-factor production for downstream predictions.
cli
    Command-line interface.

See Also
--------
- GitHub: https://github.com/achiefa/HigherTwistTCM
- Paper: https://arxiv.org/abs/2511.14387
- NNPDF: https://nnpdf.mi.infn.it/
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Core computation functions
from .core import (
    compute_posteriors,
    compute_posterior_covariance,
    fluctuate_with_covariance,
)

# I/O utilities
from .io import (
    load_fit_config,
    load_covariance_matrix,
    load_predictions,
    load_cntrl_pseudodata,
    save_results,
    load_results,
)

# Visualization
from .plotting import (
    plot_posterior,
    plot_covariance_heatmap,
    plot_scatter_validation,
    plot_comparison,
    save_all_posteriors,
    PLOT_SPECS,
)

# C-factor production
from .cfactors import (
    produce_cfactors,
    compute_dis_cfactor,
    compute_jet_cfactor,
    save_cfactor,
)

__version__ = "1.0.0"
__author__ = "Amedeo Chiefa"
__email__ = "amedeo.chiefa@ed.ac.uk"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core functions
    "compute_posteriors",
    "compute_posterior_covariance",
    "fluctuate_with_covariance",
    # I/O
    "load_fit_config",
    "load_covariance_matrix",
    "load_predictions",
    "load_cntrl_pseudodata",
    "save_results",
    "load_results",
    # Plotting
    "plot_posterior",
    "plot_covariance_heatmap",
    "plot_scatter_validation",
    "plot_comparison",
    "save_all_posteriors",
    "PLOT_SPECS",
    # C-factors
    "produce_cfactors",
    "compute_dis_cfactor",
    "compute_jet_cfactor",
    "save_cfactor",
]
