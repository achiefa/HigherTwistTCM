# TCM: Theory Covariance Method for Power Corrections

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.14387-b31b1b.svg)](https://arxiv.org/abs/2511.14387)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python framework for extracting power corrections from global PDF fits using Bayesian inference with theory covariance matrices.

## About This Project

This repository contains the production code I developed for:

> R. D. Ball, A. Chiefa, R. Stegeman,
> **"Parton distributions with higher twist and jet power corrections"**,
> [arXiv:2511.14387](https://arxiv.org/abs/2511.14387)

The work addresses a fundamental challenge in precision QCD: accounting for sub-leading power corrections that can bias PDF extractions if ignored. The methodology has direct applications to LHC phenomenology, including Higgs boson production cross-section predictions.

## Highlights

- **Published Research**: Core methodology for [arXiv:2511.14387](https://arxiv.org/abs/2511.14387), contributing to precision QCD phenomenology at the LHC
- **Bayesian Inference**: Extracts posterior distributions for nuisance parameters using theory covariance matrices
- **Production-Ready**: Generates C-factors compatible with the NNPDF fitting framework for downstream analyses
- **Modular Design**: Clean functional architecture with separation of I/O, computation, and visualization

## Scientific Context

### The Problem

Parton Distribution Functions (PDFs) are essential inputs for predicting cross-sections at hadron colliders like the LHC. Standard PDF fits assume leading-twist QCD, but **power corrections** — terms suppressed by powers of the hard scale — can bias the extracted PDFs if ignored:

| Correction Type | Observable | Suppression | Physical Origin |
|----------------|------------|-------------|-----------------|
| Higher twist (DIS) | F₂, σ_CC | O(1/Q²) | Multi-parton correlations |
| Jet power corrections | σ_jet | O(1/pT) | Hadronization, underlying event |

### The Solution: Theory Covariance Method

Rather than fitting power corrections directly (which would require a specific functional form), the TCM treats them as **nuisance parameters with Gaussian priors**:

1. **Prior specification**: Power corrections are parameterized at kinematic nodes with prior widths σ
2. **Covariance construction**: Build a theory covariance matrix S encoding correlations between data points
3. **Marginalization**: The PDF fit marginalizes over the nuisance parameters automatically
4. **Posterior extraction**: After the fit, extract posterior estimates using Bayes' theorem

The posterior for coefficient **δ** given data **D** and theory **T** is:

```
δ_posterior = -Ŝ (C + S)⁻¹ (T - D)
P_posterior = S̃ - Ŝ (C + S)⁻¹ Ŝᵀ + Ŝ (C + S)⁻¹ X (C + S)⁻¹ Ŝᵀ
```

where **C** is experimental covariance, **S** is theory covariance, **X** is PDF uncertainty, and **Ŝ** maps coefficient space to data space.

## Installation

```bash
# Requires NNPDF framework (validphys2)
conda install nnpdf -c conda-forge

# Install this package
pip install -e .
```

## Quick Start

### Command Line

```bash
# Extract power correction posteriors from a fit
python -m tcm analyse FITNAME

# Compare results across multiple fits
python -m tcm compare config.yaml

# Generate C-factors for predictions
python -m tcm cfactors FITNAME
```

### Python API

```python
from tcm import (
    load_fit_config,
    load_covariance_matrix,
    load_predictions,
    load_cntrl_pseudodata,
    compute_posteriors,
    fluctuate_with_covariance,
    plot_posterior,
)

# Load fit data
config = load_fit_config("my_nnpdf_fit")
C, S = load_covariance_matrix("my_nnpdf_fit")
predictions = load_predictions("my_nnpdf_fit")
pseudodata = load_cntrl_pseudodata("my_nnpdf_fit")

# Compute posteriors via Bayesian update
posteriors, P_tilde = compute_posteriors(config, C, S, predictions, pseudodata)

# Uncertainty propagation via Cholesky sampling
samples = fluctuate_with_covariance(P_tilde.values, posteriors.values, n_replicas=10000)

# Visualize results
import pandas as pd
mean = pd.Series(samples.mean(axis=1), index=posteriors.index)
std = pd.Series(samples.std(axis=1), index=posteriors.index)
fig, ax = plot_posterior(mean, std, config["nodes"], "f2p")
```

## Architecture

```
tcm/
├── core.py      # Bayesian inference: posteriors, covariances, sampling
├── io.py        # Data pipeline: fit configs, covariance matrices, serialization
├── plotting.py  # Visualization: posteriors, heatmaps, comparisons
├── cfactors.py  # Production: DIS/jet C-factor generation
└── cli.py       # Interface: argument parsing, batch processing, reporting
```

### Design Principles

- **Functional core**: Pure functions for computations enable testing and reproducibility
- **Lazy loading**: Expensive validphys API calls are deferred and cached
- **Type safety**: Comprehensive type hints for IDE support and documentation
- **Separation of concerns**: I/O, computation, and presentation are decoupled

## Key Features

### Posterior Extraction

Computes the Bayesian posterior for power correction coefficients given the data-theory residuals and covariance structure:

```python
def compute_posteriors(fit_config, C, S, predictions, pseudodata):
    """
    Bayesian update: P(δ|D) ∝ P(D|δ) P(δ)

    Returns posterior mean and covariance for power correction coefficients.
    """
```

### Uncertainty Propagation

Generates correlated samples from the posterior using Cholesky decomposition:

```python
def fluctuate_with_covariance(covariance, central_values, n_replicas=1000):
    """
    Sample from N(μ, Σ) using L L^T = Σ decomposition.

    Handles singular covariance by projecting out zero modes.
    """
```

### C-Factor Production

Converts extracted posteriors into multiplicative correction factors:

```python
def produce_cfactors(fitname, posteriors_dir, output_dir):
    """
    Generate C = 1 + H(kin)/Q^n for each FK table.

    Output format compatible with NNPDF fitting infrastructure.
    """
```

## Output

| File | Content |
|------|---------|
| `posteriors.pkl` | Posterior mean values (MultiIndex Series) |
| `P_tilde.pkl` | Posterior covariance matrix |
| `auto_covmat.csv` | Autoprediction covariance for validation |
| `heatmap.png` | Covariance structure visualization |
| `{type}_{scale}.png` | Posterior distributions by observable |

## Citation

If you use this code, please cite:

```bibtex
@article{Ball:2025xxx,
    author = "Ball, Richard D. and Chiefa, Amedeo and Stegeman, Roy",
    title = "{Parton distributions with higher twist and jet power corrections}",
    eprint = "2511.14387",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    year = "2025"
}
```

## References

- [NNPDF Collaboration](https://nnpdf.mi.infn.it/) — PDF fitting framework
- [Theory Covariance Method](https://arxiv.org/abs/1906.10698) — Original TCM formalism
- [validphys2](https://github.com/NNPDF/nnpdf) — Analysis framework

## License

MIT License — see [LICENSE](../LICENSE) for details.
