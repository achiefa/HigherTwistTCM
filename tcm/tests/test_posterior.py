"""
Unit tests for TCM core functionality.

These tests verify the mathematical correctness of the core algorithms
without requiring the full NNPDF infrastructure.
"""

from pathlib import Path

import numpy as np
import pytest
import pandas as pd
import pandas.testing as tm


from tcm.io import (
  load_fit_config, 
  load_covariance_matrix, 
  load_results, 
  _load_theory_covmat_table
)
from tcm.core import (
  compute_posteriors, 
  compute_posterior_covariance, 
  fluctuate_with_covariance,
  _compute_replica_covariance, 
)
from tcm.plotting import save_all_posteriors

FITNAME = "250814-ac-01-baseline-lowcuts-nnlo"
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "data/FULL_TEST_SIMPLE"
OUT_DIR = CURRENT_DIR / "output"

@pytest.fixture(scope="session")
def reference_files():
    posterior, covmat = load_results(DATA_DIR)
    return posterior, covmat

@pytest.fixture(scope="session")
def reference_cpost():
    C, _ = load_covariance_matrix(FITNAME)
    Cpost = _load_theory_covmat_table(DATA_DIR, "posterior_covmat.csv", C.index)
    return Cpost

def test_posteriors_file(reference_files):
                # Load fit configuration and data
    config = load_fit_config(FITNAME)
    C, S = load_covariance_matrix(FITNAME)

    # Compute posteriors
    posteriors, P_tilde = compute_posteriors(config, C, S)
    posterior_ref, covmat_ref = reference_files

    # Save plots
    config = load_fit_config(FITNAME)

    # Generate fluctuated samples for uncertainty visualization
    replicas = fluctuate_with_covariance(
        P_tilde.to_numpy(),
        posteriors.to_numpy(),
        n_replicas=10000,
        seed=2143123,
    )
    mean = pd.Series(replicas.mean(axis=1), index=posteriors.index)
    std = pd.Series(replicas.std(axis=1), index=posteriors.index)

    save_all_posteriors(mean, std, config["nodes"], OUT_DIR / "posteriors")
    
    tm.assert_series_equal(posteriors, posterior_ref)
    tm.assert_frame_equal(P_tilde, covmat_ref, rtol=0.00001)

def test_cpost_file(reference_cpost):
    Cpost_ref = reference_cpost
    C, S = load_covariance_matrix(FITNAME)
    X = _compute_replica_covariance(FITNAME)
    Cpost = compute_posterior_covariance(S, C, X)
    
    tm.assert_frame_equal(Cpost, Cpost_ref, atol=1e-6)