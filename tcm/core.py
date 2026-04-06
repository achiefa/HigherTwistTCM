"""
Core TCM computation functions.

This module implements the Theory Covariance Method (TCM) for extracting power
corrections from PDF fits. The key idea is to treat unknown power corrections as
nuisance parameters with Gaussian priors, then compute their posterior
distributions given the data.

The TCM formalism is based on the following:
    - C: covariance matrix (experimental + scale variation uncertainties)
    - S: theory covariance matrix encoding power correction uncertainties
    - beta: matrix of power correction shifts per data point
    - X: PDF replica covariance matrix

The implementation of the TCM follows arXiv:2105.05114, in particular equations
(3.37) and (3.38) for the posterior mean and covariance of the power correction
coefficients.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from functools import lru_cache

from validphys.api import API
from validphys.theorycovariance.higher_twist_functions import compute_deltas_pc

from .io import load_predictions, load_cntrl_pseudodata

def compute_posteriors(
    fit_config: Dict[str, Any],
    covariance_matrix: pd.DataFrame,
    theory_covariance: pd.DataFrame,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute the posterior distribution for power correction coefficients.

    The posterior mean is computed as:
        delta_tilde = -S_hat @ inv(C + S) @ (mean_prediction - data)

    The posterior covariance is:
        P_tilde = S_tilde - S_hat @ inv(C + S) @ S_hat^T
                  + S_hat @ inv(C + S) @ X @ inv(C + S) @ S_hat^T

    where X is the PDF replica covariance.

    Parameters
    ----------
    fit_config : dict
        Configuration dictionary containing pc_parameters, pc_excluded_exps, etc.
    covariance_matrix : pd.DataFrame
        Total experimental + scale variation covariance matrix (C).
    theory_covariance : pd.DataFrame
        Power correction theory covariance matrix (S).

    Returns
    -------
    posteriors : pd.Series
        Posterior mean values for each power correction coefficient.
    P_tilde : pd.DataFrame
        Posterior covariance matrix for the coefficients.
    """
    pc_parameters = fit_config["pc_parameters"]
    y_shifts = {name: params["yshift"] for name, params in pc_parameters.items()}

    predictions = load_predictions(fit_config["fitname"])
    pseudodata = load_cntrl_pseudodata(fit_config["fitname"])

    # Build index for power correction coefficients
    ht_index = _build_ht_index(y_shifts)

    # Compute beta matrices (data space <-> coefficient space mappings)
    beta = _construct_beta_matrix(fit_config, covariance_matrix.index)
    beta_tilde = _construct_beta_tilde_matrix(y_shifts, ht_index)

    # Compute S_tilde (prior covariance in coefficient space)
    S_tilde = _compute_S_tilde(beta_tilde)

    # Compute S_hat (cross-covariance between coefficient and data space)
    S_hat = _compute_S_hat(beta, beta_tilde)

    # Compute PDF replica covariance
    X = _compute_replica_covariance(fit_config["fitname"])

    # Compute inverse of total covariance, dropping the "group" level
    # so that the index matches S_hat columns (which come from beta after droplevel("group"))
    C_plus_S = covariance_matrix + theory_covariance
    inv_cov = np.linalg.inv(C_plus_S.values)
    data_index = C_plus_S.index.droplevel("group")
    inv_cov_df = pd.DataFrame(inv_cov, index=data_index, columns=data_index)

    # Mean prediction
    pred_replicas = predictions.iloc[:, 2:].values
    mean_prediction = pred_replicas.mean(axis=1)

    # Compute posterior mean: delta = -S_hat @ inv(C+S) @ (T - D)
    residual = mean_prediction - pseudodata
    delta_tilde = -S_hat.values @ inv_cov @ residual

    # Compute posterior covariance
    X_df = pd.DataFrame(X, index=S_hat.columns, columns=S_hat.columns)

    P_tilde = (
        S_tilde
        + S_hat @ inv_cov_df @ X_df @ inv_cov_df @ S_hat.T
        - S_hat @ inv_cov_df @ S_hat.T
    )

    # Build posterior Series with proper index
    # TODO: Maybe we will want to allow values other than zeros.
    central_values = pd.Series(np.zeros(len(ht_index)), index=ht_index)
    posteriors = central_values + delta_tilde

    return posteriors, P_tilde


def compute_posterior_covariance(
    theory_covariance: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
    replica_covariance: np.ndarray,
) -> pd.DataFrame:
    """
    Compute the posterior covariance of the power correction predictions.

    This is the covariance matrix Cpost that quantifies the uncertainty
    in the extracted power corrections:
        Cpost = S - S @ inv(C+S) @ S + S @ inv(C+S) @ X @ inv(C+S) @ S

    Parameters
    ----------
    theory_covariance : pd.DataFrame
        Power correction theory covariance (S).
    covariance_matrix : pd.DataFrame
        Total experimental covariance (C).
    replica_covariance : np.ndarray
        PDF replica covariance (X).

    Returns
    -------
    pd.DataFrame
        Posterior covariance matrix.
    """
    S = theory_covariance
    C = covariance_matrix
    X = pd.DataFrame(replica_covariance, index=C.index, columns=C.columns)

    inv_cov = np.linalg.inv((C + S).values)
    inv_cov_df = pd.DataFrame(inv_cov, index=C.index, columns=C.columns)

    Cpost = S - S @ inv_cov_df @ S + S @ inv_cov_df @ X @ inv_cov_df @ S
    return Cpost


def fluctuate_with_covariance(
    covariance: np.ndarray,
    central_values: np.ndarray,
    n_replicas: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate fluctuated samples using Cholesky decomposition.

    This is used to propagate uncertainties from the posterior covariance
    to generate replica-like samples for uncertainty visualization.

    Parameters
    ----------
    covariance : np.ndarray
        Covariance matrix (must be positive semi-definite).
    central_values : np.ndarray
        Central values to fluctuate around.
    n_replicas : int, optional
        Number of replicas to generate. Default is 1000.
    seed : int, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, n_replicas) with fluctuated values.
    """
    n_points = len(central_values)
    samples = np.zeros((n_points, n_replicas))

    # Remove zero rows/columns (for coefficients with zero prior uncertainty)
    nonzero_mask = np.any(covariance != 0, axis=0) & np.any(covariance != 0, axis=1)
    cov_reduced = covariance[np.ix_(nonzero_mask, nonzero_mask)]
    central_reduced = central_values[nonzero_mask]

    # Cholesky decomposition
    L = np.linalg.cholesky(cov_reduced)

    # Generate all samples at once
    rng = np.random.default_rng(seed=seed)
    noise = L @ rng.normal(size=(len(central_reduced), n_replicas))
    samples[:] = central_values[:, np.newaxis]
    samples[nonzero_mask, :] += noise

    return samples


# -----------------------------------------------------------------------------
# Private helper functions
# -----------------------------------------------------------------------------


def _build_ht_index(y_shifts: Dict[str, list]) -> pd.MultiIndex:
    """Build a MultiIndex for power correction coefficients. The index has two
    levels: HT name (e.g. "H2p" for the proton structure function F2p) and the
    respective node index (e.g. "H2p(0)", "H2p(1)", etc.).
    """
    ht_names = []
    node_labels = []

    for name, shifts in y_shifts.items():
        for idx in range(len(shifts)):
            ht_names.append(name)
            node_labels.append(f"{name}({idx})")

    return pd.MultiIndex.from_tuples(
        list(zip(ht_names, node_labels)), names=["HT", "nodes"]
    )


def _construct_beta_matrix(
    fit_config: Dict[str, Any], exp_index: pd.MultiIndex
) -> pd.DataFrame:
    """
    Construct the beta matrix mapping coefficients to data shifts.

    beta[i, alpha] gives the shift in data point i due to the shift in the
    coefficient alpha. Since the corrections are treated as multiplicative shifts,
    beta is computed as
          beta[i, alpha] = T_i * (1 + H[i,alpha]) - T_i = T_i * H[i, alpha]
    where T_i is the theory prediction for data point i, and H[i, alpha]
    is the relative shift computed by compute_deltas_pc for coefficient alpha.
    """
    fitname = fit_config["fitname"]
    pc_parameters = fit_config["pc_parameters"]
    pc_excluded_exps = fit_config.get("pc_excluded_exps", [])
    pc_included_prcs = fit_config.get("pc_included_procs", [])
    covmat_pdf = fit_config["covmat_pdf"] # PDF for the theory predictions

    # Get groups data
    common_dict = dict(
        dataset_inputs={"from_": "fit"},
        fit=fitname,
        use_cuts="fromfit",
        metadata_group="nnpdf31_process",
        theory={"from_": "fit"},
        theoryid={"from_": "theory"},
    )
    groups_data = API.groups_data_by_process(**common_dict)
    pdf = API.pdf(pdf=covmat_pdf)

    # Compute shifts for each included dataset
    shifts = {}
    for group in groups_data:
        for dataset in group.datasets:
            if dataset.name not in pc_excluded_exps and group.name in pc_included_prcs:
                shifts[dataset.name] = compute_deltas_pc(dataset, pdf, pc_parameters)

    # Build beta matrix
    y_shifts = {name: params["yshift"] for name, params in pc_parameters.items()}
    ht_index = _build_ht_index(y_shifts)
    col_index = ht_index.droplevel("HT")
    col_index.name = "shifts"

    beta = pd.DataFrame(
        np.zeros((len(exp_index), len(col_index))),
        index=exp_index,
        columns=col_index,
    )
    beta = beta.droplevel("group")

    for exp_name, exp_shifts in shifts.items():
        for coeff_name, shift_values in exp_shifts.items():
            beta.loc[exp_name, coeff_name] = shift_values

    return beta


def _construct_beta_tilde_matrix(
    y_shifts: Dict[str, list], ht_index: pd.MultiIndex
) -> pd.DataFrame:
    """
    Construct the diagonal beta_tilde matrix (prior standard deviations).

    beta_tilde is diagonal with the prior widths on the diagonal.
    """
    diag_values = np.concatenate(list(y_shifts.values()))
    beta_tilde = np.diag(diag_values)
    return pd.DataFrame(beta_tilde, index=ht_index, columns=ht_index)


def _compute_S_tilde(beta_tilde: pd.DataFrame) -> pd.DataFrame:
    """Compute prior covariance in coefficient space: S_tilde = beta_tilde @ beta_tilde^T."""
    return beta_tilde @ beta_tilde.T


def _compute_S_hat(beta: pd.DataFrame, beta_tilde: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-covariance between coefficient and data space."""
    return beta_tilde.droplevel(level="HT", axis=1) @ beta.T


@lru_cache(maxsize=None)
def _compute_replica_covariance(fitname: str) -> np.ndarray:
    """Compute the PDF replica covariance matrix X."""
    from .io import load_predictions
    predictions = load_predictions(fitname)
    replicas = predictions.iloc[:, 2:].values # Drop central central data and central replica
    mean = replicas.mean(axis=1, keepdims=True)
    deviations = replicas - mean
    X = deviations @ deviations.T / replicas.shape[1]
    return X
