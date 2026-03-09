"""
Input/Output utilities for TCM analysis.

This module handles loading fit configurations from validphys,
reading/writing covariance matrices, and saving/loading results.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from functools import lru_cache

import numpy as np
import pandas as pd

from validphys.api import API

logger = logging.getLogger(__name__)

@lru_cache(maxsize=None)
def load_fit_config(fitname: str) -> Dict[str, Any]:
    """
    Load fit configuration and power correction parameters using the
    valdiphys API.

    Parameters
    ----------
    fitname : str
        Name of the NNPDF fit.

    Returns
    -------
    dict
        Configuration dictionary containing:
        - fitname: str
        - fitpath: Path
        - pc_parameters: dict of power correction parameters
        - pc_included_procs: list of included process types
        - pc_excluded_exps: list of excluded experiments
        - covmat_pdf: str, PDF used for theory covariance
        - point_prescriptions: list of prescription names
    """
    logger.info(f"Loading configuration for fit: {fitname}")

    fit = API.fit(fit=fitname)
    fit_dict = fit.as_input()
    thcovmat_dict = fit_dict["theorycovmatconfig"]

    config = {
        "fitname": fitname,
        "fitpath": fit.path,
        "pc_parameters": thcovmat_dict["pc_parameters"],
        "pc_included_procs": thcovmat_dict.get("pc_included_procs", []),
        "pc_excluded_exps": thcovmat_dict.get("pc_excluded_exps", []),
        "covmat_pdf": thcovmat_dict["pdf"],
        "point_prescriptions": thcovmat_dict.get("point_prescriptions", []),
    }

    # Extract node positions and prior widths for convenience
    config["nodes"] = {
        name: params["nodes"] for name, params in config["pc_parameters"].items()
    }
    config["prior_widths"] = {
        name: params["yshift"] for name, params in config["pc_parameters"].items()
    }

    return config

@lru_cache(maxsize=None)
def load_covariance_matrix(
    fitname: str,
    include_scale_variations: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load experimental and theory covariance matrices for a fit.

    The experimental covariance C includes:
    - Statistical uncertainties
    - Systematic uncertainties
    - Optionally, scale variation theory uncertainties

    Parameters
    ----------
    fitname : str
        Name of the NNPDF fit.
    include_scale_variations : bool, optional
        Whether to add scale variation covariances to C. Default True.

    Returns
    -------
    C : pd.DataFrame
        Total experimental covariance matrix.
    S : pd.DataFrame
        Power correction theory covariance matrix.
    """
    logger.info(f"Loading covariance matrices for: {fitname}")

    common_dict = _build_common_dict(fitname)
    config = load_fit_config(fitname)
    fitpath = config["fitpath"]
    point_prescriptions = config.get("point_prescriptions", [])

    # Build prescription index map
    index_map = {name: idx for idx, name in enumerate(point_prescriptions)}

    # Load base experimental covariance
    C = API.groups_covmat_no_table(**common_dict)

    if include_scale_variations:
        # Add scale variation covariances (all prescriptions except power corrections)
        for name, idx in index_map.items():
            if name == "power corrections":
                continue

            table_name = f"datacuts_theory_theorycovmatconfig_point_prescriptions{idx}_theory_covmat_custom_per_prescription.csv"
            S_scale = _load_theory_covmat_table(fitpath, table_name, C.index)
            C = C + S_scale
            logger.debug(f"Added scale variation: {name}")

    # Load power correction covariance
    pc_idx = index_map.get("power corrections")
    if pc_idx is not None:
        table_name = f"datacuts_theory_theorycovmatconfig_point_prescriptions{pc_idx}_theory_covmat_custom_per_prescription.csv"
        S = _load_theory_covmat_table(fitpath, table_name, C.index)
    else:
        logger.warning("No power corrections prescription found, using zero matrix")
        raise ValueError("Power corrections covariance matrix is required for TCM analysis")

    return C, S


@lru_cache(maxsize=None)
def load_predictions(fitname: str) -> pd.DataFrame:
    """
    Load theory predictions for all datasets in a fit.

    Parameters
    ----------
    fitname : str
        Name of the NNPDF fit.

    Returns
    -------
    pd.DataFrame
        Predictions with columns for each replica.
    """
    common_dict = _build_common_dict(fitname)
    predictions = API.group_result_table_no_table(pdf={"from_": "fit"}, **common_dict)
    return predictions

@lru_cache(maxsize=None)
def load_cntrl_pseudodata(fitname: str) -> np.ndarray:
    """
    Load central pseudodata (average over replicas).

    Parameters
    ----------
    fitname : str
        Name of the NNPDF fit.

    Returns
    -------
    np.ndarray
        Central pseudodata values.
    """
    common_dict = _build_common_dict(fitname)
    predictions = load_predictions(fitname)
    pseudodata_list = API.read_pdf_pseudodata(**common_dict)

    # Average over replicas
    central = np.mean(
        [
            ps.pseudodata.reindex(predictions.index.to_list()).to_numpy().flatten()
            for ps in pseudodata_list
        ],
        axis=0,
    )
    return central


def save_results(
    output_dir: Path,
    posteriors: pd.Series,
    P_tilde: pd.DataFrame,
    Cpost: Optional[pd.DataFrame] = None,
) -> None:
    """
    Save TCM analysis results to disk.

    Parameters
    ----------
    output_dir : Path
        Directory to save results.
    posteriors : pd.Series
        Posterior mean values.
    P_tilde : pd.DataFrame
        Posterior covariance matrix.
    Cpost : pd.DataFrame, optional
        Posterior covariance in data space.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to {output_dir}")

    # Save as pickle for full precision
    pd.to_pickle(posteriors, output_dir / "posteriors.pkl")
    pd.to_pickle(P_tilde, output_dir / "P_tilde.pkl")

    if Cpost is not None:
        Cpost.to_csv(output_dir / "Cpost_covmat.csv")


def load_results(results_dir: Path) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Load previously saved TCM results.

    Parameters
    ----------
    results_dir : Path
        Directory containing saved results.

    Returns
    -------
    posteriors : pd.Series
        Posterior mean values.
    P_tilde : pd.DataFrame
        Posterior covariance matrix.
    """
    results_dir = Path(results_dir)

    posteriors = pd.read_pickle(results_dir / "posteriors.pkl")
    P_tilde = pd.read_pickle(results_dir / "P_tilde.pkl")

    return posteriors, P_tilde


# -----------------------------------------------------------------------------
# Private helper functions
# -----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _build_common_dict(fitname: str) -> Dict[str, Any]:
    """Build the common dictionary for validphys API calls."""
    return dict(
        dataset_inputs={"from_": "fit"},
        fit=fitname,
        fits=[fitname],
        use_cuts="fromfit",
        metadata_group="nnpdf31_process",
        theory={"from_": "fit"},
        theoryid={"from_": "theory"},
    )


def _load_theory_covmat_table(
    fitpath: Path, table_name: str, target_index: pd.MultiIndex
) -> pd.DataFrame:
    """Load a theory covariance matrix table and align to target index."""
    table_path = fitpath / "tables" / table_name

    covmat = pd.read_csv(
        table_path,
        index_col=[0, 1, 2],
        header=[0, 1, 2],
        sep=r"\t|,",
        engine="python",
    )

    # Fix known dataset name inconsistencies
    fixed_index = pd.MultiIndex.from_tuples(
        [
            (
                group,
                dataset if dataset != "CMS_2JET_7TEV_M12Y" else "CMS_2JET_7TEV_M12-Y", # TODO to be removed
                np.int64(idx),
            )
            for group, dataset, idx in covmat.index
        ],
        names=["group", "dataset", "id"],
    )

    covmat = pd.DataFrame(covmat.values, index=fixed_index, columns=fixed_index)

    # Align to target index
    covmat = covmat.reindex(target_index).T.reindex(target_index)

    return covmat
