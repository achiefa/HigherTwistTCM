"""
C-factor production for power corrections.

C-factors are multiplicative corrections applied to theory predictions
to include power corrections. They are computed as:

    C = 1 + H(x, Q) / Q^n

where H is the extracted higher-twist coefficient and n depends on
the observable (n=2 for DIS, n=1 for jets).

The C-factors are saved in NNPDF format for use in subsequent fits
or predictions.
"""

import datetime
import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import pandas as pd

from pineappl.fk_table import FkTable
from validphys.api import API
from validphys.core import DataSetSpec
from validphys.theorycovariance.higher_twist_functions import (
    DIJET_RAPIDITY_VAR,
    dis_pc_func,
    jets_pc_func,
    get_pc_type,
)

from .io import load_fit_config, load_results, _build_common_dict

logger = logging.getLogger(__name__)

# Default metadata for C-factor files
DEFAULT_AUTHOR = "Amedeo Chiefa"
DEFAULT_DESCRIPTION = "C-factor for power corrections from TCM analysis"


def produce_cfactors(
    fit_config: Dict[str, Any],
    posteriors_dir: Path,
    output_dir: Path,
    label: str = "PC",
) -> None:
    """
    Produce C-factors for all datasets in a fit using extracted posteriors.

    This is the main entry point for C-factor production. It loads the
    posterior values, determines which datasets need C-factors, and
    generates them for each FK table.

    Parameters
    ----------
    fit_config : dict
        Configuration dictionary containing pc_parameters, pc_excluded_exps, etc.
    posteriors_dir : Path
        Directory containing posterior results (posteriors.pkl, P_tilde.pkl).
    output_dir : Path
        Directory to save C-factor files.
    label : str, optional
        Label for C-factor filenames. Default "PC".
    """
    fitname = fit_config["fitname"]
    logger.info(f"Producing C-factors for fit: {fitname}")

    # Load posterior values and fit configuration
    pc_parameters = fit_config["pc_parameters"]
    excluded_exps = fit_config["pc_excluded_exps"]
    included_procs = fit_config["pc_included_procs"]

    pstr, _ = load_results(posteriors_dir)

    # Build parameter dictionaries with posterior values
    pc_shift = defaultdict(dict)

    for ht_name in pc_parameters.keys():
        pc_shift[ht_name] = {
            "yshift": pstr.xs(ht_name, level=0).tolist(),
            "nodes": pc_parameters[ht_name]["nodes"],
        }

    # Setup output directory
    cfactor_path = output_dir / f"cfactors_{fitname}"
    cfactor_path.mkdir(parents=True, exist_ok=True)

    # Get dataset specifications
    common_dict = _build_common_dict(fitname)
    groups_data = API.groups_data_by_process(**common_dict)

    # Process each dataset
    for group in groups_data:
        for dataset in group.datasets:
            if dataset.name in excluded_exps:
                continue
            if group.name not in included_procs:
                continue

            process_type = dataset.commondata.metadata.process_type.name

            try:
                if process_type.startswith("DIS"):
                    _process_dis_dataset(
                        dataset, pc_shift, fitname, cfactor_path, label
                    )
                elif process_type == "JET":
                    _process_jet_dataset(
                        dataset, pc_shift, fitname, cfactor_path, label
                    )
                elif process_type == "DIJET":
                    _process_dijet_dataset(
                        dataset, pc_shift, fitname, cfactor_path, label
                    )
                else:
                    logger.warning(f"Unknown process type: {process_type}")

            except Exception as e:
                logger.error(f"Failed to process {dataset.name}: {e}")
                raise

    logger.info(f"C-factors saved to {cfactor_path}")


def compute_dis_cfactor(
    x: np.ndarray,
    Q2: np.ndarray,
    yshift: List[float],
    nodes: List[float],
) -> np.ndarray:
    """
    Compute DIS C-factors for higher twist corrections.

    The DIS power correction has the form H(x)/Q^2, where H(x) is
    interpolated from the extracted posterior values at the nodes.

    Parameters
    ----------
    x : np.ndarray
        Bjorken x values.
    Q2 : np.ndarray
        Q^2 values in GeV^2.
    yshift : list
        Posterior values at each node.
    nodes : list
        Node positions in x.

    Returns
    -------
    np.ndarray
        C-factor values (1 + correction).
    """
    correction = dis_pc_func(delta_h=yshift, nodes=nodes, x=x, Q2=Q2)
    return 1.0 + correction


def compute_jet_cfactor(
    rapidity: np.ndarray,
    pT: np.ndarray,
    yshift: List[float],
    nodes: List[float],
) -> np.ndarray:
    """
    Compute jet C-factors for power corrections.

    The jet power correction has the form H(y)/pT, where H(y) is
    interpolated from the extracted posterior values.

    Parameters
    ----------
    rapidity : np.ndarray
        Rapidity values.
    pT : np.ndarray
        Transverse momentum values in GeV.
    yshift : list
        Posterior values at each node.
    nodes : list
        Node positions in rapidity.

    Returns
    -------
    np.ndarray
        C-factor values (1 + correction).
    """
    correction = jets_pc_func(delta_h=yshift, nodes=nodes, rap=rapidity, pT=pT)
    return 1.0 + correction


def save_cfactor(
    cfactor: np.ndarray,
    filename: Path,
    fitname: str,
    setname: str = "",
    cfactor_error: Optional[np.ndarray] = None,
    author: str = DEFAULT_AUTHOR,
    description: str = DEFAULT_DESCRIPTION,
) -> None:
    """
    Save C-factor to file in NNPDF format.

    The format consists of a header with metadata followed by
    two columns: the C-factor value and its uncertainty.

    Parameters
    ----------
    cfactor : np.ndarray
        C-factor values.
    filename : Path
        Output file path.
    fitname : str
        Name of the fit used to extract posteriors.
    setname : str, optional
        Dataset name for the header.
    cfactor_error : np.ndarray, optional
        C-factor uncertainties. Defaults to zeros.
    author : str, optional
        Author for metadata.
    description : str, optional
        Description for metadata.
    """
    date = datetime.datetime.now().strftime("%Y-%m-%d")

    header = f"""********************************************************************************
SetName: {setname}
Author: {author}
Date: {date}
CodesUsed: {description}
FromFit: {fitname}
TheoryInput:
PDFset:
Warnings:
********************************************************************************"""

    error = cfactor_error if cfactor_error is not None else np.zeros(len(cfactor))
    data = np.column_stack((cfactor, error))

    with open(filename, "w") as f:
        f.write(header + "\n")
        for row in data:
            f.write(f"{row[0]:.5f} {row[1]:.5f}\n")


# -----------------------------------------------------------------------------
# Private helper functions
# -----------------------------------------------------------------------------

def _process_fktable_list(
    fk_paths: List[Path],
    kin_var: np.ndarray,
    scale_var: np.ndarray,
    pc_func: Callable,
    fitname: str,
    output_dir: Path,
    label: str,
) -> None:
    """Process a list of FK tables for a single dataset."""
    idx_ref = 0

    for fk_path in fk_paths:
        fk = FkTable.read(fk_path)
        bin_size = fk.table().shape[0]

        # Extract kinematics for this FK table
        kin_bin = kin_var[idx_ref : idx_ref + bin_size]
        scale_bin = scale_var[idx_ref : idx_ref + bin_size]

        # Compute C-factor
        cfactor = pc_func(kin_bin, scale_bin)

        # Save
        bin_name = fk_path.name.split(".")[0]
        filename = output_dir / f"CF_{label}_{bin_name}.dat"
        save_cfactor(cfactor, filename, fitname)

        idx_ref += bin_size


def _process_dis_dataset(
    dataset: DataSetSpec,
    pc_params: Dict,
    fitname: str,
    output_dir: Path,
    label: str,
) -> None:
    """Process a DIS dataset and compute C-factors."""
    exp_name = dataset.name
    process_type = dataset.commondata.metadata.process_type.name
    kinematics = dataset.commondata.metadata.load_kinematics()
    x = kinematics["x"].to_numpy().reshape(-1)
    Q2 = kinematics["Q2"].to_numpy().reshape(-1)

    pc_type = get_pc_type(exp_name, process_type)

    # NMC ratio: special case with two FK specs (proton and deuteron)
    if isinstance(pc_type, tuple):
        _process_nmc_ratio(dataset, pc_params, x, Q2, fitname, output_dir, label)
        return

    params = pc_params[pc_type]

    if len(dataset.fkspecs) > 1:
        raise ValueError(f"Multiple FK specs not supported for: {exp_name}")

    def pc_func(kin, scale):
        return compute_dis_cfactor(kin, scale, params["yshift"], params["nodes"])

    _process_fktable_list(
        dataset.fkspecs[0].fkpath, x, Q2, pc_func, fitname, output_dir, label
    )


def _process_nmc_ratio(
    dataset: DataSetSpec,
    pc_params: Dict,
    x: np.ndarray,
    Q2: np.ndarray,
    fitname: str,
    output_dir: Path,
    label: str,
) -> None:
    """Special handling for NMC F2 ratio dataset (proton and deuteron)."""
    fk_names = [spec.fkpath[0].name.split(".")[0] for spec in dataset.fkspecs]

    for fk_name in fk_names:
        if "_P_" in fk_name:
            params = pc_params["f2p"]
        elif "_D_" in fk_name:
            params = pc_params["f2d"]
        else:
            continue

        cfactor = compute_dis_cfactor(x, Q2, params["yshift"], params["nodes"])
        filename = output_dir / f"CF_{label}_{fk_name}.dat"
        save_cfactor(cfactor, filename, fitname)


def _process_jet_dataset(
    dataset: DataSetSpec,
    pc_params: Dict,
    fitname: str,
    output_dir: Path,
    label: str,
) -> None:
    """Process a jet dataset and compute C-factors."""
    kinematics = dataset.commondata.metadata.load_kinematics()
    eta = kinematics["y"].to_numpy().reshape(-1)
    pT = kinematics["pT"].to_numpy().reshape(-1)
    params = pc_params["Hj"]

    if len(dataset.fkspecs) > 1:
        raise ValueError(f"Multiple FK specs not supported for: {dataset.name}")

    def pc_func(kin, scale):
        return compute_jet_cfactor(kin, scale, params["yshift"], params["nodes"])

    _process_fktable_list(
        dataset.fkspecs[0].fkpath, eta, pT, pc_func, fitname, output_dir, label
    )


def _process_dijet_dataset(
    dataset: DataSetSpec,
    pc_params: Dict,
    fitname: str,
    output_dir: Path,
    label: str,
) -> None:
    """Process a dijet dataset and compute C-factors."""
    experiment = dataset.commondata.metadata.experiment
    process_type = dataset.commondata.metadata.process_type.name
    kinematics = dataset.commondata.metadata.load_kinematics()

    pc_type = get_pc_type(
        dataset.name, process_type, experiment=experiment, pc_dict=pc_params
    )
    if process_type == "DIJET_3D":
        rap_var = 'ystar'
    else:
        rap_var = DIJET_RAPIDITY_VAR.get(experiment)  
        if rap_var is None:
            raise ValueError(f"Unknown dijet experiment: {experiment}")

    kin_var = kinematics[rap_var].to_numpy().reshape(-1)
    scale_var = kinematics["m_jj"].to_numpy().reshape(-1)

    params = pc_params[pc_type]

    if len(dataset.fkspecs) > 1:
        raise ValueError(f"Multiple FK specs not supported for: {dataset.name}")

    def pc_func(kin, scale):
        return compute_jet_cfactor(kin, scale, params["yshift"], params["nodes"])

    _process_fktable_list(
        dataset.fkspecs[0].fkpath, kin_var, scale_var, pc_func, fitname, output_dir, label
    )
