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
    compute_deltas_pc
)

from .io import load_fit_config, load_results, _build_common_dict

logger = logging.getLogger(__name__)

# Default metadata for C-factor files
DEFAULT_AUTHOR = "Amedeo Chiefa"
DEFAULT_DESCRIPTION = "C-factor for power corrections from TCM analysis"

def _uniform_pars_combination(parameters_dict: Dict) -> List[Dict]:
    combination = {}
    for key, values in parameters_dict.items():
        combination[key] = values['yshift']

    return [{"label": "all", "comb": combination}]

def produce_cfactors(
    fitname: str,
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
    fit_config = API.fit(fit=fitname).as_input()['theorycovmatconfig']
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

    # Dummy pdf
    pdf = API.pdf(pdf="NNPDF31_nlo_as_0118")

    # Process each dataset
    for group in groups_data:
        for dataset in group.datasets:
            if dataset.name in excluded_exps:
                continue
            if group.name not in included_procs:
                continue
            
            # Collect metadata
            process_type = dataset.commondata.metadata.process_type.name

            # Compute shifts for this dataset
            cfactors = compute_deltas_pc(dataset, pdf, pc_shift, _uniform_pars_combination, True)


            if dataset.name == "NMC_NC_NOTFIXED_EM-F2":
                fk_names = [spec.fkpath[0].name for spec in dataset.fkspecs]
                for fk_name in fk_names:
                    if "_D_" in fk_name: idx_ref = 0
                    elif "_P_" in fk_name: idx_ref = 1
                    target_cfac = cfactors["all"][idx_ref]
                    filename = cfactor_path / f"CF_{label}_{fk_name.split(".pineappl")[0]}.dat"
                    save_cfactor(target_cfac, filename, fitname)

            elif process_type != "DIJET_3D":
                if len(dataset.fkspecs) > 1:
                    raise ValueError(f"Multiple FK specs not supported for: {dataset.name}")
                fkpaths = dataset.fkspecs[0].fkpath
                idx_ref = 0
                for fkp in fkpaths:
                    fk = FkTable.read(fkp)
                    bin_size = fk.table().shape[0]

                    # Save
                    bin_name = fkp.name.split(".")[0]
                    filename = cfactor_path / f"CF_{label}_{bin_name}.dat"
                    save_cfactor(cfactors['all'], filename, fitname)
                    idx_ref += bin_size
            else:
                raise NotImplementedError("C-factor production for DIJET_3D not implemented yet.")

    logger.info(f"C-factors saved to {cfactor_path}")
    return cfactor_path


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
    try:
      error = cfactor_error if cfactor_error is not None else np.zeros(len(cfactor))
      data = np.column_stack((cfactor, error))

      with open(filename, "w") as f:
          f.write(header + "\n")
          for row in data:
              f.write(f"{row[0]:.5f} {row[1]:.5f}\n")
    except Exception as e:
        import ipdb; ipdb.set_trace()