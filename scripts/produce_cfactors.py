from pathlib import Path
from typing import List
import argparse
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd

from pineappl.fk_table import FkTable
from validphys.api import API
from validphys.core import DataSetSpec, FKTableSpec
from validphys.theorycovariance.higher_twist_functions import (
  dis_pc_func, 
  jets_pc_func,
  F2D_exps, F2P_exps, 
  NC_SIGMARED_P_EM, 
  NC_SIGMARED_P_EP, 
  NC_SIGMARED_P_EAVG)

# Default directories for saving and loading files
DEFAULT_POSTERIORS_DIR = Path(__file__).parent.parent / "Results"
DEFAULT_SAVE_DIR = Path(__file__).parent / "cfactors"

# Default suffix for saved files
DEFAULT_PC_LABEL = "PC"

# Metadata
DESCRIPTION = "C-factor for power corrections"
DATE  = datetime.datetime.now().strftime("%Y-%m-%d")
AUTHOR = "Amedeo Chiefa"

S_dict = dict(
    theorycovmatconfig={"from_": "fit"},
    pdf={"from_": "theorycovmatconfig"},
    point_prescription="power corrections",
    use_t0=True,
    datacuts={"from_": "fit"},
    t0pdfset={"from_": "datacuts"},
)

def save_cfactor(cfactor: List[float],
                 fitname: str,
                 setname: str = "",
                 author: str = AUTHOR, 
                 date: str = DATE, 
                 description: str = DESCRIPTION,
                 savedir: Path = DEFAULT_SAVE_DIR,
                 filename: str = "cfactor.dat") -> None:
   """Save the C-factor to .dat file using the NNPDF c-factor format.

   Args:
       cfactor (List[float]): The 1D array of C-factor values to save.
       fitname (str): Name of the fit used to generate the C-factor.
       setname (str, optional): Name of the dataset. Defaults to "".
       author (str, optional): Author of the file. Defaults to AUTHOR.
       date (str, optional): Date of the file. Defaults to DATE.
       description (str, optional): Description of the file. Defaults to DESCRIPTION.

   Returns:
       None: The function saves the C-factor to a file.
   """
   # Create header
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
   
   # From the 1-D, generate c-factor format
   # First column is array of values, second column is zero
   data = np.column_stack((cfactor, np.zeros(len(cfactor))))

   # Save to file
   with open(savedir / filename, 'w') as f:
      f.write(header + "\n")
      # Format each row with proper spacing
      for row in data:
          f.write(f"{row[0]:.5f} {row[1]:.5f}\n")


def compute_cfactor(pc_function: callable,
                    yshift: float,
                    nodes: List[float],
                    kin_var: List[float],
                    scale_var: List[float]) -> float:
  """Produce the C-factor for power corrections.
  
  Args:
      process_type (str): Type of the process ('DIS', 'JET', 'DIJET').
      yshift (float): Shift in rapidity.
      nodes (List[float]): Nodes for the power correction function.
      kin_var (List[float]): Kinematic variable (e.g., rapidity).
      scale_var (List[float]): Scale variable (e.g., Q^2 or pT).
  
  Returns:
      float: The computed C-factor value.
  """

  if 'DIS' in process_type:
      result = pc_function(
         delta_h=yshift, 
         nodes=nodes, 
         x=kin_var,
         Q2=scale_var)
  elif process_type in ['JET', 'DIJET']:
      result = jets_pc_func(
          delta_h=yshift,
          nodes=nodes,
          rap=kin_var,
          pT=scale_var)
      
  # Add 1 for the leading twist contribution
  result += 1.0

  return result

def process_fktable_list(fk_table_paths: List[Path], 
                         pc_func: callable, 
                         kin_var: List[float], 
                         scale_var: List[float],
                         pc_label: str,
                         fitname: str,
                         savedir: Path):
    """Process a list of FK tables in a dataset. The different FK tables
    refer to the same dataset, but cover different kinematic bins. For this
    reason, the kinematic variables are matched to the FK table."""
    idx_ref = 0
    for fk_path in fk_table_paths:
      # Load FK table and read the bin size
      fk = FkTable.read(fk_path)
      table_bin_size = fk.table().shape[0]

      # Construct filename
      bin_name = fk_path.name.split('.')[0]
      filename = f"CF_{pc_label}_{bin_name}.dat"

      # Select the kinematic variables for the current FK table and compute the C-factor
      kin_var_bin = kin_var[idx_ref:idx_ref + table_bin_size]
      scale_var_bin = scale_var[idx_ref:idx_ref + table_bin_size]
      result = pc_func(kin_var=kin_var_bin, scale_var=scale_var_bin)
      result += np.ones_like(result)

      # Save the C-factor to a file
      save_cfactor(result, fitname=fitname, filename=filename, savedir=savedir)

      # Update the reference index for the next FK table
      idx_ref += table_bin_size

def process_dis_dataset(dataset_sp: DataSetSpec, 
                        pc_dict: dict, 
                        fitname: str,
                        savedir: Path,
                        pc_label: str):
  """Process DIS dataset and compute C-factors."""
  exp_name = dataset_sp.name
  x = dataset_sp.commondata.metadata.load_kinematics()['x'].to_numpy().reshape(-1)
  q2 = dataset_sp.commondata.metadata.load_kinematics()['Q2'].to_numpy().reshape(-1)

  # Handle special case for NMC ratio, which has two
  # fk specs, one for the deuteron and one for the proton
  # target. This is different from a list of fk tables within
  # the same fk spec, which is used to cover different kinematic
  # regions.
  if exp_name == "NMC_NC_NOTFIXED_EM-F2":
    # Collect fk tables
    fk_names = [fkspec.fkpath[0].name.split('.')[0] for fkspec in dataset_sp.fkspecs]
    fk_name_proton = [name for i, name in enumerate(fk_names) if '_P_' in name][0]
    fk_name_deuteron = [name for i, name in enumerate(fk_names) if '_D_' in name][0]

    # Compute C-factor for deuteron
    h2d = dis_pc_func(pc_dict["f2d"]['yshift'], pc_dict["f2d"]['nodes'], x, q2)
    h2d += np.ones_like(h2d)

    # Compute C-factor for proton
    h2p = dis_pc_func(pc_dict["f2p"]['yshift'], pc_dict["f2p"]['nodes'], x, q2)
    h2p += np.ones_like(h2p)

    # Save C-factors
    
    save_cfactor(h2d, fitname=fitname, savedir=savedir, filename=f"CF_{pc_label}_{fk_name_deuteron}.dat")
    save_cfactor(h2p, fitname=fitname, savedir=savedir, filename=f"CF_{pc_label}_{fk_name_proton}.dat")
    return

  # Select appropriate parameters based on experiment type
  elif exp_name in F2P_exps:
      yshift, nodes = pc_dict["f2p"]['yshift'], pc_dict["f2p"]['nodes']
  elif exp_name in F2D_exps:
      yshift, nodes = pc_dict["f2d"]['yshift'], pc_dict["f2d"]['nodes']
  elif exp_name.startswith('EMC_NC_250GEV'):
      raise NotImplementedError(f"The DIS observable for {exp_name} has not been implemented.")
  elif exp_name in np.concatenate([NC_SIGMARED_P_EM, NC_SIGMARED_P_EP, NC_SIGMARED_P_EAVG]):
      yshift, nodes = pc_dict["f2p"]['yshift'], pc_dict["f2p"]['nodes']
  elif any(exp_name.startswith(prefix) for prefix in ['CHORUS_CC', 'NUTEV_CC', 'HERA_CC']):
      yshift, nodes = pc_dict["dis_cc"]['yshift'], pc_dict["dis_cc"]['nodes']
  else:
      raise ValueError(f"The DIS observable for {exp_name} has not been implemented.")
  
  # Ensure that all the datasets but the NMC ratio have only one fk table spec
  if len(dataset_sp.fkspecs) > 1:
        raise ValueError(f"Multiple FKS specifications not supported for {dataset_sp.name}.")
  
  # Construct the function for computing the C-factor
  def pc_func(kin_var, scale_var):
      return dis_pc_func(delta_h=yshift, nodes=nodes, x=kin_var, Q2=scale_var)

  # Process the list of fk tables
  process_fktable_list(
      fk_table_paths=dataset_sp.fkspecs[0].fkpath,
      pc_func=pc_func,
      kin_var=x,
      scale_var=q2,
      pc_label=pc_label,
      fitname=fitname,
      savedir=savedir
  )

def process_jet_dataset(dataset_sp: DataSetSpec, 
                        pc_dict: dict, 
                        fitname: str,
                        savedir: Path,
                        pc_label: str):
    """Process JET dataset and compute c-factors."""
    # Get kinematics and pc parameters
    eta = dataset_sp.commondata.metadata.load_kinematics()['y'].to_numpy().reshape(-1)
    pT = dataset_sp.commondata.metadata.load_kinematics()['pT'].to_numpy().reshape(-1)
    yshift, nodes = pc_dict["Hj"]['yshift'], pc_dict["Hj"]['nodes']

    # Ensure that all the datasets but the NMC ratio have only one fk table spec
    if len(dataset_sp.fkspecs) > 1:
        raise ValueError(f"Multiple FKS specifications not supported for {dataset_sp.name}.")
    
    # Construct the function for computing the C-factor
    def pc_func(kin_var, scale_var):
      return jets_pc_func(delta_h=yshift, nodes=nodes, pT=scale_var, rap=kin_var)

    # Process the list of fk tables
    process_fktable_list(
        fk_table_paths=dataset_sp.fkspecs[0].fkpath,
        pc_func=pc_func,
        kin_var=eta,
        scale_var=pT,
        pc_label=pc_label,
        fitname=fitname,
        savedir=savedir
    )

def process_dijet_dataset(dataset_sp: DataSetSpec, 
                        pc_dict: dict, 
                        fitname: str,
                        savedir: Path,
                        pc_label: str):
    """Process dijet dataset and compute c-factors."""
    exp = dataset_sp.commondata.metadata.experiment

    if exp == 'ATLAS':
        kin_var = dataset_sp.commondata.metadata.load_kinematics()['ystar'].to_numpy().reshape(-1)
        scale_var = dataset_sp.commondata.metadata.load_kinematics()['m_jj'].to_numpy().reshape(-1)
        yshift, nodes = pc_dict["H2j_ATLAS"]['yshift'], pc_dict["H2j_ATLAS"]['nodes']
    
    elif exp == 'CMS':
        kin_var = dataset_sp.commondata.metadata.load_kinematics()['ydiff'].to_numpy().reshape(-1)
        scale_var = dataset_sp.commondata.metadata.load_kinematics()['m_jj'].to_numpy().reshape(-1)
        yshift, nodes = pc_dict["H2j_CMS"]['yshift'], pc_dict["H2j_CMS"]['nodes']
    
    else:
        raise ValueError(f"{exp} is not implemented for DIJET.")

    # Ensure that all the datasets but the NMC ratio have only one fk table spec
    if len(dataset_sp.fkspecs) > 1:
        raise ValueError(f"Multiple FKS specifications not supported for {dataset_sp.name}.")
    
    # Construct the function for computing the C-factor
    def pc_func(kin_var, scale_var):
      return jets_pc_func(delta_h=yshift, nodes=nodes, pT=scale_var, rap=kin_var)

    # Process the list of fk tables
    process_fktable_list(
        fk_table_paths=dataset_sp.fkspecs[0].fkpath,
        pc_func=pc_func,
        kin_var=kin_var,
        scale_var=scale_var,
        pc_label=pc_label,
        fitname=fitname,
        savedir=savedir
    )
  
def load_configuration(fitname,
                       posterior_dir: Path = DEFAULT_POSTERIORS_DIR) -> tuple:
    """Extract power correction parameters from the fit configuration."""
    # Load theory covmat dict
    thcovmat_dict = API.fit(fit=fitname).as_input()["theorycovmatconfig"]
    pc_excluded_exps = thcovmat_dict.get("pc_excluded_exps", [])
    pc_included_prcs = thcovmat_dict.get("pc_included_procs", [])

    pc_parameters = thcovmat_dict['pc_parameters']
    pc_parameters_shift = defaultdict(dict[list])
    pc_parameters_unc = defaultdict(dict[list])
    
    # Load and apply posteriors
    posteriors = pd.read_pickle(posterior_dir / f"{fitname}/posteriors.pkl")
    Ptilde = pd.read_pickle(posterior_dir / f"{fitname}/P_tilde.pkl")

    for ht in pc_parameters.keys():
        pc_parameters_shift[ht] = {
            "yshift": posteriors.xs(ht, level=0).to_list(),
            "nodes": pc_parameters[ht]['nodes'],
        }
        pc_parameters_unc[ht] = {
            "yshift": np.sqrt(Ptilde.xs(ht, level=0).T.xs(ht, level=0).to_numpy().diagonal()).tolist(),
            "nodes": pc_parameters[ht]['nodes'],
        }
  
    return pc_parameters_shift, pc_parameters_unc, pc_excluded_exps, pc_included_prcs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save C-factors for power corrections.")
    parser.add_argument("fitname", type=str, help="Name of the fit")
    parser.add_argument("--posterior_path", type=Path, default=DEFAULT_POSTERIORS_DIR,
                        help="Path to the directory containing posterior files")
    parser.add_argument("--save_dir", type=Path, default=DEFAULT_SAVE_DIR,
                        help="Directory to save the C-factor files")
    parser.add_argument("--pc_label", type=str, default=DEFAULT_PC_LABEL,
                        help="Suffix for the saved C-factor files")
    args = parser.parse_args()

    # Construct dictionary for validphys API
    common_dict = dict(
        dataset_inputs={"from_": "fit"},
        fit=args.fitname,
        fits=[args.fitname],
        use_cuts="fromfit",
        metadata_group="nnpdf31_process",
        theory={"from_": "fit"},
        theoryid={"from_": "theory"},
    )

    cfactor_path = args.save_dir / f"cfactors_{args.fitname}"
    cfactor_path.mkdir(parents=True, exist_ok=True)

    # Load configurations for power corrections
    pc_parameters, pc_parameters_unc, pc_excluded_exps, pc_included_prcs = load_configuration(args.fitname)

    # Process each dataset
    groups_data_by_process = API.groups_data_by_process(**common_dict)
    for group_proc in groups_data_by_process:
        for dataset_sp in group_proc.datasets:
            # Process only if the experiment is used to extract power corrections
            if (dataset_sp.name not in pc_excluded_exps and group_proc.name in pc_included_prcs):
                process_type = dataset_sp.commondata.metadata.process_type.name
                if process_type.startswith('DIS'):
                    process_dis_dataset(dataset_sp, pc_parameters, args.fitname, cfactor_path, args.pc_label)

                elif process_type == 'JET':
                    process_jet_dataset(dataset_sp, pc_parameters, args.fitname, cfactor_path, args.pc_label)

                elif process_type == 'DIJET':
                    process_dijet_dataset(dataset_sp, pc_parameters, args.fitname, cfactor_path, args.pc_label)

                else:
                    raise RuntimeError(f"{process_type} has not been implemented.")