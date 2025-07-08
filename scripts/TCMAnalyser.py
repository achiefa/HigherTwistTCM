import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib import colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
rc('text',usetex=True)

from validphys.theorycovariance.output import matrix_plot_labels
from validphys.api import API
from validphys.theorycovariance.higher_twist_functions import compute_deltas_pc


SAVEDIR = Path(__file__).parent.parent / "Results"

def setup_logging(log_level: str = 'INFO') -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

reportengine_logger = logging.getLogger("reportengine")
reportengine_logger.setLevel(logging.WARNING)

validphys_logger = logging.getLogger("validphys")
validphys_logger.setLevel(logging.INFO)

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)

PIL_logger = logging.getLogger("PIL")
PIL_logger.setLevel(logging.WARNING)

urllib3_logger = logging.getLogger("urllib3")
urllib3_logger.setLevel(logging.WARNING)

asyncio_logger = logging.getLogger("asyncio")
asyncio_logger.setLevel(logging.WARNING)

HT_PLOT_SPEC = {"f2p": { "y_label": r"$H \left( F_2^p \right) \; [\textrm{GeV}^2]$",
                        "x_label": r"$x$"},
                "f2d": { "y_label": r"$H \left( F_2^d \right) \; [\textrm{GeV}^2]$",
                        "x_label": r"$x$"},
                "dis_cc": { "y_label": r"$H \left( \sigma_{CC} \right) \; [\textrm{GeV}^2]$",
                            "x_label": r"$x$"},
                "Hj": { "y_label": r"$H \left( \sigma_j \right) \; [\textrm{GeV}]$",
                        "x_label": r"$y$"},
                "H2j_ATLAS": { "y_label": r"$H \left( \sigma_{2j}^{\rm ATLAS} \right) \; [\textrm{GeV}]$",
                              "x_label": r"$y^*$"},
                "H2j_CMS": { "y_label": r"$H \left( \sigma_{2j}^{\rm CMS} \right) \; [\textrm{GeV}]$",
                            "x_label": r"$|y|_{{max}}$"}}  

def plot_covmat_heatmap(covmat: pd.DataFrame, title: str) -> plt.Figure:
    """Matrix plot of a covariance matrix."""
    df = covmat

    matrix = df.values
    fig, ax = plt.subplots(figsize=(15, 15))
    matrixplot = ax.matshow(
        matrix,
        cmap=cm.Spectral_r,
        norm=mcolors.SymLogNorm(
            linthresh=0.00001, linscale=1, vmin=-matrix.max(), vmax=matrix.max()
        ),
    )

    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    cbar = fig.colorbar(matrixplot, cax=cax)
    cbar.set_label(label=r"$\widetilde{P}$", fontsize=20)
    cbar.ax.tick_params(labelsize=20)
    ax.set_title(title, fontsize=25)
    ticklocs, ticklabels, startlocs = matrix_plot_labels(df)
    ax.set_xticks(ticklocs)
    ax.set_xticklabels(ticklabels, rotation=30, ha="right", fontsize=20)
    ax.xaxis.tick_bottom()
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(ticklabels, fontsize=20)
    
    # Shift startlocs elements 0.5 to left so lines are between indexes
    startlocs_lines = [x - 0.5 for x in startlocs]
    ax.vlines(startlocs_lines, -0.5, len(matrix) - 0.5, linestyles="dashed")
    ax.hlines(startlocs_lines, -0.5, len(matrix) - 0.5, linestyles="dashed")
    ax.margins(x=0, y=0)
    return fig

def fluctuate_points_cholesky(mat: np.ndarray, central_data: np.ndarray, 
                            replicas: int = 1, seed: int = 1) -> np.ndarray:
    """Generate fluctuated points using Cholesky decomposition"""
    fluctuate_points = np.zeros((central_data.size, replicas))

    # Remove zero columns and rows
    non_zero_columns = np.any(mat != 0, axis=0)
    covmat = mat[:, non_zero_columns]
    non_zero_rows = np.any(covmat != 0, axis=1)
    covmat = covmat[non_zero_rows, :]
    
    # Apply the same mask to central data
    data = central_data[non_zero_rows]

    # Compute Cholesky decomposition
    L = np.linalg.cholesky(covmat)

    # Loop over replicas
    for k in range(replicas):
        rng = np.random.default_rng(seed=seed + k)
        noise = L @ rng.normal(size=data.size)
        fluctuate_points[:, k] = central_data
        fluctuate_points[non_zero_rows, k] += noise

    return fluctuate_points

class TCMAnalyzer:
  """Class to handle the TCM analysis."""

  def __init__(self, fitname: str):
    self.fitname = fitname
    self.logger = logging.getLogger(self.__class__.__name__ + f"({fitname})")

    self.save_dir = SAVEDIR / fitname
    self.save_dir.mkdir(parents=True, exist_ok=True)

    # Define dictionaries for validphys API calls
    self.common_dict = dict(
    dataset_inputs={"from_": "fit"},
    fit=fitname,
    fits=[fitname],
    use_cuts="fromfit",
    metadata_group="nnpdf31_process",
    theory={"from_": "fit"},
    theoryid={"from_": "theory"},
    )

    # Calculate theory predictions of the input PDF
    self.S_dict = dict(
        theorycovmatconfig={"from_": "fit"},
        pdf={"from_": "theorycovmatconfig"},
        use_t0=True,
        datacuts={"from_": "fit"},
        t0pdfset={"from_": "datacuts"},
    )

    self._setup_config()
  
  def _setup_config(self):
    """Load fit configuration and parameters"""
    self.logger.info("Loading fit configuration")
    
    # Get fit configuration
    fit_dict = API.fit(fit=self.fitname).as_input()
    self.thcovmat_dict = fit_dict["theorycovmatconfig"]
    self.covmat_pdf = self.thcovmat_dict['pdf']
    self.pc_included_prcs = self.thcovmat_dict['pc_included_procs']
    self.pc_excluded_exps = self.thcovmat_dict['pc_excluded_exps']
    self.fitpath = API.fit(fit=self.fitname).path

    pc_parameters = self.thcovmat_dict['pc_parameters']
    # for key in pc_parameters:
    #     pc_parameters[key]['yshift'] = [round(val, 6) for val in pc_parameters[key]['yshift']]
    self.pc_parameters = pc_parameters

    x_nodes = {}
    yshifts = {}

    for pc_name, pc_par in self.pc_parameters.items():
          x_nodes[pc_name] = pc_par['nodes']
          yshifts[pc_name] = pc_par['yshift']
    
    self.x_nodes = x_nodes
    self.y_shifts = yshifts

  def _generate_plots(self):
        """Generate all plots"""
        self.logger.info("Generating plots")
        
        # Plot correlation heatmap
        fig_matmap = plot_covmat_heatmap(self.P_tilde, "")
        fig_matmap.tight_layout()
        fig_matmap.savefig(self.save_dir / 'heatmap.png', dpi=150)
        plt.close(fig_matmap)
        
        # Generate replicas and plot posteriors
        replicas = fluctuate_points_cholesky(self.P_tilde.to_numpy(), 
                                           self.posteriors.to_numpy(), 
                                           replicas=10000, seed=2143123)
        mean = pd.Series(replicas.mean(axis=1), index=self.posteriors.index)
        std = pd.Series(replicas.std(axis=1), index=self.posteriors.index)
        
        # Plot pseudo points against real data
        fig_scatter, ax = plt.subplots(figsize=(8, 8))
        x = np.linspace(-15, 10, 100)
        ax.scatter(self.posteriors.to_numpy(), mean.to_numpy())
        ax.plot(x, x, color="red", ls="-", alpha=0.3)
        ax.set_xlabel("Central values")
        ax.set_ylabel("Average of pseudodata")
        fig_scatter.savefig(self.save_dir / 'scatter.png', dpi=150)
        plt.close(fig_scatter)
        
        # Generate posterior plots
        self._plot_posteriors(mean, std)

  def _plot_posteriors(self, mean: pd.Series, std: pd.Series):
        """Plot posterior distributions"""
        color = "red"
        available_keys = [k for k in self.pc_parameters.keys()]
    
        for key in available_keys:
            fig, ax = plt.subplots(figsize=(10, 5))
            ylabel = HT_PLOT_SPEC[key]['y_label']
            xlabel = HT_PLOT_SPEC[key]['x_label']
            xaxis = self.x_nodes[key]
            
            try:
                shift_central = mean.xs(level='HT', key=key).to_numpy()
                shift_std = std.xs(level='HT', key=key).to_numpy()
                

                pl = ax.plot(xaxis, shift_central, ls="-", lw=1, color=color)
                ax.scatter(xaxis, shift_central, color=color)
                pl_lg = ax.fill(np.NaN, np.NaN, alpha=0.3, color=pl[0].get_color())
                pl_fb = ax.fill_between(xaxis, shift_central - shift_std, 
                                      shift_central + shift_std, 
                                      color=pl[0].get_color(), alpha=0.3)
                ax.plot(xaxis, np.zeros_like(xaxis), ls="dashed", lw=2, 
                        color="black", alpha=0.6)
                
                ax.set_xlabel(xlabel, fontsize=30)
                ax.set_ylabel(ylabel, fontsize=30)
                ax.set_xscale('log')
                
            except KeyError:
                raise KeyError(f"Key '{key}' not found in posteriors data.")
        
            fig.tight_layout()
            fig.savefig(self.save_dir / f"{key}_log_scale.png", dpi=150)
        
            ax.set_xscale('linear')
            fig.savefig(self.save_dir / f"{key}_linear_scale.png", dpi=150)
            plt.close(fig)
  

  def save_results(self):
      """Save results to disk"""
      self.logger.info(f"Saving results to {self.save_dir}")
        
      # Save pickle files
      pd.to_pickle(self.posteriors, self.save_dir / 'posteriors.pkl')
      pd.to_pickle(self.P_tilde, self.save_dir / 'P_tilde.pkl')
      
      # Save CSV versions for readability
      # self.posteriors.to_csv(self.save_dir / 'posteriors.csv')
      # self.P_tilde.to_csv(self.save_dir / 'P_tilde.csv')


  def run_analysis(self, produce_plots=True, save=True):
      """Run the TCM analysis."""
      try:
        # Check if data has been already processed.
        # If so, load the results and skip the analysis.
        self.logger.info(f"Checking if results for {self.fitname} already exist")
        if (self.save_dir / 'posteriors.pkl').exists() and (self.save_dir / 'P_tilde.pkl').exists():
          self.logger.info(f"Results for {self.fitname} already exist. Loading from disk.")
          self.posteriors = pd.read_pickle(self.save_dir / 'posteriors.pkl')
          self.P_tilde = pd.read_pickle(self.save_dir / 'P_tilde.pkl')
        else:
          self.logger.info(f"Starting analysis for {self.fitname}")
          # Compute posteriors
          self._compute_posteriors()

          # Save results
          if save:
              self.save_results()

          self._generate_plots()

          self.logger.info(f"Analysis completed successfully for {self.fitname}")

        if produce_plots:
          self._generate_plots()
          self.logger.info(f"Plots generated for {self.fitname}")
        return True, self.fitname, None
      
      except Exception as e:
        self.logger.error(f"Analysis failed for {self.fitname}: {str(e)}")
        return False, self.fitname, str(e)

  @property
  def C(self):
      """Return the covariance matrix."""
      if not hasattr(self, '_C'):
          self.logger.info("Loading covariance matrix")
          C = API.groups_covmat_no_table(**self.common_dict)

          # Try to load MHO covmat
          try:
              S_scale_var_path = self.fitpath / "tables/datacuts_theory_theorycovmatconfig_point_prescriptions1_theory_covmat_custom_per_prescription.csv"
              S_scale_var = pd.read_csv(S_scale_var_path, index_col=[0, 1, 2], 
                                      header=[0, 1, 2], sep="\t|,", engine="python")
              
              storedcovmat_index = pd.MultiIndex.from_tuples(
                  [(aa, bb, np.int64(cc)) for aa, bb, cc in S_scale_var.index],
                  names=["group", "dataset", "id"],
              )
              S_scale_var = pd.DataFrame(
                  S_scale_var.values, index=storedcovmat_index, columns=storedcovmat_index
              )
              S_scale_var = S_scale_var.reindex(C.index).T.reindex(C.index)
              C = C + S_scale_var
              self.logger.info("MHO covariance matrix loaded and added")
          except FileNotFoundError:
              self.logger.info("No scale variations found")
          self._C = C
      return self._C
  
  @property
  def exp_index(self):
      """Return the experimental index."""
      if not hasattr(self, '_exp_index'):
          self.logger.info("Loading experimental index")
          self._exp_index = self.C.index
      return self._exp_index
  
  @property
  def ht_index(self):
      """Return the higher twist index."""
      if not hasattr(self, '_ht_index'):
          # Initialize MultiIndex for beta
          ht_names = []
          ht_node_idx = []

          for name, yshift in self.y_shifts.items():
            for idx_node in range(len(yshift)):
                ht_names.append(name)
                ht_node_idx.append(name + f"({idx_node})")

          # Construct the MultiIndex
          ht_index_tuple = list(zip(ht_names, ht_node_idx))
          self._ht_index = pd.MultiIndex.from_tuples(ht_index_tuple, names=["HT", "nodes"])
        
      return self._ht_index
  
  @property
  def S(self):
      """Return the theory covariance matrix."""
      if not hasattr(self, '_S'):
          self._load_pc_theory_covmat()
      return self._S
  
  @property
  def preds(self):
      """Return the predictions."""
      if not hasattr(self, '_preds'):
          self._load_predictions()
      return self._preds
  
  @property
  def dat_central(self):
      """Return the central pseudodata."""
      if not hasattr(self, '_dat_central'):
          self._load_pseudodata()
      return self._dat_central
  
  @property
  def beta(self):
      """Return the beta matrix."""
      if not hasattr(self, '_beta'):
          self._construct_beta()
      return self._beta
  
  @property
  def beta_tilde(self):
      """Return the beta tilde matrix."""
      if not hasattr(self, '_beta_tilde'):
          self._construct_beta_tilde()

      return self._beta_tilde

  @property
  def S_tilde(self):
      """Return the theory covariance matrix with beta tilde."""
      if not hasattr(self, '_S_tilde'):
          self._construct_S_tilde()
          
      return self._S_tilde
  
  @property
  def X(self):
      """Return the X matrix."""
      if not hasattr(self, '_X'):
          self._construct_X()
      return self._X

  def _load_pc_theory_covmat(self):
    """Load the theory covariance matrix."""
    self.logger.info("Loading theory covariance matrices")

    # Experimental index
    index = self.exp_index

    # Load power corrections covmat
    S_path = self.fitpath / "tables/datacuts_theory_theorycovmatconfig_point_prescriptions0_theory_covmat_custom_per_prescription.csv"
    S = pd.read_csv(S_path, index_col=[0, 1, 2], header=[0, 1, 2], 
                    sep="\t|,", engine="python")
    
    storedcovmat_index = pd.MultiIndex.from_tuples(
        [(aa, bb, np.int64(cc)) for aa, bb, cc in S.index],
        names=["group", "dataset", "id"],
    )
    S = pd.DataFrame(S.values, index=storedcovmat_index, columns=storedcovmat_index)
    self._S = S.reindex(index).T.reindex(index)

  def _load_predictions(self):
    """Load predictions and pseudodata for the TCM analysis."""
    self.logger.info("Loading predictions")

    self._preds = API.group_result_table_no_table(pdf={"from_": "fit"}, **self.common_dict)

  def _load_pseudodata(self):
    """Load pseudodata for the TCM analysis."""
    self.logger.info("Loading pseudodata")
    preds = self.preds

    pseudodata = API.read_pdf_pseudodata(**self.common_dict)
    self._dat_central = np.mean(
        [i.pseudodata.reindex(preds.index.to_list()).to_numpy().flatten() 
          for i in pseudodata],
        axis=0,
    )

  def _construct_beta_tilde(self):
    """Construct the beta vector for the TCM analysis."""
    self.logger.info("Constructing beta tilde...")

    # Construct the dataframe for beta
    beta_tilde = np.concatenate([yshift for yshift in self.y_shifts.values()])
    tmp_mat = np.zeros(shape=(len(beta_tilde), len(beta_tilde)))
    np.fill_diagonal(tmp_mat, beta_tilde)
    self._beta_tilde = pd.DataFrame(tmp_mat, index=self.ht_index, columns=self.ht_index)

  def _construct_beta(self):
    """Construct the beta matrix for the TCM analysis."""
    self.logger.info("Constructing beta...")

    groups_data_by_process = API.groups_data_by_process(**self.common_dict)
    pdf = API.pdf(pdf=self.covmat_pdf)
    shifts = {}
    for group_proc in groups_data_by_process:
        for exp_set in group_proc.datasets:
            if exp_set.name not in self.pc_excluded_exps and group_proc.name in self.pc_included_prcs:
              shifts[exp_set.name] = compute_deltas_pc(exp_set, pdf, self.pc_parameters, 'linear')
    
    # Construct the dataframe
    col_index = self.ht_index.droplevel('HT')
    col_index.name = 'shifts'
    row_index = self.exp_index
    beta = pd.DataFrame(np.zeros(shape=(row_index.size, col_index.size)), index=row_index, columns=col_index)
    beta = beta.droplevel('group')

    for exp_name in shifts.keys():
      for combs_name in shifts[exp_name].keys():
        beta.loc[(exp_name), combs_name] = shifts[exp_name][combs_name]
    self._beta = beta

    # Check if beta is constructed correctly
    S_test = np.zeros((beta.shape[0], beta.shape[0]))
    for shift in beta.columns:
      S_test += np.outer(beta[shift].to_numpy(), beta[shift].to_numpy())

    S_test = pd.DataFrame(S_test, columns=beta.index, index=beta.index)

    S_test_aligned, S_aligned = S_test.align(self.S.droplevel(0).T.droplevel(0))
    if np.allclose(S_test_aligned.to_numpy(), S_aligned.to_numpy()):
        self.logger.info("Beta construction is consistent with S.")
    else:
        raise RuntimeError("Beta construction is NOT consistent with S.")

  def _construct_S_tilde(self):
    """Construct the S_tilde matrix for the TCM analysis."""
    self.logger.info("Constructing S_tilde...")

    if not hasattr(self, '_S_tilde'):
        S_tilde = np.zeros(self.beta_tilde.shape)
        for shift in self.beta_tilde.columns:
              S_tilde += np.outer(self.beta_tilde[shift], self.beta_tilde[shift])
        self._S_tilde = pd.DataFrame(S_tilde, index=self.beta_tilde.index, columns=self.beta_tilde.columns)
    return self._S_tilde
  
  def _construct_X(self):
      """Construct the X matrix for the TCM analysis."""
      preds_onlyreplicas = self.preds.iloc[:, 2:].to_numpy()
      mean_prediction = np.mean(preds_onlyreplicas, axis=1)

      X = np.zeros((mean_prediction.shape[0], mean_prediction.shape[0]))
      for i in range(preds_onlyreplicas.shape[1]):
          X += np.outer(
              (preds_onlyreplicas[:, i] - mean_prediction),
              (preds_onlyreplicas[:, i] - mean_prediction),
          )
      X *= 1 / preds_onlyreplicas.shape[1]
      self._X = X

  def _compute_posteriors(self):
      """Compute the TCM posteriors."""
      self.logger.info("Computing TCM posteriors...")

      # Construct Shat
      S_hat = np.zeros((self.beta_tilde.shape[0], self.beta.shape[0]))
      for shift in self.beta.columns:
          S_hat += np.outer(self.beta_tilde.droplevel(level="HT", axis=1)[shift], self.beta[shift])
      S_hat = pd.DataFrame(S_hat, index=self.beta_tilde.index, columns=self.beta.index)

      central_ht_coeffs = pd.DataFrame(np.zeros(self.ht_index.shape), index=self.ht_index, columns=['central'])

      preds_onlyreplicas = self.preds.iloc[:, 2:].to_numpy()
      mean_prediction = np.mean(preds_onlyreplicas, axis=1)

      invcov = np.linalg.inv(self.C + self.S)
      delta_T_tilde = -S_hat @ invcov @ (mean_prediction - self.dat_central)
      invcov = pd.DataFrame(invcov, columns=S_hat.columns, index=S_hat.columns)
      X = pd.DataFrame(self.X, columns=S_hat.columns, index=S_hat.columns)

      self.P_tilde = S_hat @ invcov @ X @ invcov @ S_hat.T + (self.S_tilde - S_hat @ invcov @ S_hat.T)
      self.posteriors = central_ht_coeffs['central'] + delta_T_tilde


def main():
    parser = argparse.ArgumentParser(
        description="Power corrections analysis"
    )
    parser.add_argument(
        'fitname',
        type=str,
        help="Fit name to analyze"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging('DEBUG')
    logger = logging.getLogger("main")

    logger.info(f"Processing {args.fitname} fits")

    analyzer = TCMAnalyzer(args.fitname)
    success, _, error = analyzer.run_analysis()

    # Summary
    print("="*50)
    print("SUMMARY")
    print("="*50)

    if success:
        print(f"  ✓ {args.fitname}")
    else:
        print(f"  ✗ {args.fitname}: {error}")

if __name__ == "__main__":
    main()
