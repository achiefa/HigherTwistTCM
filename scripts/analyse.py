import argparse
import logging
from typing import List, Tuple, Optional
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import lhapdf
lhapdf.setVerbosity(0)  # Suppress lhapdf output

import pandas as pd
import numpy as np

from TCMAnalyser import TCMAnalyzer, setup_logging, SAVEDIR, fluctuate_points_cholesky, HT_PLOT_SPEC

def process_single_fit(fitname: str, log_level: str, force: bool) -> Tuple[bool, str, str]:
    """Process a single fit (for multiprocessing)"""

    setup_logging(log_level)
    analyzer = TCMAnalyzer(fitname)
    success, _, error = analyzer.run_analysis(force=force)
    return success, fitname, error

def batch_process_with_monitoring(fitargs: List[Tuple[str, str]], max_workers: int = 4, force: bool = False):
    """
    Process multiple fits with real-time monitoring and progress tracking
    """
    setup_logging('INFO')
    
    results = {}
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_fit = {
            executor.submit(process_single_fit, fit, log_level, force): fit
            for fit, log_level in fitargs
        }
        
        # Process completed jobs
        completed = 0
        total = len(fitargs)

        for future in as_completed(future_to_fit):
            fit = future_to_fit[future]
            completed += 1
            
            try:
                success, _, error = future.result()
                results[fit] = {'success': success, 'error': error}
                
                # Progress update
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                eta = avg_time * (total - completed)
                
                status = "✓" if success else "✗"
                print(f"[{completed}/{total}] {status} {fit} "
                      f"(elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s)")
                
            except Exception as exc:
                results[fit] = {'success': False, 'error': str(exc)}
                print(f"[{completed}/{total}] ✗ {fit} failed with exception: {exc}")
    
    return results

def generate_summary_report(output_file: str = "summary_report.html"):
    """
    Generate an HTML summary report for all processed fits
    """
    from datetime import datetime
    
    html_parts = [
        f"<html><head><title>HT Analysis Summary - {datetime.now()}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #4CAF50; color: white; }",
        "tr:nth-child(even) { background-color: #f2f2f2; }",
        ".success { color: green; } .failed { color: red; }",
        "img { max-width: 100%; height: auto; margin: 10px 0; }",
        "</style></head><body>",
        f"<h1>Higher Twist TCM Analysis Summary</h1>",
        f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        "<table><tr><th>Fit Name</th><th>Status</th><th>Plots</th></tr>"
    ]
    
    fit_dirs = [d for d in SAVEDIR.iterdir() if d.is_dir()]
    fit_dirs.sort(key=lambda x: x.name)  # Sort by directory name
    for fit_dir in fit_dirs:
        status = "success" if (fit_dir / "posteriors.pkl").exists() else "failed"
        status_class = status
        
        plots_html = ""
        for plot in ["heatmap.png", "scatter.png", "log_scale.png", "linear_scale.png"]:
            if (fit_dir / plot).exists():
                absolute_path = fit_dir.resolve() / plot
                encoded_path = str(absolute_path).replace(" ", "%20")
                plots_html += f'<a href="file://{encoded_path}" target="_blank">{plot}</a><br>'
        html_parts.append(
            f"<tr><td>{fit_dir.name}</td>"
            f"<td class='{status_class}'>{status.upper()}</td>"
            f"<td>{plots_html}</td></tr>"
        )
    
    html_parts.extend(["</table>", "</body></html>"])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(html_parts))
    
    print(f"Summary report saved to {output_file}")


def compare_multiple_fits(yaml_file: str,
                          custom_keys: Optional[List[List[str]]] = None,
                          output_dir: str = "./Comparison"):
    """
    Compare results from multiple fits and create comparison plots
    """
    from validphys.api import API
    from collections import defaultdict
    from matplotlib import pyplot as plt
    from  matplotlib import rc
    rc('text',usetex=True)
    import yaml

    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    fitnames = config.get('fitnames', [])
    if not fitnames:
        raise ValueError("No fit names provided in the configuration file.")
    fit_labels = config.get('fit_labels', None)
    colors = config.get('colors', None)
    hatchs = config.get('hatchs', None)

    output_path = Path(output_dir + "/" + "_VS_".join(fitnames))
    output_path.mkdir(exist_ok=True, parents=True)

    x_nodes = {}
    posteriors_dict = {}
    P_tilde_dict = {}
    for fitname in fitnames:
      thcovmat_dict = API.fit(fit=fitname).as_input()["theorycovmatconfig"]
      ht_parameters = thcovmat_dict['pc_parameters']
      x_nodes_dict = defaultdict(list)
      for pc_name, pc_par in ht_parameters.items():
        x_nodes_dict[pc_name] = pc_par['nodes']
      x_nodes[fitname] = x_nodes_dict

      try:
        posteriors_dict[fitname] = pd.read_pickle(SAVEDIR / f"{fitname}/posteriors.pkl")
        P_tilde_dict[fitname] = pd.read_pickle(SAVEDIR / f"{fitname}/P_tilde.pkl")
      except FileNotFoundError as e:
        print(e)
    
    # Allow only common keys
    common_keys = list(set.intersection(*[set(x_nodes[fitname].keys()) for fitname in fitnames]))

    def make_plot(key1, key2=None):
        legends = []
        legend_names = []

        fig, ax = plt.subplots(figsize=(10, 5))
        if key2 is None:
            keys = [key1, key1]
        else:
            keys = [key1, key2]

        ylabel = HT_PLOT_SPEC[key1]["y_label"]
        xlabel = HT_PLOT_SPEC[key1]["x_label"]

        xaxis = x_nodes[fitnames[0]][key1]

        for fit_idx, fitname in enumerate(fitnames):
          posteriors = posteriors_dict[fitname]
          P_tilde = P_tilde_dict[fitname]

          replicas = fluctuate_points_cholesky(P_tilde.to_numpy(), posteriors.to_numpy(), replicas=10000, seed=2143123)
          mean = pd.Series(replicas.mean(axis=1), index=posteriors.index)
          std = pd.Series(replicas.std(axis=1), index=posteriors.index)

          shift_central = mean.xs(level='HT', key=keys[fit_idx]).to_numpy()
          shift_std = std.xs(level='HT', key=keys[fit_idx]).to_numpy()

          color = colors[fit_idx] if colors else None
          hatch = hatchs[fit_idx] if hatchs else None

          pl = ax.plot(xaxis, shift_central, ls="-", lw=1, color=color)
          ax.scatter(xaxis, shift_central, color=color)
          pl_lg = ax.fill(np.NaN, np.NaN, alpha=0.3, color=pl[0].get_color())
          pl_fb = ax.fill_between(xaxis, shift_central - shift_std, 
                                shift_central + shift_std, 
                                color=pl[0].get_color(), alpha=0.3, hatch=hatch)
          legends.append((pl[0], pl_lg[0]))
          legend_names.append(fit_labels[fit_idx])
        
        ax.plot(xaxis, np.zeros_like(xaxis), ls="dashed", lw=2, 
                color="black", alpha=0.6)
        
        ax.set_xlabel(xlabel, fontsize=30)
        ax.set_ylabel(ylabel, fontsize=30)
        ax.set_xscale('log')

        # Legend labels
        ax.legend(legends, legend_names, loc='best', fontsize=20)

        file_name = f"{key1}_vs_{key2}" if key2 != None else f"{key1}"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(output_path / f"{file_name}_log_scale.png")
        ax.set_xscale('linear')
        fig.savefig(output_path /  f"{file_name}_linear_scale.png")

    for key in common_keys:
        make_plot(key)

    # Handle custom keys
    for custom_key_pair in (custom_keys or []):
        make_plot(*custom_key_pair)

    # Copy config file to output directory
    with open(output_path / "config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(
        description="Power corrections analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparser = parser.add_subparsers(dest='command', help='Available commands', required=True)

    analyse_parser = subparser.add_parser('analyse', help='Run TCM analysis on specified fits')
    analyse_parser.add_argument(
        'fitnames',
        nargs='+',
        help="Fit name(s) to analyze"
    )
    analyse_parser.add_argument(
          '-j', '--jobs', 
          type=int, 
          default=1,
          help='Number of parallel jobs (default: number of CPU cores)'
      )
    analyse_parser.add_argument(
          '--log-level',
          default='INFO',
          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
          help='Logging level'
      )
    analyse_parser.add_argument(
          '--sequential',
          action='store_true',
          help='Run sequentially instead of in parallel'
      )
    analyse_parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Force re-analysis of fits even if results already exist'
    )
    
    compare_parser = subparser.add_parser('compare', help='Compare multiple fits and generate comparison plots')
    compare_parser.add_argument(
        'yaml_file',
        type=str,
        help="YAML configuration file with fit names and options"
    )
    compare_parser.add_argument(
        '-k', '--custom-keys',
        nargs='*',
        type=str,
        help="Key pairs to compare (e.g., 'key1,key2 key1,key3')"
    )
    compare_parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='./Comparison',
        help="Directory to save comparison plots"
    )

    subparser.add_parser('summary', help='Only generate summary report without running analysis')
    
    args = parser.parse_args()

    if args.command == 'summary':
        # Generate summary report without running analysis
        generate_summary_report()

    if args.command == 'compare':
        # Parser the custom keys into pairs
        custom_keys = None
        if args.custom_keys:
            custom_keys = []
            for pair in args.custom_keys:
                keys = pair.split(',')
                if len(keys) == 2:
                    custom_keys.append(keys)
                else:
                    raise ValueError(f"Invalid key pair: {pair}. Expected format 'key1,key2'.")
        # Compare multiple fits
        compare_multiple_fits(args.yaml_file, 
                              custom_keys=custom_keys,
                              output_dir=args.output_dir)
        
    if args.command == 'analyse':
        setup_logging(args.log_level)
        logger = logging.getLogger("TCM analysis")
        logger.info(f"Processing {len(args.fitnames)} fits")

        # Prepare arguments for each fit
        fit_args = [(fit, args.log_level) for fit in args.fitnames]

        if args.jobs == 1 or args.sequential:
            # Sequential processing
            logger.info("Running in sequential mode. Single process.")

            results = {}
            for fitname, log_level in fit_args:
                result = process_single_fit(fitname, log_level, args.force)
                results[fitname] = {
                    'success': result[0],
                    'error': result[2]
                }

        else:
            max_workers = args.jobs or 4
            logger.info(f"Using {max_workers} parallel processes")

            results = batch_process_with_monitoring(fit_args, max_workers, args.force)

        # Summary
        print("\n"+ "="*50)
        print("SUMMARY")
        print("="*50)

        successful = []
        failed = []

        for fitname, value in results.items():
            if value['success']:
                successful.append(fitname)
            else:
                failed.append((fitname, value['error']))

        logger.info(f"Successful: {len(successful)}/{len(args.fitnames)}")
        for fit in successful:
            logger.info(f"  ✓ {fit}")
        
        if failed:
            logger.error(f"Failed: {len(failed)}/{len(args.fitnames)}")
            for fit, error in failed:
                logger.error(f"  ✗ {fit}: {error}")
            sys.exit(1)
        else:
            logger.info("All fits processed successfully!")

    generate_summary_report()
    
   
if __name__ == "__main__":
  main()