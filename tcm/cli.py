#!/usr/bin/env python
"""
Command-line interface for TCM analysis.

This module provides the main entry points for:
- Running TCM analysis on fits
- Comparing results between fits
- Producing C-factors from posteriors

Usage:
    python -m tcm.cli analyse FIT1 FIT2 ...
    python -m tcm.cli compare config.yaml
    python -m tcm.cli cfactors FITNAME
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Suppress noisy loggers
logging.getLogger("reportengine").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

# Try to suppress LHAPDF output
try:
    import lhapdf

    lhapdf.setVerbosity(0)
except ImportError:
    pass

logger = logging.getLogger("tcm")


# Default paths
DEFAULT_RESULTS_DIR = Path(__file__).parent.parent / "Results"
DEFAULT_CFACTORS_DIR = Path(__file__).parent.parent.parent / "cfactors"
DEFAULT_COMPARISONS_DIR = Path(__file__).parent.parent.parent / "Comparisons"


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def run_analysis(
    fitname: str,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    force: bool = False,
    produce_plots: bool = True,
) -> Tuple[bool, str, Optional[str]]:
    """
    Run TCM analysis for a single fit.

    Parameters
    ----------
    fitname : str
        Name of the fit to analyse.
    results_dir : Path
        Directory to save results.
    force : bool
        Force recomputation even if results exist.
    produce_plots : bool
        Whether to generate plots.

    Returns
    -------
    tuple
        (success: bool, fitname: str, error: Optional[str])
    """
    from .core import (
        compute_posteriors,
        compute_posterior_covariance,
        fluctuate_with_covariance,
    )
    from .io import (
        load_fit_config,
        load_covariance_matrix,
        save_results,
        load_results,
    )
    from .plotting import (
        plot_covariance_heatmap,
        plot_scatter_validation,
        save_all_posteriors,
    )
    import pandas as pd

    output_dir = results_dir / fitname
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Check for existing results
        if not force and (output_dir / "posteriors.pkl").exists():
            logger.info(f"Results exist for {fitname}, loading from disk")
            posteriors, P_tilde = load_results(output_dir)
        else:
            logger.info(f"Running TCM analysis for {fitname}")

            # Load fit configuration and data
            config = load_fit_config(fitname)
            C, S = load_covariance_matrix(fitname)

            # Compute posteriors
            posteriors, P_tilde = compute_posteriors(config, C, S)

            # Compute additional covariance matrices
            from .core import _compute_replica_covariance

            X = _compute_replica_covariance(fitname)
            Cpost = compute_posterior_covariance(S, C, X)

            # Save results
            save_results(output_dir, posteriors, P_tilde, Cpost)

        # Generate plots
        if produce_plots:
            logger.info(f"Generating plots for {fitname}")

            config = load_fit_config(fitname)

            # Generate fluctuated samples for uncertainty visualization
            replicas = fluctuate_with_covariance(
                P_tilde.to_numpy(),
                posteriors.to_numpy(),
                n_replicas=10000,
                seed=2143123,
            )
            mean = pd.Series(replicas.mean(axis=1), index=posteriors.index)
            std = pd.Series(replicas.std(axis=1), index=posteriors.index)

            # Covariance heatmap
            fig = plot_covariance_heatmap(P_tilde)
            fig.tight_layout()
            fig.savefig(output_dir / "heatmap.png", dpi=150)

            # Scatter validation plot
            fig = plot_scatter_validation(posteriors.to_numpy(), mean.to_numpy())
            fig.savefig(output_dir / "scatter.png", dpi=150)

            # Posterior plots
            save_all_posteriors(mean, std, config["nodes"], output_dir)

        logger.info(f"Analysis completed for {fitname}")
        return True, fitname, None

    except Exception as e:
        logger.error(f"Analysis failed for {fitname}: {e}")
        return False, fitname, str(e)


def run_comparison(
    config_file: Path,
    output_dir: Path = DEFAULT_COMPARISONS_DIR,
    result_name: Optional[str] = None,
) -> None:
    """
    Compare TCM results from multiple fits.

    Parameters
    ----------
    config_file : Path
        YAML file with fit names and plot options.
    output_dir : Path
        Directory to save comparison plots.
    result_name : str, optional
        Custom name for the output directory.
    """
    import pandas as pd
    from .io import load_results
    from .core import fluctuate_with_covariance
    from .plotting import plot_comparison, PLOT_SPECS
    from validphys.api import API

    with open(config_file) as f:
        config = yaml.safe_load(f)

    fitnames = config.get("fitnames", [])
    labels = config.get("fit_labels", fitnames)
    colors = config.get("colors", [f"C{i}" for i in range(len(fitnames))])
    hatches = config.get("hatchs", [None] * len(fitnames))

    if not fitnames:
        raise ValueError("No fit names in configuration file")

    # Setup output directory
    if result_name:
        comparison_dir = output_dir / result_name
    else:
        comparison_dir = output_dir / "_VS_".join(fitnames)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Load results and nodes for each fit
    results = {}
    nodes = {}

    for fitname in fitnames:
        logger.info(f"Loading results for {fitname}")

        posteriors, P_tilde = load_results(DEFAULT_RESULTS_DIR / fitname)

        # Generate samples for uncertainty
        replicas = fluctuate_with_covariance(
            P_tilde.to_numpy(), posteriors.to_numpy(), n_replicas=10000, seed=2143123
        )
        mean = pd.Series(replicas.mean(axis=1), index=posteriors.index)
        std = pd.Series(replicas.std(axis=1), index=posteriors.index)

        results[fitname] = (mean, std)

        # Get nodes from fit config
        fit_dict = API.fit(fit=fitname).as_input()
        pc_params = fit_dict["theorycovmatconfig"]["pc_parameters"]
        nodes[fitname] = {name: params["nodes"] for name, params in pc_params.items()}

    # Find common PC types across all fits
    common_types = set.intersection(*[set(n.keys()) for n in nodes.values()])

    # Generate comparison plots
    for pc_type in common_types:
        if pc_type not in PLOT_SPECS:
            continue

        logger.info(f"Generating comparison plot for {pc_type}")

        fig, ax = plot_comparison(
            results, nodes, pc_type, labels=labels, colors=colors, hatches=hatches
        )

        fig.tight_layout()
        fig.savefig(comparison_dir / f"{pc_type}_log_scale.png", dpi=150)

        ax.set_xscale("linear")
        fig.savefig(comparison_dir / f"{pc_type}_linear_scale.png", dpi=150)

    # Save config copy
    with open(comparison_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    logger.info(f"Comparison plots saved to {comparison_dir}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="TCM: Theory Covariance Method for Power Corrections",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Analyse command
    analyse = subparsers.add_parser("analyse", help="Run TCM analysis on fits")
    analyse.add_argument("fitnames", nargs="+", help="Fit names to analyse")
    analyse.add_argument("-j", "--jobs", type=int, default=1, help="Parallel jobs")
    analyse.add_argument("-f", "--force", action="store_true", help="Force recompute")
    analyse.add_argument("--log-level", default="INFO", help="Logging level")
    analyse.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_RESULTS_DIR, help="Output directory"
    )

    # Compare command
    compare = subparsers.add_parser("compare", help="Compare multiple fits")
    compare.add_argument("config", type=Path, help="YAML configuration file")
    compare.add_argument("-n", "--name", help="Result directory name")
    compare.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_COMPARISONS_DIR, help="Output directory"
    )

    # C-factors command
    cfactors = subparsers.add_parser("cfactors", help="Produce C-factors")
    cfactors.add_argument("fitname", help="Fit name")
    cfactors.add_argument(
        "--posteriors", type=Path, default=DEFAULT_RESULTS_DIR, help="Posteriors directory"
    )
    cfactors.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_CFACTORS_DIR, help="Output directory"
    )
    cfactors.add_argument("--label", default="PC", help="C-factor label")

    args = parser.parse_args()
    setup_logging(getattr(args, "log_level", "INFO"))

    if args.command == "analyse":
        results = {}

        if args.jobs > 1 and len(args.fitnames) > 1:
            # Parallel processing
            logger.info(f"Running {len(args.fitnames)} fits with {args.jobs} workers")

            with ProcessPoolExecutor(max_workers=args.jobs) as executor:
                futures = {
                    executor.submit(run_analysis, fit, args.output, args.force): fit
                    for fit in args.fitnames
                }

                for future in as_completed(futures):
                    success, fitname, error = future.result()
                    results[fitname] = {"success": success, "error": error}
                    status = "OK" if success else f"FAILED: {error}"
                    print(f"  {fitname}: {status}")
        else:
            # Sequential processing
            for fitname in args.fitnames:
                success, _, error = run_analysis(fitname, args.output, args.force)
                results[fitname] = {"success": success, "error": error}

        # Summary
        successful = sum(1 for r in results.values() if r["success"])
        print(f"\nCompleted: {successful}/{len(args.fitnames)} successful")

        if successful < len(args.fitnames):
            sys.exit(1)

    elif args.command == "compare":
        run_comparison(args.config, args.output, args.name)

    elif args.command == "cfactors":
        from .cfactors import produce_cfactors

        produce_cfactors(args.fitname, args.posteriors, args.output, args.label)


if __name__ == "__main__":
    main()
