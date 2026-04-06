"""
Entry point for running TCM as a module.

Usage:
    python -m tcm analyse FITNAME
    python -m tcm compare config.yaml
    python -m tcm cfactors FITNAME
"""

from .cli import main

main()
