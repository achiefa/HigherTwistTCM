"""
Unit tests for TCM core functionality.

These tests verify the mathematical correctness of the core algorithms
without requiring the full NNPDF infrastructure.
"""

from pathlib import Path

import numpy as np
import pytest

from validphys.fkparser import parse_cfactor

from tcm.io import load_fit_config
from tcm.cfactors import produce_cfactors

FITNAME = "250814-ac-01-baseline-lowcuts-nnlo"
CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "data/FULL_TEST_SIMPLE"
CFAC_REFERENCE_DIR = DATA_DIR / "cfactors"
OUT_DIR = CURRENT_DIR / "output"


@pytest.fixture(scope="session")
def produced_folder():
    fit_config = load_fit_config(FITNAME)
    return produce_cfactors(fit_config, DATA_DIR, OUT_DIR, "PCNNLO")


@pytest.mark.parametrize(
    "reference_file",
    list(CFAC_REFERENCE_DIR.glob("*.dat")),
    ids=lambda p: p.name,
)
def test_cfactor_file(produced_folder, reference_file):
    output_file = Path(produced_folder) / reference_file.name
    assert output_file.exists(), f"Output file {output_file} not found"

    output = parse_cfactor(output_file.open("rb"))
    reference = parse_cfactor(reference_file.open("rb"))

    np.testing.assert_allclose(output.central_value, reference.central_value, atol=1e-6)
