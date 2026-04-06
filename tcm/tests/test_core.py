"""
Unit tests for TCM core functionality.

These tests verify the mathematical correctness of the core algorithms
without requiring the full NNPDF infrastructure.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


class TestFluctuateWithCovariance:
    """Tests for the Cholesky-based sampling function."""

    def test_identity_covariance_preserves_statistics(self):
        """With identity covariance, samples should have unit variance."""
        from tcm.core import fluctuate_with_covariance

        n_points = 10
        covariance = np.eye(n_points)
        central = np.zeros(n_points)

        samples = fluctuate_with_covariance(
            covariance, central, n_replicas=10000, seed=42
        )

        # Mean should be close to central values
        np.testing.assert_allclose(samples.mean(axis=1), central, atol=0.1)

        # Variance should be close to 1 (diagonal of identity)
        np.testing.assert_allclose(samples.var(axis=1), np.ones(n_points), atol=0.1)

    def test_zero_covariance_returns_central(self):
        """Zero covariance should return central values unchanged."""
        from tcm.core import fluctuate_with_covariance

        n_points = 5
        covariance = np.zeros((n_points, n_points))
        central = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        samples = fluctuate_with_covariance(covariance, central, n_replicas=100)

        # All samples should equal central values
        for i in range(100):
            np.testing.assert_array_equal(samples[:, i], central)

    def test_correlation_structure_preserved(self):
        """Samples should reproduce the input correlation structure."""
        from tcm.core import fluctuate_with_covariance

        # Create a covariance with known correlation
        variances = np.array([1.0, 4.0, 9.0])
        correlation = np.array([
            [1.0, 0.8, 0.5],
            [0.8, 1.0, 0.3],
            [0.5, 0.3, 1.0],
        ])
        std = np.sqrt(variances)
        covariance = np.outer(std, std) * correlation

        central = np.zeros(3)
        samples = fluctuate_with_covariance(
            covariance, central, n_replicas=50000, seed=123
        )

        # Check that sample covariance approximates input
        sample_cov = np.cov(samples)
        np.testing.assert_allclose(sample_cov, covariance, rtol=0.1)

    def test_reproducibility_with_seed(self):
        """Same seed should produce identical results."""
        from tcm.core import fluctuate_with_covariance

        covariance = np.eye(3)
        central = np.array([1.0, 2.0, 3.0])

        samples1 = fluctuate_with_covariance(covariance, central, n_replicas=10, seed=42)
        samples2 = fluctuate_with_covariance(covariance, central, n_replicas=10, seed=42)

        np.testing.assert_array_equal(samples1, samples2)

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different samples."""
        from tcm.core import fluctuate_with_covariance

        covariance = np.eye(3)
        central = np.array([1.0, 2.0, 3.0])

        samples1 = fluctuate_with_covariance(covariance, central, n_replicas=10, seed=42)
        samples2 = fluctuate_with_covariance(covariance, central, n_replicas=10, seed=43)

        assert not np.allclose(samples1, samples2)

    @given(
        arrays(
            dtype=np.float64,
            shape=st.integers(min_value=2, max_value=10),
            elements=st.floats(min_value=-100, max_value=100, allow_nan=False),
        )
    )
    @settings(max_examples=20)
    def test_output_shape(self, central):
        """Output shape should be (n_points, n_replicas)."""
        from tcm.core import fluctuate_with_covariance

        n_points = len(central)
        n_replicas = 50
        covariance = np.eye(n_points)

        samples = fluctuate_with_covariance(covariance, central, n_replicas=n_replicas)

        assert samples.shape == (n_points, n_replicas)


class TestBuildHtIndex:
    """Tests for the MultiIndex construction helper."""

    def test_single_pc_type(self):
        """Single power correction type should produce correct index."""
        from tcm.core import _build_ht_index

        y_shifts = {"f2p": [0.1, 0.2, 0.3]}
        index = _build_ht_index(y_shifts)

        assert len(index) == 3
        assert index.names == ["HT", "nodes"]
        assert list(index.get_level_values("HT")) == ["f2p", "f2p", "f2p"]

    def test_multiple_pc_types(self):
        """Multiple power correction types should be concatenated correctly."""
        from tcm.core import _build_ht_index

        y_shifts = {
            "f2p": [0.1, 0.2],
            "Hj": [0.3, 0.4, 0.5],
        }
        index = _build_ht_index(y_shifts)

        assert len(index) == 5
        ht_values = list(index.get_level_values("HT"))
        assert ht_values == ["f2p", "f2p", "Hj", "Hj", "Hj"]

    def test_empty_input(self):
        """Empty input should return empty index."""
        from tcm.core import _build_ht_index

        y_shifts = {}
        index = _build_ht_index(y_shifts)

        assert len(index) == 0


class TestComputeSTilde:
    """Tests for prior covariance matrix construction."""

    def test_diagonal_beta_tilde(self):
        """S_tilde should equal beta_tilde @ beta_tilde.T for diagonal input."""
        from tcm.core import _compute_S_tilde

        index = pd.MultiIndex.from_tuples(
            [("f2p", "f2p(0)"), ("f2p", "f2p(1)")],
            names=["HT", "nodes"]
        )
        beta_tilde = pd.DataFrame(
            np.diag([0.5, 1.0]),
            index=index,
            columns=index
        )

        S_tilde = _compute_S_tilde(beta_tilde)

        expected = np.diag([0.25, 1.0])
        np.testing.assert_allclose(S_tilde.values, expected)


# Fixtures for integration tests (require validphys)
@pytest.fixture
def mock_fit_config():
    """Mock fit configuration for testing."""
    return {
        "fitname": "test_fit",
        "pc_parameters": {
            "f2p": {"yshift": [0.5, 0.5], "nodes": [0.01, 0.1]},
            "Hj": {"yshift": [1.0], "nodes": [0.0]},
        },
        "pc_included_procs": ["DIS NC"],
        "pc_excluded_exps": [],
        "covmat_pdf": "NNPDF40_nnlo_as_01180",
    }


class TestIntegration:
    """Integration tests that verify components work together.

    These tests use mock data and don't require the NNPDF framework.
    """

    def test_end_to_end_mock_data(self, mock_fit_config):
        """Test full pipeline with synthetic data."""
        from tcm.core import fluctuate_with_covariance

        # Create mock posteriors
        n_coeffs = 3
        posteriors = pd.Series(
            [0.1, 0.2, 0.3],
            index=pd.MultiIndex.from_tuples([
                ("f2p", "f2p(0)"),
                ("f2p", "f2p(1)"),
                ("Hj", "Hj(0)"),
            ], names=["HT", "nodes"])
        )

        # Create mock covariance
        P_tilde = pd.DataFrame(
            np.eye(n_coeffs) * 0.01,
            index=posteriors.index,
            columns=posteriors.index,
        )

        # Generate samples
        samples = fluctuate_with_covariance(
            P_tilde.values, posteriors.values, n_replicas=1000
        )

        # Verify samples have correct properties
        assert samples.shape == (n_coeffs, 1000)
        np.testing.assert_allclose(samples.mean(axis=1), posteriors.values, atol=0.05)