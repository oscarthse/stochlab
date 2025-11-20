"""Tests for Monte Carlo seeding utilities."""

import numpy as np
import pytest

from stochlab.mc.seeding import generate_batch_seeds, generate_path_seeds, make_rng


class TestGeneratePathSeeds:
    """Test path seed generation with SeedSequence."""

    def test_generates_correct_number(self):
        """Test that correct number of seeds is generated."""
        seeds = generate_path_seeds(base_seed=42, n_paths=10)
        assert len(seeds) == 10

    def test_seeds_are_unique(self):
        """Test that all generated seeds are unique."""
        seeds = generate_path_seeds(base_seed=42, n_paths=100)
        assert len(set(seeds)) == 100  # All seeds should be different

    def test_reproducibility(self):
        """Test that same base seed produces same path seeds."""
        seeds1 = generate_path_seeds(base_seed=42, n_paths=50)
        seeds2 = generate_path_seeds(base_seed=42, n_paths=50)
        assert seeds1 == seeds2

    def test_different_base_seeds(self):
        """Test that different base seeds produce different sequences."""
        seeds1 = generate_path_seeds(base_seed=42, n_paths=50)
        seeds2 = generate_path_seeds(base_seed=123, n_paths=50)
        assert seeds1 != seeds2

    def test_seeds_are_integers(self):
        """Test that all seeds are valid integers."""
        seeds = generate_path_seeds(base_seed=42, n_paths=10)
        assert all(isinstance(s, (int, np.integer)) for s in seeds)

    def test_invalid_n_paths(self):
        """Test that invalid n_paths raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            generate_path_seeds(base_seed=42, n_paths=0)

        with pytest.raises(ValueError, match="must be positive"):
            generate_path_seeds(base_seed=42, n_paths=-5)

    def test_large_n_paths(self):
        """Test that large number of paths works."""
        seeds = generate_path_seeds(base_seed=42, n_paths=10000)
        assert len(seeds) == 10000
        assert len(set(seeds)) == 10000  # All unique


class TestGenerateBatchSeeds:
    """Test batch seed generation."""

    def test_generates_correct_number(self):
        """Test that correct number of batch seeds generated."""
        seeds = generate_batch_seeds(base_seed=42, n_batches=5)
        assert len(seeds) == 5

    def test_batch_seeds_are_unique(self):
        """Test that batch seeds are unique."""
        seeds = generate_batch_seeds(base_seed=42, n_batches=20)
        assert len(set(seeds)) == 20

    def test_reproducibility(self):
        """Test that batch seeds are reproducible."""
        seeds1 = generate_batch_seeds(base_seed=42, n_batches=10)
        seeds2 = generate_batch_seeds(base_seed=42, n_batches=10)
        assert seeds1 == seeds2


class TestMakeRNG:
    """Test RNG creation."""

    def test_creates_generator(self):
        """Test that make_rng creates a Generator."""
        rng = make_rng(seed=42)
        assert isinstance(rng, np.random.Generator)

    def test_reproducibility(self):
        """Test that same seed produces reproducible results."""
        rng1 = make_rng(seed=42)
        rng2 = make_rng(seed=42)

        vals1 = rng1.random(10)
        vals2 = rng2.random(10)

        assert np.allclose(vals1, vals2)

    def test_different_seeds_produce_different_values(self):
        """Test that different seeds produce different values."""
        rng1 = make_rng(seed=42)
        rng2 = make_rng(seed=123)

        vals1 = rng1.random(100)
        vals2 = rng2.random(100)

        # Should be different
        assert not np.allclose(vals1, vals2)


class TestSeedSequenceProperties:
    """Test statistical properties of seed generation."""

    def test_statistical_independence(self):
        """
        Test that seeds from SeedSequence are statistically independent.

        This is a basic test - true independence requires more sophisticated tests.
        """
        seeds = generate_path_seeds(base_seed=42, n_paths=1000)

        # Create RNGs and generate values
        rngs = [make_rng(s) for s in seeds[:100]]
        values = [rng.random() for rng in rngs]

        # Check that values are reasonably distributed
        # (not clustered, which would suggest correlation)
        mean = np.mean(values)
        assert 0.4 < mean < 0.6  # Should be near 0.5 for uniform [0, 1]

        std = np.std(values)
        expected_std = 1 / np.sqrt(12)  # Theoretical for uniform [0, 1]
        assert 0.2 < std < 0.4  # Should be near expected_std (~0.289)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
