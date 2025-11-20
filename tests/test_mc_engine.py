"""Tests for Monte Carlo engine with parallel execution and batching."""

import numpy as np
import pytest

from stochlab.core import SimulationResult
from stochlab.mc import MonteCarloEngine, ReturnMode
from stochlab.models import MarkovChain, RandomWalk


@pytest.fixture
def simple_markov_chain():
    """Create a simple 2-state Markov chain for testing."""
    from stochlab.core.state_space import StateSpace

    ss = StateSpace(states=["A", "B"])
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    return MarkovChain(state_space=ss, P=P)


@pytest.fixture
def random_walk():
    """Create a simple random walk for testing."""
    return RandomWalk(lower_bound=-10, upper_bound=10, p=0.5)


class TestMonteCarloEngineBasic:
    """Test basic engine functionality."""

    def test_engine_creation(self, simple_markov_chain):
        """Test that engine can be created."""
        engine = MonteCarloEngine(simple_markov_chain)
        assert engine.process is simple_markov_chain

    def test_sequential_simulation(self, simple_markov_chain):
        """Test sequential simulation produces correct result."""
        engine = MonteCarloEngine(simple_markov_chain)
        result = engine.simulate(n_paths=10, T=5, seed=42, parallel=False)

        assert isinstance(result, SimulationResult)
        assert len(result.paths) == 10
        assert all(len(path) == 6 for path in result.paths)  # T+1 time steps

    def test_sequential_with_initial_state(self, simple_markov_chain):
        """Test simulation with specified initial state."""
        engine = MonteCarloEngine(simple_markov_chain)
        result = engine.simulate(n_paths=5, T=10, x0="A", seed=42)

        # All paths should start at "A"
        assert all(path.states[0] == "A" for path in result.paths)

    def test_reproducibility_with_seed(self, simple_markov_chain):
        """Test that same seed produces same results."""
        engine = MonteCarloEngine(simple_markov_chain)

        result1 = engine.simulate(n_paths=10, T=5, seed=42)
        result2 = engine.simulate(n_paths=10, T=5, seed=42)

        # Should produce identical paths
        for p1, p2 in zip(result1.paths, result2.paths):
            assert np.array_equal(p1.states, p2.states)

    def test_different_seeds_produce_different_results(self, simple_markov_chain):
        """Test that different seeds produce different results."""
        engine = MonteCarloEngine(simple_markov_chain)

        result1 = engine.simulate(n_paths=100, T=10, seed=42)
        result2 = engine.simulate(n_paths=100, T=10, seed=123)

        # Should be different (with very high probability)
        same_count = sum(
            1
            for p1, p2 in zip(result1.paths, result2.paths)
            if np.array_equal(p1.states, p2.states)
        )
        assert same_count < 10  # Less than 10% identical paths

    def test_invalid_n_paths(self, simple_markov_chain):
        """Test that invalid n_paths raises error."""
        engine = MonteCarloEngine(simple_markov_chain)

        with pytest.raises(ValueError, match="must be positive"):
            engine.simulate(n_paths=0, T=10)

        with pytest.raises(ValueError, match="must be positive"):
            engine.simulate(n_paths=-5, T=10)


class TestMonteCarloParallel:
    """Test parallel execution."""

    def test_parallel_simulation(self, simple_markov_chain):
        """Test that parallel simulation works."""
        engine = MonteCarloEngine(simple_markov_chain)
        result = engine.simulate(n_paths=100, T=10, seed=42, parallel=True, n_jobs=2)

        assert isinstance(result, SimulationResult)
        assert len(result.paths) == 100
        assert result.metadata["parallel"] is True
        assert result.metadata["n_jobs"] == 2

    def test_parallel_reproducibility(self, simple_markov_chain):
        """Test that parallel execution with same seed is reproducible."""
        engine = MonteCarloEngine(simple_markov_chain)

        result1 = engine.simulate(n_paths=50, T=10, seed=42, parallel=True, n_jobs=2)
        result2 = engine.simulate(n_paths=50, T=10, seed=42, parallel=True, n_jobs=2)

        # Results should be identical (same seed)
        # Note: order might differ, so compare sorted
        states1 = sorted([tuple(p.states) for p in result1.paths])
        states2 = sorted([tuple(p.states) for p in result2.paths])

        assert states1 == states2

    def test_parallel_vs_sequential_same_seed(self, simple_markov_chain):
        """Test that parallel and sequential are both reproducible (may differ in results)."""
        engine = MonteCarloEngine(simple_markov_chain)

        result_seq = engine.simulate(n_paths=20, T=10, seed=42, parallel=False)
        result_par = engine.simulate(n_paths=20, T=10, seed=42, parallel=True)

        # Both should have correct number of paths
        assert len(result_seq.paths) == 20
        assert len(result_par.paths) == 20

        # Both should be reproducible when run again with same seed
        result_seq2 = engine.simulate(n_paths=20, T=10, seed=42, parallel=False)
        result_par2 = engine.simulate(n_paths=20, T=10, seed=42, parallel=True)

        # Sequential should match sequential
        states_seq1 = sorted([tuple(p.states) for p in result_seq.paths])
        states_seq2 = sorted([tuple(p.states) for p in result_seq2.paths])
        assert states_seq1 == states_seq2

        # Parallel should match parallel
        states_par1 = sorted([tuple(p.states) for p in result_par.paths])
        states_par2 = sorted([tuple(p.states) for p in result_par2.paths])
        assert states_par1 == states_par2

    def test_parallel_with_different_n_jobs(self, simple_markov_chain):
        """Test that different n_jobs values all work correctly."""
        engine = MonteCarloEngine(simple_markov_chain)

        for n_jobs in [1, 2, 4, -1]:
            result = engine.simulate(
                n_paths=20, T=5, seed=42, parallel=True, n_jobs=n_jobs
            )
            assert len(result.paths) == 20


class TestReturnModes:
    """Test different return modes."""

    def test_paths_mode(self, simple_markov_chain):
        """Test paths mode returns full Path objects."""
        engine = MonteCarloEngine(simple_markov_chain)
        result = engine.simulate(n_paths=10, T=5, mode="paths", seed=42)

        assert len(result.paths) == 10
        assert all(hasattr(path, "states") for path in result.paths)
        assert all(hasattr(path, "times") for path in result.paths)

    def test_values_mode(self, simple_markov_chain):
        """Test values mode (lighter than full paths)."""
        engine = MonteCarloEngine(simple_markov_chain)
        result = engine.simulate(n_paths=10, T=5, mode="values", seed=42, parallel=True)

        # Should still return paths (compatibility), but they're minimal
        assert len(result.paths) == 10

    def test_stats_mode(self, simple_markov_chain):
        """Test stats mode (lightest)."""
        engine = MonteCarloEngine(simple_markov_chain)
        result = engine.simulate(n_paths=10, T=5, mode="stats", seed=42, parallel=True)

        # Stats mode doesn't return full paths
        assert "statistics" in result.metadata or len(result.paths) == 0

    def test_invalid_mode(self, simple_markov_chain):
        """Test that invalid mode raises error."""
        engine = MonteCarloEngine(simple_markov_chain)

        with pytest.raises(ValueError, match="Invalid mode"):
            engine.simulate(n_paths=10, T=5, mode="invalid")


class TestBatching:
    """Test batching behavior."""

    def test_custom_batch_size(self, simple_markov_chain):
        """Test that custom batch size is respected."""
        engine = MonteCarloEngine(simple_markov_chain)
        result = engine.simulate(
            n_paths=100, T=5, parallel=True, batch_size=10, seed=42
        )

        # Should have ~10 batches (100 paths / 10 per batch)
        assert result.metadata.get("n_batches") >= 8
        assert result.metadata.get("n_batches") <= 12

    def test_auto_batch_size(self, simple_markov_chain):
        """Test that automatic batch sizing works."""
        engine = MonteCarloEngine(simple_markov_chain)
        result = engine.simulate(
            n_paths=1000, T=5, parallel=True, batch_size=None, seed=42
        )

        # Should automatically compute reasonable batch size
        assert "n_batches" in result.metadata
        assert result.metadata["n_batches"] > 0


class TestEstimate:
    """Test Monte Carlo estimation with custom estimators."""

    def test_estimate_probability(self, simple_markov_chain):
        """Test estimating probability using estimate method."""
        engine = MonteCarloEngine(simple_markov_chain)

        # Estimate P(final state = "B")
        def final_is_b(path):
            return 1.0 if path.states[-1] == "B" else 0.0

        stats = engine.estimate(
            estimator_fn=final_is_b,
            n_paths=1000,
            T=10,
            seed=42,
            parallel=False,
        )

        # Should have computed statistics
        assert stats.mean is not None
        assert stats.std is not None
        assert stats.stderr is not None
        assert stats.confidence_interval is not None

        # Mean should be between 0 and 1 (it's a probability)
        assert 0 <= stats.mean <= 1

        # Confidence interval should contain mean
        ci_lower, ci_upper = stats.confidence_interval
        assert ci_lower <= stats.mean <= ci_upper

    def test_estimate_with_parallel(self, random_walk):
        """Test estimate method with parallel execution."""
        engine = MonteCarloEngine(random_walk)

        # Estimate expected final position
        def final_position(path):
            return float(path.states[-1])

        stats = engine.estimate(
            estimator_fn=final_position,
            n_paths=1000,
            T=100,
            seed=42,
            parallel=True,
            n_jobs=2,
        )

        assert stats.n_paths == 1000
        assert stats.mean is not None

        # For symmetric random walk, expected final position should be near 0
        assert abs(stats.mean) < 5  # Should be close to 0 with high probability


class TestIntegrationWithModels:
    """Test integration with different process models."""

    def test_with_random_walk(self, random_walk):
        """Test engine works with RandomWalk."""
        engine = MonteCarloEngine(random_walk)
        result = engine.simulate(n_paths=20, T=10, seed=42)

        assert len(result.paths) == 20
        assert all(len(path) == 11 for path in result.paths)

    def test_with_random_walk_parallel(self, random_walk):
        """Test parallel simulation with RandomWalk."""
        engine = MonteCarloEngine(random_walk)
        result = engine.simulate(n_paths=100, T=10, seed=42, parallel=True, n_jobs=2)

        assert len(result.paths) == 100

    def test_metadata_includes_process_type(self, simple_markov_chain):
        """Test that metadata includes process type."""
        engine = MonteCarloEngine(simple_markov_chain)
        result = engine.simulate(n_paths=10, T=5)

        assert "process_type" in result.metadata
        assert result.metadata["process_type"] == "MarkovChain"


class TestSeeding:
    """Test seed management and reproducibility."""

    def test_seed_independence(self, simple_markov_chain):
        """Test that paths with different seeds are independent."""
        from stochlab.mc.seeding import generate_path_seeds

        seeds = generate_path_seeds(base_seed=42, n_paths=100)

        # All seeds should be different
        assert len(set(seeds)) == 100

        # Seeds should be reproducible
        seeds2 = generate_path_seeds(base_seed=42, n_paths=100)
        assert seeds == seeds2

    def test_batch_seed_generation(self, simple_markov_chain):
        """Test that batch seeds are properly generated."""
        from stochlab.mc.seeding import generate_batch_seeds

        batch_seeds = generate_batch_seeds(base_seed=42, n_batches=10)

        assert len(batch_seeds) == 10
        assert len(set(batch_seeds)) == 10  # All unique


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_paths_raises_error(self, simple_markov_chain):
        """Test that n_paths=0 raises appropriate error."""
        engine = MonteCarloEngine(simple_markov_chain)

        with pytest.raises(ValueError):
            engine.simulate(n_paths=0, T=10)

    def test_negative_paths_raises_error(self, simple_markov_chain):
        """Test that negative n_paths raises error."""
        engine = MonteCarloEngine(simple_markov_chain)

        with pytest.raises(ValueError):
            engine.simulate(n_paths=-10, T=10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
