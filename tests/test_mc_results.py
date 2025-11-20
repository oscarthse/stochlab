"""Tests for Monte Carlo result types."""

import numpy as np
import pytest

from stochlab.core.simulation import Path
from stochlab.mc.results import (
    BatchResult,
    MCStatistics,
    ReturnMode,
    merge_batch_results,
)


class TestReturnMode:
    """Test ReturnMode enum."""

    def test_mode_values(self):
        """Test that mode values are correct."""
        assert ReturnMode.PATHS.value == "paths"
        assert ReturnMode.VALUES.value == "values"
        assert ReturnMode.STATS.value == "stats"

    def test_mode_from_string(self):
        """Test creating mode from string."""
        assert ReturnMode("paths") == ReturnMode.PATHS
        assert ReturnMode("values") == ReturnMode.VALUES
        assert ReturnMode("stats") == ReturnMode.STATS


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_paths_mode_creation(self):
        """Test creating BatchResult in PATHS mode."""
        paths = [
            Path(times=np.array([0, 1]), states=np.array(["A", "B"])),
            Path(times=np.array([0, 1]), states=np.array(["A", "A"])),
        ]

        result = BatchResult(
            batch_id=0, n_paths=2, mode=ReturnMode.PATHS, paths=paths
        )

        assert result.batch_id == 0
        assert result.n_paths == 2
        assert result.mode == ReturnMode.PATHS
        assert len(result.paths) == 2

    def test_values_mode_creation(self):
        """Test creating BatchResult in VALUES mode."""
        values = np.array(["A", "B", "A"])

        result = BatchResult(
            batch_id=1, n_paths=3, mode=ReturnMode.VALUES, values=values
        )

        assert result.mode == ReturnMode.VALUES
        assert result.values is not None
        assert len(result.values) == 3

    def test_stats_mode_creation(self):
        """Test creating BatchResult in STATS mode."""
        stats = {"mean": 0.5, "std": 0.2, "n": 100}

        result = BatchResult(
            batch_id=2, n_paths=100, mode=ReturnMode.STATS, partial_stats=stats
        )

        assert result.mode == ReturnMode.STATS
        assert result.partial_stats is not None
        assert result.partial_stats["mean"] == 0.5

    def test_validation_paths_mode(self):
        """Test that PATHS mode requires paths."""
        with pytest.raises(ValueError, match="mode=PATHS requires paths"):
            BatchResult(batch_id=0, n_paths=2, mode=ReturnMode.PATHS, paths=None)

    def test_validation_values_mode(self):
        """Test that VALUES mode requires values."""
        with pytest.raises(ValueError, match="mode=VALUES requires values"):
            BatchResult(batch_id=0, n_paths=2, mode=ReturnMode.VALUES, values=None)

    def test_validation_stats_mode(self):
        """Test that STATS mode requires partial_stats."""
        with pytest.raises(ValueError, match="mode=STATS requires partial_stats"):
            BatchResult(
                batch_id=0, n_paths=2, mode=ReturnMode.STATS, partial_stats=None
            )

    def test_metadata(self):
        """Test that metadata can be stored."""
        paths = [Path(times=np.array([0]), states=np.array(["A"]))]

        result = BatchResult(
            batch_id=0,
            n_paths=1,
            mode=ReturnMode.PATHS,
            paths=paths,
            metadata={"worker_id": 3, "duration_ms": 123},
        )

        assert result.metadata["worker_id"] == 3
        assert result.metadata["duration_ms"] == 123


class TestMergeBatchResults:
    """Test merging batch results."""

    def test_merge_paths_mode(self):
        """Test merging PATHS mode results."""
        paths1 = [
            Path(times=np.array([0, 1]), states=np.array(["A", "B"])),
            Path(times=np.array([0, 1]), states=np.array(["A", "A"])),
        ]
        paths2 = [
            Path(times=np.array([0, 1]), states=np.array(["B", "A"])),
        ]

        batch1 = BatchResult(batch_id=0, n_paths=2, mode=ReturnMode.PATHS, paths=paths1)
        batch2 = BatchResult(batch_id=1, n_paths=1, mode=ReturnMode.PATHS, paths=paths2)

        merged = merge_batch_results([batch1, batch2])

        assert merged.n_paths == 3
        assert merged.mode == ReturnMode.PATHS
        assert len(merged.paths) == 3

    def test_merge_values_mode(self):
        """Test merging VALUES mode results."""
        values1 = np.array(["A", "B"])
        values2 = np.array(["B", "A", "A"])

        batch1 = BatchResult(
            batch_id=0, n_paths=2, mode=ReturnMode.VALUES, values=values1
        )
        batch2 = BatchResult(
            batch_id=1, n_paths=3, mode=ReturnMode.VALUES, values=values2
        )

        merged = merge_batch_results([batch1, batch2])

        assert merged.n_paths == 5
        assert merged.mode == ReturnMode.VALUES
        assert len(merged.values) == 5

    def test_merge_stats_mode(self):
        """Test merging STATS mode results."""
        stats1 = {"mean": 0.5, "n": 100}
        stats2 = {"mean": 0.6, "n": 150}

        batch1 = BatchResult(
            batch_id=0, n_paths=100, mode=ReturnMode.STATS, partial_stats=stats1
        )
        batch2 = BatchResult(
            batch_id=1, n_paths=150, mode=ReturnMode.STATS, partial_stats=stats2
        )

        merged = merge_batch_results([batch1, batch2])

        assert merged.n_paths == 250
        assert merged.mode == ReturnMode.STATS
        assert "batch_stats" in merged.partial_stats

    def test_merge_empty_list_raises_error(self):
        """Test that merging empty list raises error."""
        with pytest.raises(ValueError, match="Cannot merge empty list"):
            merge_batch_results([])

    def test_merge_inconsistent_modes_raises_error(self):
        """Test that merging batches with different modes raises error."""
        paths = [Path(times=np.array([0]), states=np.array(["A"]))]
        values = np.array(["A"])

        batch1 = BatchResult(batch_id=0, n_paths=1, mode=ReturnMode.PATHS, paths=paths)
        batch2 = BatchResult(
            batch_id=1, n_paths=1, mode=ReturnMode.VALUES, values=values
        )

        with pytest.raises(ValueError, match="must have the same mode"):
            merge_batch_results([batch1, batch2])


class TestMCStatistics:
    """Test MCStatistics dataclass."""

    def test_creation(self):
        """Test creating MCStatistics."""
        stats = MCStatistics(
            n_paths=1000,
            mean=0.5,
            std=0.2,
            stderr=0.0063,
            confidence_interval=(0.487, 0.513),
            confidence_level=0.95,
        )

        assert stats.n_paths == 1000
        assert stats.mean == 0.5
        assert stats.confidence_level == 0.95

    def test_defaults(self):
        """Test default values."""
        stats = MCStatistics(n_paths=100)

        assert stats.n_paths == 100
        assert stats.mean is None
        assert stats.std is None
        assert stats.variance_reduction_factor == 1.0
        assert stats.confidence_level == 0.95

    def test_variance_reduction_factor(self):
        """Test variance reduction factor."""
        stats = MCStatistics(
            n_paths=1000, mean=0.5, std=0.1, variance_reduction_factor=2.5
        )

        assert stats.variance_reduction_factor == 2.5

    def test_metadata(self):
        """Test that additional metadata can be stored."""
        stats = MCStatistics(
            n_paths=1000,
            mean=0.5,
            metadata={"method": "antithetic", "n_batches": 10},
        )

        assert stats.metadata["method"] == "antithetic"
        assert stats.metadata["n_batches"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

