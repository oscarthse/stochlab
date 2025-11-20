"""Result types for Monte Carlo simulations with multiple return modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from ..core.simulation import Path


class ReturnMode(str, Enum):
    """
    Mode for what data to return from Monte Carlo simulation.

    Modes
    -----
    PATHS : str
        Return full Path objects (highest memory, most flexible).
    VALUES : str
        Return only final state values (lower memory).
    STATS : str
        Return only aggregated statistics (lowest memory, streaming-friendly).
    """

    PATHS = "paths"
    VALUES = "values"
    STATS = "stats"


@dataclass(slots=True)
class BatchResult:
    """
    Result from simulating a batch of paths.

    The content depends on the return mode:
    - PATHS mode: paths contains list[Path]
    - VALUES mode: values contains np.ndarray of final states
    - STATS mode: partial_stats contains aggregated statistics

    Attributes
    ----------
    batch_id : int
        Identifier for this batch.
    n_paths : int
        Number of paths in this batch.
    mode : ReturnMode
        What type of data is stored.
    paths : list[Path] | None
        Full path objects (if mode=PATHS).
    values : np.ndarray | None
        Final state values (if mode=VALUES).
    partial_stats : dict[str, Any] | None
        Partial statistics for aggregation (if mode=STATS).
    metadata : dict[str, Any]
        Additional batch-specific information.
    """

    batch_id: int
    n_paths: int
    mode: ReturnMode
    paths: list[Path] | None = None
    values: np.ndarray | None = None
    partial_stats: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that appropriate data is present for the mode."""
        if self.mode == ReturnMode.PATHS and self.paths is None:
            raise ValueError("mode=PATHS requires paths to be set")
        if self.mode == ReturnMode.VALUES and self.values is None:
            raise ValueError("mode=VALUES requires values to be set")
        if self.mode == ReturnMode.STATS and self.partial_stats is None:
            raise ValueError("mode=STATS requires partial_stats to be set")


@dataclass(slots=True)
class MCStatistics:
    """
    Statistical summary of Monte Carlo simulation results.

    Attributes
    ----------
    n_paths : int
        Total number of simulated paths.
    mean : float | None
        Mean of the estimator (if computed).
    std : float | None
        Standard deviation (if computed).
    stderr : float | None
        Standard error of the mean (std / sqrt(n)).
    confidence_interval : tuple[float, float] | None
        Confidence interval at specified level.
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%).
    variance_reduction_factor : float
        Ratio of baseline variance to reduced variance (>1 means improvement).
    metadata : dict[str, Any]
        Additional statistics or diagnostics.
    """

    n_paths: int
    mean: float | None = None
    std: float | None = None
    stderr: float | None = None
    confidence_interval: tuple[float, float] | None = None
    confidence_level: float = 0.95
    variance_reduction_factor: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


def merge_batch_results(batch_results: list[BatchResult]) -> BatchResult:
    """
    Merge multiple batch results into a single result.

    Parameters
    ----------
    batch_results : list[BatchResult]
        Batch results to merge (must all have same mode).

    Returns
    -------
    BatchResult
        Merged result containing all data from input batches.

    Raises
    ------
    ValueError
        If batches have inconsistent modes or if empty list provided.
    """
    if not batch_results:
        raise ValueError("Cannot merge empty list of batch results")

    # Check mode consistency
    mode = batch_results[0].mode
    if not all(br.mode == mode for br in batch_results):
        raise ValueError("All batch results must have the same mode")

    total_paths = sum(br.n_paths for br in batch_results)

    if mode == ReturnMode.PATHS:
        # Concatenate all paths
        all_paths = []
        for br in batch_results:
            if br.paths:
                all_paths.extend(br.paths)

        return BatchResult(
            batch_id=-1,  # merged result
            n_paths=total_paths,
            mode=mode,
            paths=all_paths,
        )

    elif mode == ReturnMode.VALUES:
        # Stack all values
        all_values = []
        for br in batch_results:
            if br.values is not None:
                all_values.append(br.values)

        merged_values = np.concatenate(all_values) if all_values else None

        return BatchResult(
            batch_id=-1,
            n_paths=total_paths,
            mode=mode,
            values=merged_values,
        )

    elif mode == ReturnMode.STATS:
        # Aggregate statistics
        # For now, just collect all partial stats
        # Actual aggregation depends on what statistics are being computed
        merged_stats = {
            "batch_stats": [br.partial_stats for br in batch_results],
            "n_batches": len(batch_results),
        }

        return BatchResult(
            batch_id=-1,
            n_paths=total_paths,
            mode=mode,
            partial_stats=merged_stats,
        )

    else:
        raise ValueError(f"Unknown return mode: {mode}")
