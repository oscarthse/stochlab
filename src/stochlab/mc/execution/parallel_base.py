"""Abstract base class for parallel execution strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ..results import BatchResult


class ParallelExecutor(ABC):
    """
    Abstract interface for parallel execution of Monte Carlo simulations.

    This abstraction allows swapping between different parallelization backends
    (ProcessPoolExecutor, joblib, Ray, etc.) without changing user-facing code.

    The executor handles:
    - Worker process management
    - Task batching and scheduling
    - Seed distribution
    - Progress tracking
    - Error propagation

    Subclasses must implement:
    - execute_batches: Core parallel execution logic
    """

    @abstractmethod
    def execute_batches(
        self,
        batch_fn: Callable[[int, int], BatchResult],
        n_batches: int,
        batch_seeds: list[int],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BatchResult]:
        """
        Execute batch_fn in parallel across n_batches.

        Parameters
        ----------
        batch_fn : Callable[[int, int], BatchResult]
            Function to execute for each batch. Takes (batch_id, seed) and
            returns a BatchResult containing simulated paths or aggregates.
        n_batches : int
            Number of batches to execute.
        batch_seeds : list[int]
            Independent seeds for each batch (length must equal n_batches).
        show_progress : bool, optional
            Show progress bar if possible.
        **kwargs : Any
            Backend-specific options (e.g., timeout, chunksize).

        Returns
        -------
        list[BatchResult]
            Results from all batches. Order may not match submission order
            unless guaranteed by implementation.

        Raises
        ------
        RuntimeError
            If any batch execution fails.
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Clean up resources (close pools, shutdown workers).

        Optional: implement if executor holds persistent resources.
        """
        pass

    def __enter__(self) -> ParallelExecutor:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - clean up resources."""
        self.close()

