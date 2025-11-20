"""Process-based parallel executor with batching and worker initialization."""

from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, TYPE_CHECKING

from .parallel_base import ParallelExecutor

if TYPE_CHECKING:
    from ...core.process import StochasticProcess
    from ..results import BatchResult

# Global variable to hold the process in each worker
# This is initialized once per worker and reused across batches
_WORKER_PROCESS: StochasticProcess | None = None


def _initialize_worker(pickled_process: bytes) -> None:
    """
    Initialize worker process with unpickled StochasticProcess.

    This runs once per worker when the pool starts. It deserializes
    the process and stores it in a global variable for reuse.

    Parameters
    ----------
    pickled_process : bytes
        Pickled StochasticProcess object.

    Notes
    -----
    Using global state is acceptable here because:
    1. Each worker is a separate process (no shared state issues)
    2. Initialization happens once, reducing overhead
    3. Process is read-only after initialization
    """
    global _WORKER_PROCESS
    _WORKER_PROCESS = pickle.loads(pickled_process)


def _simulate_batch_worker(
    batch_id: int,
    batch_seed: int,
    batch_size: int,
    T: int,
    x0: Any,
    mode: str,
    kwargs: dict[str, Any],
) -> BatchResult:
    """
    Worker function to simulate a batch of paths.

    This function runs in a worker process and:
    1. Uses the pre-initialized _WORKER_PROCESS
    2. Generates batch_size paths with independent seeds
    3. Returns results according to mode

    Parameters
    ----------
    batch_id : int
        Identifier for this batch.
    batch_seed : int
        Seed for this batch (will spawn seeds for individual paths).
    batch_size : int
        Number of paths to simulate in this batch.
    T : int
        Time horizon for each path.
    x0 : Any
        Initial state.
    mode : str
        Return mode ("paths", "values", or "stats").
    kwargs : dict
        Additional arguments to pass to sample_path.

    Returns
    -------
    BatchResult
        Batch results according to mode.
    """
    global _WORKER_PROCESS

    if _WORKER_PROCESS is None:
        raise RuntimeError("Worker not initialized - _initialize_worker not called")

    import numpy as np
    from ..results import BatchResult, ReturnMode
    from ..seeding import generate_path_seeds

    # Generate independent seeds for each path in this batch
    path_seeds = generate_path_seeds(batch_seed, batch_size)

    # Create RNGs for each path
    from ..seeding import make_rng

    # Simulate paths
    if mode == ReturnMode.PATHS.value:
        # Return full Path objects
        paths = []
        for seed in path_seeds:
            rng = make_rng(seed)
            np.random.seed(seed)  # Fallback for processes without rng support
            try:
                path = _WORKER_PROCESS.sample_path(T=T, x0=x0, rng=rng, **kwargs)
            except TypeError:
                # Process doesn't accept rng parameter
                path = _WORKER_PROCESS.sample_path(T=T, x0=x0, **kwargs)
            paths.append(path)

        return BatchResult(
            batch_id=batch_id,
            n_paths=batch_size,
            mode=ReturnMode.PATHS,
            paths=paths,
        )

    elif mode == ReturnMode.VALUES.value:
        # Return only final values (much lighter!)
        values = []
        for seed in path_seeds:
            rng = make_rng(seed)
            np.random.seed(seed)
            try:
                path = _WORKER_PROCESS.sample_path(T=T, x0=x0, rng=rng, **kwargs)
            except TypeError:
                path = _WORKER_PROCESS.sample_path(T=T, x0=x0, **kwargs)
            values.append(path.states[-1])  # Only final state

        return BatchResult(
            batch_id=batch_id,
            n_paths=batch_size,
            mode=ReturnMode.VALUES,
            values=np.array(values),
        )

    elif mode == ReturnMode.STATS.value:
        # Compute partial statistics (lightest!)
        final_values = []
        for seed in path_seeds:
            rng = make_rng(seed)
            np.random.seed(seed)
            try:
                path = _WORKER_PROCESS.sample_path(T=T, x0=x0, rng=rng, **kwargs)
            except TypeError:
                path = _WORKER_PROCESS.sample_path(T=T, x0=x0, **kwargs)
            final_values.append(path.states[-1])

        # Compute batch statistics
        values_array = np.array(final_values, dtype=object)
        partial_stats = {
            "sum": (
                float(np.sum(values_array == _WORKER_PROCESS.state_space.states[0]))
                if hasattr(values_array[0], "__eq__")
                else None
            ),
            "n": batch_size,
            # Can add more statistics here (mean, variance, etc.)
        }

        return BatchResult(
            batch_id=batch_id,
            n_paths=batch_size,
            mode=ReturnMode.STATS,
            partial_stats=partial_stats,
        )

    else:
        raise ValueError(f"Unknown return mode: {mode}")


class ProcessPoolParallelExecutor(ParallelExecutor):
    """
    Parallel executor using ProcessPoolExecutor with batching.

    This implementation:
    - Batches paths to reduce overhead (10-20x fewer futures)
    - Initializes workers once with the process (avoids repeated pickling)
    - Uses proper seed management for reproducibility
    - Supports progress tracking

    Parameters
    ----------
    n_jobs : int
        Number of worker processes (-1 = all CPUs).
    batch_size : int | None
        Paths per batch. If None, computed automatically based on n_jobs.

    Attributes
    ----------
    n_jobs : int
        Number of worker processes.
    batch_size : int | None
        Paths per batch.
    """

    def __init__(self, n_jobs: int = -1, batch_size: int | None = None):
        self.n_jobs = n_jobs if n_jobs > 0 else (os.cpu_count() or 1)
        self.batch_size = batch_size
        self._pool: ProcessPoolExecutor | None = None

    def _compute_batch_size(self, n_paths: int) -> int:
        """
        Compute optimal batch size based on workload.

        Parameters
        ----------
        n_paths : int
            Total number of paths.

        Returns
        -------
        int
            Optimal batch size.

        Notes
        -----
        Heuristic aims for 4-10 batches per worker:
        - Too few batches: poor load balancing
        - Too many batches: high overhead
        """
        if self.batch_size is not None:
            return self.batch_size

        # Target: 8 batches per worker
        target_batches_per_worker = 8
        ideal_batch_size = n_paths // (self.n_jobs * target_batches_per_worker)

        # Clamp to reasonable range
        min_batch_size = 10
        max_batch_size = 1000
        return max(min_batch_size, min(ideal_batch_size, max_batch_size))

    def execute_batches(
        self,
        batch_fn: Callable[[int, int], BatchResult],
        n_batches: int,
        batch_seeds: list[int],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BatchResult]:
        """
        Execute batch_fn in parallel using process pool.

        This method is generic and works with any batch function.
        For Monte Carlo, see simulate_monte_carlo() which provides
        a more convenient interface.
        """
        if len(batch_seeds) != n_batches:
            raise ValueError(
                f"batch_seeds length ({len(batch_seeds)}) must equal n_batches ({n_batches})"
            )

        batch_results = []

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            # Submit all batches
            futures = {
                executor.submit(batch_fn, batch_id, seed): batch_id
                for batch_id, seed in enumerate(batch_seeds)
            }

            # Collect as they complete
            iterator = as_completed(futures)

            # Optional progress bar
            if show_progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(
                        iterator,
                        total=n_batches,
                        desc="Simulating batches",
                        unit="batch",
                    )
                except ImportError:
                    print(
                        f"Simulating {n_batches} batches "
                        "(install tqdm for progress bar)..."
                    )

            # Gather results
            for future in iterator:
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    # Cancel remaining futures on error
                    for f in futures:
                        f.cancel()
                    raise RuntimeError(f"Batch simulation failed: {e}") from e

        return batch_results

    def simulate_monte_carlo(
        self,
        process: StochasticProcess,
        n_paths: int,
        T: int,
        x0: Any = None,
        mode: str = "paths",
        seed: int = 42,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[BatchResult]:
        """
        Simulate Monte Carlo paths with batching and worker initialization.

        This is the main entry point for parallel Monte Carlo simulation.

        Parameters
        ----------
        process : StochasticProcess
            Process to simulate.
        n_paths : int
            Total number of paths to generate.
        T : int
            Time horizon.
        x0 : Any, optional
            Initial state.
        mode : str
            Return mode: "paths", "values", or "stats".
        seed : int
            Base random seed.
        show_progress : bool
            Show progress bar.
        **kwargs : Any
            Additional arguments for sample_path.

        Returns
        -------
        list[BatchResult]
            Results from all batches.
        """
        from ..seeding import generate_batch_seeds

        # Compute batch parameters
        batch_size = self._compute_batch_size(n_paths)
        n_batches = (n_paths + batch_size - 1) // batch_size  # Ceiling division
        batch_seeds = generate_batch_seeds(seed, n_batches)

        # Adjust last batch size if needed
        batch_sizes = [batch_size] * (n_batches - 1)
        last_batch_size = n_paths - (batch_size * (n_batches - 1))
        batch_sizes.append(last_batch_size)

        # Pickle the process once for all workers
        pickled_process = pickle.dumps(process)

        # Create pool with initializer
        batch_results = []

        with ProcessPoolExecutor(
            max_workers=self.n_jobs,
            initializer=_initialize_worker,
            initargs=(pickled_process,),
        ) as executor:
            # Submit all batches
            futures = {}
            for batch_id, (batch_seed, batch_sz) in enumerate(
                zip(batch_seeds, batch_sizes)
            ):
                future = executor.submit(
                    _simulate_batch_worker,
                    batch_id=batch_id,
                    batch_seed=batch_seed,
                    batch_size=batch_sz,
                    T=T,
                    x0=x0,
                    mode=mode,
                    kwargs=kwargs,
                )
                futures[future] = batch_id

            # Collect as they complete
            iterator = as_completed(futures)

            if show_progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(
                        iterator,
                        total=n_batches,
                        desc="Simulating",
                        unit="batch",
                    )
                except ImportError:
                    pass

            # Gather results
            for future in iterator:
                try:
                    result = future.result()
                    batch_results.append(result)
                except Exception as e:
                    # Cancel remaining
                    for f in futures:
                        f.cancel()
                    raise RuntimeError(f"Simulation failed: {e}") from e

        return batch_results

    def close(self) -> None:
        """Clean up pool resources."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
