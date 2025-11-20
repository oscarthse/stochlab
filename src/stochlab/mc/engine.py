"""Monte Carlo simulation engine with advanced features."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from ..core.results import SimulationResult
from .execution.parallel_process import ProcessPoolParallelExecutor
from .results import BatchResult, MCStatistics, ReturnMode, merge_batch_results
from .seeding import generate_path_seeds

if TYPE_CHECKING:
    from ..core.process import StochasticProcess
    from ..core.state_space import State


class MonteCarloEngine:
    """
    Advanced Monte Carlo simulation engine with parallelization and variance reduction.

    This engine provides high-performance simulation with:
    - Batched parallel execution (10-20x lower overhead)
    - Worker initialization (avoid repeated pickling)
    - Multiple return modes (paths/values/stats)
    - Reproducible seeding with SeedSequence
    - Progress tracking
    - Statistical analysis

    Parameters
    ----------
    process : StochasticProcess
        The stochastic process to simulate.

    Examples
    --------
    >>> from stochlab.models import MarkovChain
    >>> from stochlab.mc import MonteCarloEngine
    >>> import numpy as np
    >>>
    >>> # Create process
    >>> P = np.array([[0.7, 0.3], [0.4, 0.6]])
    >>> mc = MarkovChain(transition_matrix=P, labels=["A", "B"])
    >>>
    >>> # Basic parallel simulation
    >>> engine = MonteCarloEngine(mc)
    >>> result = engine.simulate(n_paths=10000, T=100, parallel=True)
    >>>
    >>> # With statistics mode (memory-efficient)
    >>> result = engine.simulate(
    ...     n_paths=100000,
    ...     T=100,
    ...     parallel=True,
    ...     mode="stats"
    ... )
    """

    def __init__(self, process: StochasticProcess):
        """
        Initialize Monte Carlo engine for a process.

        Parameters
        ----------
        process : StochasticProcess
            Process to simulate.
        """
        self.process = process
        self._executor: ProcessPoolParallelExecutor | None = None

    def simulate(
        self,
        n_paths: int,
        T: int,
        x0: State | None = None,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
        batch_size: int | None = None,
        mode: str = "paths",
        seed: int | None = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        n_paths : int
            Number of independent paths to simulate.
        T : int
            Time horizon for each path.
        x0 : State, optional
            Initial state for all paths.
        parallel : bool, default=False
            Enable parallel execution using multiple cores.
        n_jobs : int, default=-1
            Number of parallel workers (-1 = all CPUs). Only used if parallel=True.
        batch_size : int, optional
            Number of paths per batch. If None, computed automatically.
            Larger batches = lower overhead but less granular progress tracking.
        mode : str, default="paths"
            Return mode:
            - "paths": Full Path objects (highest memory, most flexible)
            - "values": Only final state values (lower memory)
            - "stats": Only aggregated statistics (lowest memory)
        seed : int, optional
            Random seed for reproducibility. If None, uses random seed.
        show_progress : bool, default=False
            Show progress bar (requires tqdm).
        **kwargs : Any
            Additional arguments passed to process.sample_path().

        Returns
        -------
        SimulationResult
            Simulation results with paths and metadata.

        Raises
        ------
        ValueError
            If n_paths <= 0 or invalid mode specified.

        Notes
        -----
        Parallel execution is recommended when:
        - n_paths > 100 (overhead is amortized)
        - T is large (each path takes significant time)
        - You have multiple CPU cores available

        Examples
        --------
        >>> # Sequential simulation
        >>> result = engine.simulate(n_paths=1000, T=100)
        >>>
        >>> # Parallel simulation
        >>> result = engine.simulate(
        ...     n_paths=100000,
        ...     T=100,
        ...     parallel=True,
        ...     seed=42
        ... )
        >>>
        >>> # Memory-efficient mode
        >>> result = engine.simulate(
        ...     n_paths=1000000,
        ...     T=100,
        ...     parallel=True,
        ...     mode="values"  # Don't store full paths
        ... )
        """
        if n_paths <= 0:
            raise ValueError(f"n_paths must be positive, got {n_paths}")

        # Validate mode
        try:
            return_mode = ReturnMode(mode)
        except ValueError:
            valid_modes = [m.value for m in ReturnMode]
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of {valid_modes}"
            ) from None

        # Generate base seed if not provided
        if seed is None:
            import numpy as np

            seed = np.random.randint(0, 2**31 - 1)

        # Sequential execution
        if not parallel:
            return self._simulate_sequential(
                n_paths=n_paths,
                T=T,
                x0=x0,
                mode=return_mode,
                seed=seed,
                show_progress=show_progress,
                **kwargs,
            )

        # Parallel execution
        return self._simulate_parallel(
            n_paths=n_paths,
            T=T,
            x0=x0,
            n_jobs=n_jobs,
            batch_size=batch_size,
            mode=return_mode,
            seed=seed,
            show_progress=show_progress,
            **kwargs,
        )

    def _simulate_sequential(
        self,
        n_paths: int,
        T: int,
        x0: State | None,
        mode: ReturnMode,
        seed: int,
        show_progress: bool,
        **kwargs: Any,
    ) -> SimulationResult:
        """Sequential simulation (fallback)."""
        from .seeding import make_rng

        # Generate seeds
        path_seeds = generate_path_seeds(seed, n_paths)

        paths = []

        iterator = range(n_paths)
        if show_progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, desc="Simulating", unit="path")
            except ImportError:
                pass

        # Simulate paths - use RNG for better isolation
        for i in iterator:
            # Create fresh RNG for this path
            rng = make_rng(path_seeds[i])
            # Also set global state as fallback for processes that don't support rng kwarg
            import numpy as np
            np.random.seed(path_seeds[i])
            # Try to pass rng, but don't fail if process doesn't accept it
            try:
                path = self.process.sample_path(T=T, x0=x0, rng=rng, **kwargs)
            except TypeError:
                # Process doesn't accept rng parameter, rely on np.random.seed()
                path = self.process.sample_path(T=T, x0=x0, **kwargs)
            paths.append(path)

        # Build metadata
        metadata = {
            "n_paths": n_paths,
            "T": T,
            "x0": x0,
            "process_type": type(self.process).__name__,
            "seed": seed,
            "parallel": False,
            "mode": mode.value,
        }

        return SimulationResult(paths=paths, metadata=metadata)

    def _simulate_parallel(
        self,
        n_paths: int,
        T: int,
        x0: State | None,
        n_jobs: int,
        batch_size: int | None,
        mode: ReturnMode,
        seed: int,
        show_progress: bool,
        **kwargs: Any,
    ) -> SimulationResult:
        """Parallel simulation using batched execution."""
        # Create executor
        executor = ProcessPoolParallelExecutor(n_jobs=n_jobs, batch_size=batch_size)

        # Run simulation
        batch_results = executor.simulate_monte_carlo(
            process=self.process,
            n_paths=n_paths,
            T=T,
            x0=x0,
            mode=mode.value,
            seed=seed,
            show_progress=show_progress,
            **kwargs,
        )

        # Merge batch results
        merged = merge_batch_results(batch_results)

        # Convert to SimulationResult
        if mode == ReturnMode.PATHS:
            paths = merged.paths or []
        elif mode == ReturnMode.VALUES:
            # Create minimal paths from values (just for compatibility)
            # In future, we might want a different result type
            import numpy as np

            paths = []
            if merged.values is not None:
                for val in merged.values:
                    from ..core.simulation import Path

                    path = Path(
                        times=np.array([T]),
                        states=np.array([val]),
                        extras={"mode": "values_only"},
                    )
                    paths.append(path)
        else:  # STATS mode
            # For stats mode, we don't have full paths
            # Create minimal placeholder path to satisfy SimulationResult
            import numpy as np
            from ..core.simulation import Path
            
            placeholder = Path(
                times=np.array([0]),
                states=np.array([None], dtype=object),
                extras={"mode": "stats_only", "note": "No full paths stored in stats mode"}
            )
            paths = [placeholder]  # Minimal placeholder

        metadata = {
            "n_paths": n_paths,
            "T": T,
            "x0": x0,
            "process_type": type(self.process).__name__,
            "seed": seed,
            "parallel": True,
            "n_jobs": n_jobs,
            "batch_size": batch_size,
            "n_batches": len(batch_results),
            "mode": mode.value,
        }

        if mode == ReturnMode.STATS and merged.partial_stats:
            metadata["statistics"] = merged.partial_stats

        return SimulationResult(paths=paths, metadata=metadata)

    def estimate(
        self,
        estimator_fn: Any,
        n_paths: int,
        T: int,
        x0: State | None = None,
        *,
        parallel: bool = False,
        confidence_level: float = 0.95,
        seed: int | None = None,
        **kwargs: Any,
    ) -> MCStatistics:
        """
        Estimate expectation of a function using Monte Carlo.

        This is a convenience method for computing E[f(X)] where X is a sample path.

        Parameters
        ----------
        estimator_fn : Callable[[Path], float]
            Function to estimate (takes Path, returns scalar).
        n_paths : int
            Number of Monte Carlo samples.
        T : int
            Time horizon.
        x0 : State, optional
            Initial state.
        parallel : bool, default=False
            Use parallel execution.
        confidence_level : float, default=0.95
            Confidence level for interval.
        seed : int, optional
            Random seed.
        **kwargs : Any
            Additional arguments for sample_path.

        Returns
        -------
        MCStatistics
            Statistical summary including mean, std, confidence interval.

        Examples
        --------
        >>> # Estimate probability of reaching state "B" by time 10
        >>> def reaches_b(path):
        ...     return 1.0 if "B" in path.states else 0.0
        >>>
        >>> stats = engine.estimate(
        ...     estimator_fn=reaches_b,
        ...     n_paths=10000,
        ...     T=10,
        ...     parallel=True
        ... )
        >>> print(f"P(reach B) = {stats.mean:.3f} ± {stats.stderr:.3f}")
        """
        # Simulate paths
        result = self.simulate(
            n_paths=n_paths,
            T=T,
            x0=x0,
            parallel=parallel,
            mode="paths",  # Need full paths for arbitrary estimator
            seed=seed,
            **kwargs,
        )

        # Apply estimator to each path
        import numpy as np

        estimates = np.array([estimator_fn(path) for path in result.paths])

        # Compute statistics
        mean = float(np.mean(estimates))
        std = float(np.std(estimates, ddof=1))
        stderr = std / np.sqrt(n_paths)

        # Confidence interval (assume normal)
        try:
            from scipy import stats as sp_stats
            z_score = sp_stats.norm.ppf(1 - (1 - confidence_level) / 2)
        except ImportError:
            # Fallback to normal approximation without scipy
            # For 95% CI, z ≈ 1.96; for 99% CI, z ≈ 2.576
            if confidence_level == 0.95:
                z_score = 1.959963984540054
            elif confidence_level == 0.99:
                z_score = 2.5758293035489004
            else:
                # Rough approximation for other levels
                import warnings
                warnings.warn(
                    f"scipy not installed, using rough approximation for {confidence_level} CI",
                    UserWarning
                )
                z_score = 2.0  # Conservative estimate
        
        ci_lower = mean - z_score * stderr
        ci_upper = mean + z_score * stderr

        return MCStatistics(
            n_paths=n_paths,
            mean=mean,
            std=std,
            stderr=stderr,
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=confidence_level,
        )

