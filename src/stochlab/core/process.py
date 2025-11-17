from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .state_space import StateSpace, State
from .simulation import Path
from .results import SimulationResult


class StochasticProcess(ABC):
    """
    Abstract base class for discrete-time stochastic processes on a finite state space.

    Any concrete model (Markov chain, queue, branching process, etc.) must:

    - expose its finite `state_space`,
    - implement `sample_path` to generate a single Path.

    This gives a uniform interface that other components (Monte Carlo, analytics,
    reporting) can rely on.
    """

    @property
    @abstractmethod
    def state_space(self) -> StateSpace:
        """
        The finite state space on which this process evolves.

        Returns
        -------
        StateSpace
            The finite set of allowed states.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_path(self, T: int, x0: State | None = None, **kwargs: Any) -> Path:
        """
        Generate a single discrete-time sample path (X_0, ..., X_T).

        Parameters
        ----------
        T : int
            Horizon (number of time steps minus one). The returned path will
            have length T+1, with times typically [0, 1, ..., T].
        x0 : State, optional
            Optional initial state. If None, the process may use a default
            initial distribution or a default starting state.
        **kwargs : Any
            Additional model-specific parameters for the simulation, if needed.

        Returns
        -------
        Path
            The simulated path.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Minimal convenience Monte Carlo helper
    # ------------------------------------------------------------------
    def simulate_paths(
        self,
        n_paths: int,
        T: int,
        x0: State | None = None,
        **kwargs: Any,
    ) -> SimulationResult:
        """
        Generate multiple independent sample paths and wrap them in a SimulationResult.

        This is a thin convenience wrapper around repeated calls to `sample_path`.
        More advanced Monte Carlo features (variance reduction, seeding strategies,
        parallelism, etc.) will live in a separate `mc` module.

        Parameters
        ----------
        n_paths : int
            Number of independent paths to simulate. Must be >= 1.
        T : int
            Horizon for each path (see sample_path).
        x0 : State, optional
            Optional common initial state for all paths.
        **kwargs : Any
            Additional model-specific arguments forwarded to `sample_path`.

        Returns
        -------
        SimulationResult
            Collection of simulated paths and basic metadata.
        """
        if n_paths <= 0:
            raise ValueError(f"n_paths must be >= 1, got {n_paths}.")

        paths: list[Path] = []
        for _ in range(n_paths):
            path = self.sample_path(T=T, x0=x0, **kwargs)
            paths.append(path)

        metadata = {
            "n_paths": n_paths,
            "T": T,
            "x0": x0,
            "process_type": type(self).__name__,
        }

        return SimulationResult(paths=paths, metadata=metadata)
