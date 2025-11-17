from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from stochlab.core.process import StochasticProcess
from stochlab.core.state_space import StateSpace, State
from stochlab.core.simulation import Path


class MarkovChain(StochasticProcess):
    """
    Finite-state, time-homogeneous Markov chain.

    Parameters
    ----------
    state_space : StateSpace
        Finite set of states.
    P : np.ndarray
        Transition matrix of shape (n_states, n_states).
        P[i, j] = P(X_{t+1} = state_j | X_t = state_i).
    initial_dist : np.ndarray | None
        Optional initial distribution over states (length n_states).
        If None, a default (e.g. unit mass on first state) is used.
    """

    def __init__(
        self,
        state_space: StateSpace,
        P: np.ndarray,
        initial_dist: np.ndarray | None = None,
    ) -> None:
        self._state_space = state_space
        self.P = P
        self.initial_dist = initial_dist
        self._validate_and_setup()

    @property
    def state_space(self) -> StateSpace:
        """The finite state space for this Markov chain."""
        return self._state_space

    def _validate_and_setup(self) -> None:
        n = self._state_space.n_states

        # Ensure P is a 2D numpy array
        if self.P.ndim != 2:
            raise ValueError("Transition matrix P must be 2D.")

        if self.P.shape != (n, n):
            raise ValueError(f"P must have shape ({n}, {n}), got {self.P.shape}.")

        # Convert to float and validate rows
        P = np.asarray(self.P, dtype=float)

        if np.any(P < 0.0):
            raise ValueError("Transition matrix P must have non-negative entries.")

        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-8):
            raise ValueError("Each row of P must sum to 1.")

        # Store the cleaned version back
        self.P = P

        # Handle initial distribution
        if self.initial_dist is not None:
            init = np.asarray(self.initial_dist, dtype=float)

            if init.shape != (n,):
                raise ValueError(
                    f"initial_dist must have shape ({n},), got {init.shape}."
                )

            if np.any(init < 0.0):
                raise ValueError("initial_dist must be non-negative.")

            if not np.allclose(init.sum(), 1.0, atol=1e-8):
                raise ValueError("initial_dist must sum to 1.")

            self.initial_dist = init
        else:
            # default: point mass on first state
            init = np.zeros(n, dtype=float)
            init[0] = 1.0
            self.initial_dist = init

    def sample_path(
        self,
        T: int,
        x0: State | None = None,
        **kwargs: Any,
    ) -> Path:
        """
        Generate a single sample path (X_0, ..., X_T) from the Markov chain.
        """
        if T < 0:
            raise ValueError(f"T must be >= 0, got {T}.")

        # Extract rng from kwargs if provided
        rng = kwargs.pop("rng", None)
        if rng is None:
            rng = np.random.default_rng()

        # If there are any unexpected kwargs, fail fast
        if kwargs:
            raise TypeError(f"Unused keyword arguments: {list(kwargs.keys())}")

        n = self.state_space.n_states

        # Determine initial index
        if x0 is None:
            # Draw from initial_dist
            init_probs = self.initial_dist  # type: ignore[assignment]
            idx0 = rng.choice(n, p=init_probs)
        else:
            idx0 = self.state_space.index(x0)

        # Simulate indices
        indices = np.empty(T + 1, dtype=int)
        indices[0] = idx0

        for t in range(1, T + 1):
            i_prev = indices[t - 1]
            probs = self.P[i_prev]
            indices[t] = rng.choice(n, p=probs)

        # Map indices back to state labels
        states = np.array(
            [self.state_space.state(i) for i in indices],
            dtype=object,
        )
        times = np.arange(T + 1)

        return Path(times=times, states=states)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_transition_matrix(
        cls,
        states: Sequence[State],
        P: np.ndarray,
        initial_dist: np.ndarray | None = None,
    ) -> "MarkovChain":
        """
        Build a MarkovChain directly from states and a transition matrix.
        """
        ss = StateSpace(states=list(states))
        return cls(state_space=ss, P=P, initial_dist=initial_dist)
