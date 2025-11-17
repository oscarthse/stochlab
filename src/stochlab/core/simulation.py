from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .state_space import State


@dataclass(slots=True)
class Path:
    """
    A single discrete-time sample path (X_0, X_1, ..., X_T).

    Represents one realization of a stochastic process over discrete time steps.

    Attributes
    ----------
    times : np.ndarray
        Time points, typically [0, 1, 2, ..., T].
    states : np.ndarray
        State values at each time point (labels or indices).
    extras : dict[str, Any]
        Optional metadata (waiting times, payoffs, etc.).

    Examples
    --------
    >>> times = np.array([0, 1, 2])
    >>> states = np.array(["A", "B", "A"])
    >>> path = Path(times=times, states=states)
    >>> len(path)
    3
    >>> path[1]
    'B'
    """

    times: np.ndarray
    states: np.ndarray
    extras: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate inputs and make arrays immutable."""
        if len(self.times) != len(self.states):
            raise ValueError(
                f"times and states must have same length: {len(self.times)} != {len(self.states)}"
            )
        if len(self.states) == 0:
            raise ValueError("Path cannot be empty")

        # Make arrays read-only to preserve trajectory immutability
        self.times.flags.writeable = False
        self.states.flags.writeable = False

    def __len__(self) -> int:
        """Return the number of time steps in the path."""
        return len(self.states)

    def __getitem__(self, idx: int) -> State:
        """Return the state at the given time index."""
        return self.states[idx]
