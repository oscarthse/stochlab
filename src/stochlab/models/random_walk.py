"""Random walk on bounded integers with reflecting boundaries."""

from __future__ import annotations

from typing import Any, Hashable

import numpy as np

from stochlab.core import Path, StateSpace, StochasticProcess


class RandomWalk(StochasticProcess):
    """Simple random walk on integers [lower_bound, upper_bound] with reflecting boundaries.

    At each step: move up with probability p, down with probability 1-p.
    At boundaries: must move back into interior.
    """

    def __init__(
        self,
        lower_bound: int,
        upper_bound: int,
        p: float = 0.5,
    ):
        if not 0 < p < 1:
            raise ValueError(f"p must be in (0, 1), got {p}")
        if lower_bound >= upper_bound:
            raise ValueError(
                f"lower_bound must be < upper_bound, got [{lower_bound}, {upper_bound}]"
            )

        self._lower = lower_bound
        self._upper = upper_bound
        self._p = p
        self._state_space = StateSpace(list(range(lower_bound, upper_bound + 1)))

    @property
    def state_space(self) -> StateSpace:
        return self._state_space

    def sample_path(self, T: int, x0: Hashable | None = None, **kwargs: Any) -> Path:
        # Extract rng if provided, otherwise use default
        rng = kwargs.pop("rng", None)
        if rng is None:
            rng = np.random.default_rng()
        
        if kwargs:
            raise TypeError(f"Unused keyword arguments: {list(kwargs.keys())}")
        
        if x0 is None:
            # State space contains ints, convert to list for np.random.choice
            states_list = list(self._state_space.states)
            x0 = int(rng.choice(states_list))  # Convert to Python int immediately
        else:
            # Convert to int - handle both Python int and numpy integer types
            # First check if it's in state space (more lenient check)
            if x0 not in self._state_space:
                raise ValueError(f"x0={x0} not in state space")
            # Then convert to int (handles numpy scalars, Python ints, etc.)
            # We know x0 is in state_space which contains ints, so this is safe
            try:
                x0 = int(x0)  # type: ignore[call-overload]
            except (ValueError, TypeError) as e:
                raise TypeError(f"x0 must be convertible to int, got {type(x0).__name__}: {e}") from e
        
        if x0 not in self._state_space:
            raise ValueError(f"x0={x0} not in state space")
        
        current = x0

        states = np.empty(T + 1, dtype=int)
        states[0] = current

        for t in range(T):
            x = states[t]
            if x == self._lower:
                states[t + 1] = x + 1
            elif x == self._upper:
                states[t + 1] = x - 1
            else:
                states[t + 1] = x + 1 if rng.random() < self._p else x - 1

        return Path(times=np.arange(T + 1), states=states)
