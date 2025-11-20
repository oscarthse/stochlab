"""Random walk on bounded integers with reflecting boundaries."""

from __future__ import annotations

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

    def sample_path(self, T: int, x0: int | None = None, **kwargs) -> Path:
        if x0 is None:
            x0 = np.random.choice(self._state_space.states)
        if x0 not in self._state_space:
            raise ValueError(f"x0={x0} not in state space")

        states = np.empty(T + 1, dtype=int)
        states[0] = x0

        for t in range(T):
            x = states[t]
            if x == self._lower:
                states[t + 1] = x + 1
            elif x == self._upper:
                states[t + 1] = x - 1
            else:
                states[t + 1] = x + 1 if np.random.rand() < self._p else x - 1

        return Path(times=np.arange(T + 1), states=states)
