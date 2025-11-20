"""M/M/1 queueing model with Poisson arrivals and exponential service."""

from __future__ import annotations

from typing import Any

import numpy as np

from stochlab.core import Path, StateSpace, StochasticProcess
from stochlab.core.state_space import State


class MM1Queue(StochasticProcess):
    """
    Single-server queue with Poisson arrivals and exponential service times.

    The state is the queue length (including any job in service) capped at
    `max_queue_length`, which provides the finite state space required by the
    core architecture. If the simulated queue would exceed this cap, a
    RuntimeError is raised, signalling that the user should increase the cap.
    """

    def __init__(
        self,
        arrival_rate: float,
        service_rate: float,
        max_queue_length: int,
    ) -> None:
        if arrival_rate <= 0:
            raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}.")
        if service_rate <= 0:
            raise ValueError(f"service_rate must be > 0, got {service_rate}.")
        if max_queue_length < 0:
            raise ValueError(f"max_queue_length must be >= 0, got {max_queue_length}.")

        self.arrival_rate = float(arrival_rate)
        self.service_rate = float(service_rate)
        self.max_queue_length = int(max_queue_length)
        self._state_space = StateSpace(list(range(self.max_queue_length + 1)))

    @property
    def state_space(self) -> StateSpace:
        """Finite state space {0, 1, ..., max_queue_length}."""
        return self._state_space

    def sample_path(
        self,
        T: int,
        x0: State | None = None,
        **kwargs: Any,
    ) -> Path:
        """
        Simulate T event transitions of the embedded discrete-time chain.

        Times in the returned Path correspond to event epochs and are spaced
        by exponential inter-arrival times with rate λ + μ · 𝟙{queue>0}.
        """
        if T < 0:
            raise ValueError(f"T must be >= 0, got {T}.")

        rng = kwargs.pop("rng", None)
        if rng is None:
            rng = np.random.default_rng()

        if kwargs:
            raise TypeError(f"Unused keyword arguments: {list(kwargs.keys())}")

        if x0 is None:
            queue_length = 0
        else:
            if x0 not in self.state_space:
                raise ValueError(
                    f"x0={x0} not in queue state space "
                    f"[0, {self.max_queue_length}]."
                )
            # x0 is validated to be in state_space, which contains ints
            # Handle both Python int and numpy int types
            if not isinstance(x0, (int, np.integer)):
                raise TypeError(f"x0 must be an int, got {type(x0).__name__}")
            queue_length = int(x0)

        states = np.empty(T + 1, dtype=int)
        times = np.empty(T + 1, dtype=float)
        event_types: list[str] = ["start"]

        states[0] = queue_length
        times[0] = 0.0

        for t in range(1, T + 1):
            total_rate = self.arrival_rate
            service_active = queue_length > 0
            if service_active:
                total_rate += self.service_rate

            dt = rng.exponential(scale=1.0 / total_rate)
            times[t] = times[t - 1] + dt

            if not service_active:
                event = "arrival"
            else:
                p_arrival = self.arrival_rate / total_rate
                event = "arrival" if rng.random() < p_arrival else "departure"

            if event == "arrival":
                if queue_length == self.max_queue_length:
                    raise RuntimeError(
                        "Queue length exceeded max_queue_length during "
                        "simulation. Increase the cap to avoid truncation."
                    )
                queue_length += 1
            else:
                queue_length -= 1

            states[t] = queue_length
            event_types.append(event)

        return Path(
            times=times,
            states=states,
            extras={"event_types": np.array(event_types, dtype=object)},
        )
