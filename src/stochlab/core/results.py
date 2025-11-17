from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .simulation import Path
from .state_space import State


@dataclass(slots=True)
class SimulationResult:
    """
    Collection of sample paths with basic analysis methods.

    Represents the output of a Monte Carlo simulation, containing
    multiple realizations of a stochastic process.

    Attributes
    ----------
    paths : list[Path]
        Collection of sample paths from the simulation.
    metadata : dict[str, Any]
        Optional information about the simulation (process name, seed, etc.).

    Examples
    --------
    >>> path1 = Path(times=np.array([0, 1]), states=np.array(["A", "B"]))
    >>> path2 = Path(times=np.array([0, 1]), states=np.array(["A", "A"]))
    >>> result = SimulationResult(paths=[path1, path2])
    >>> len(result)
    2
    >>> df = result.to_dataframe()
    >>> result.state_distribution(t=1)
    {'A': 0.5, 'B': 0.5}
    """

    paths: list[Path]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Basic validation of the paths collection."""
        if not self.paths:
            raise ValueError("SimulationResult must contain at least one Path.")

        for p in self.paths:
            if not isinstance(p, Path):
                raise TypeError(
                    f"All elements of 'paths' must be Path instances. Got {type(p)!r}."
                )

    def __len__(self) -> int:
        """Return the number of paths in the simulation."""
        return len(self.paths)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert simulation results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - 'path_id' : int       (which simulated path)
            - 't'       : int       (time index, 0-based)
            - 'time'    : Any       (actual time value from Path.times)
            - 'state'   : State     (state label or index)
        """
        records: list[dict[str, Any]] = []
        for path_id, path in enumerate(self.paths):
            # rely on Path guarantees: times and states have same length
            for t_idx, (time_value, state) in enumerate(
                zip(path.times, path.states, strict=True)
            ):
                records.append(
                    {
                        "path_id": path_id,
                        "t": t_idx,
                        "time": time_value,
                        "state": state,
                    }
                )
        return pd.DataFrame.from_records(records)

    def state_distribution(self, t: int) -> dict[State, float]:
        """
        Empirical distribution of states at time index t across all paths.

        Parameters
        ----------
        t : int
            Time index to analyze (0-based index into each Path).

        Returns
        -------
        dict[State, float]
            Mapping from state to empirical probability.
        """
        counts: dict[State, int] = {}
        total = 0

        for path in self.paths:
            if t >= len(path):
                continue  # this path doesn't reach time t
            state: State = path[t]
            counts[state] = counts.get(state, 0) + 1
            total += 1

        if total == 0:
            return {}

        # Convert counts to probabilities
        return {state: count / total for state, count in counts.items()}
