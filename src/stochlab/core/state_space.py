from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence


# Public alias for anything that can be used as a state label.
# This matches how we'll treat states throughout the library.
State = Hashable


@dataclass(frozen=True, slots=True)
class StateSpace:
    """
    Finite set of states with a stable index mapping.

    This class provides a clean bridge between *labels* (e.g. "A", "BBB",
    ("bull", "high-vol"), Enums, etc.) and *integer indices* 0, 1, ..., n-1.

    It is:

    - Finite: only a fixed, finite list of states is allowed.
    - Unique: all states must be distinct (no duplicates).
    - Immutable: once created, states and indices cannot change.

    Examples
    --------
    >>> ss = StateSpace(states=["A", "B", "C"])
    >>> len(ss)
    3
    >>> ss.index("B")
    1
    >>> ss.state(2)
    'C'
    >>> "A" in ss
    True
    >>> "Z" in ss
    False

    Notes
    -----
    The order of `states` defines the indexing:
    - states[0] has index 0
    - states[1] has index 1
    - ...
    This ordering will be used to index rows/columns of transition matrices
    in Markov chain models.
    """

    # Ordered sequence of states (labels). Must be unique.
    states: Sequence[State]

    # Optional precomputed mapping from state -> index.
    # If None, it will be built automatically from `states` in __post_init__.
    _index_map: Mapping[State, int] | None = None

    def __post_init__(self) -> None:
        # Build index_map if not provided
        if self._index_map is None:
            # Validate uniqueness of state labels
            if len(set(self.states)) != len(self.states):
                raise ValueError("States in StateSpace must be unique.")

            # Build the mapping state -> index
            index_map = {state: i for i, state in enumerate(self.states)}

            # Because the dataclass is frozen, we need object.__setattr__
            object.__setattr__(self, "_index_map", index_map)
        else:
            # If a custom index_map is provided, ensure it is consistent.
            states_set = set(self.states)
            keys_set = set(self._index_map.keys())
            if states_set != keys_set:
                raise ValueError(
                    "index_map keys must match states exactly. "
                    f"states={states_set!r}, keys={keys_set!r}"
                )

    # ------------------------------------------------------------------
    # Public properties and basic protocol methods
    # ------------------------------------------------------------------
    @property
    def index_map(self) -> Mapping[State, int]:
        """
        Mapping from state label -> integer index.

        This is intended to be read-only. Do not mutate it.
        """
        # mypy needs ignore because _index_map is declared as Optional
        return self._index_map  # type: ignore[return-value]

    def index(self, state: State) -> int:
        """
        Return the integer index corresponding to a state label.

        Parameters
        ----------
        state : State
            The state label to look up.

        Returns
        -------
        int
            The index in [0, n_states-1].

        Raises
        ------
        KeyError
            If the state is not part of this StateSpace.
        """
        try:
            return self._index_map[state]  # type: ignore[index]
        except KeyError as exc:  # pragma: no cover - trivial error path
            raise KeyError(f"Unknown state {state!r}") from exc

    def state(self, idx: int) -> State:
        """
        Return the state label corresponding to an integer index.

        Parameters
        ----------
        idx : int
            Index in [0, n_states-1].

        Returns
        -------
        State
            The corresponding state label.

        Raises
        ------
        IndexError
            If idx is out of range.
        """
        try:
            return self.states[idx]
        except IndexError as exc:  # pragma: no cover - trivial error path
            raise IndexError(f"State index out of range: {idx}") from exc

    # Small convenience aliases to integrate nicely with Python's containers
    def __len__(self) -> int:
        """Number of states in the state space."""
        return len(self.states)

    def __contains__(self, state: State) -> bool:
        """Return True if `state` is a member of this state space."""
        # mypy ignore because _index_map is Optional at the type level
        return state in self._index_map  # type: ignore[operator]

    @property
    def n_states(self) -> int:
        """Explicit property for the number of states (alias of len(ss))."""
        return len(self.states)
