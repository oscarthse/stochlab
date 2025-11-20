"""Analytical routines for finite-state Markov chains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from stochlab.core.state_space import State, StateSpace
from stochlab.models.markov_chain import MarkovChain


@dataclass(slots=True)
class StationaryResult:
    """
    Stationary distribution along with numerical diagnostics.

    Attributes
    ----------
    distribution : np.ndarray
        Array of shape (n_states,) with float dtype.
    states : tuple[State, ...] | None
        Optional state labels matching the distribution order.
    residual : float
        L1 residual ||πP - π|| computed after solving.
    method : str
        Numerical method used (e.g., "lstsq").
    """

    distribution: np.ndarray
    states: tuple[State, ...] | None
    residual: float
    method: str


@dataclass(slots=True)
class HittingTimesResult:
    """
    Expected hitting times for a target set.

    Attributes
    ----------
    times : np.ndarray
        Float array of shape (n_states,) giving E[τ_T | X0 = state].
    states : tuple[State, ...] | None
        Optional state labels aligned with `times`.
    target_mask : np.ndarray
        Boolean mask indicating which states belong to the target set.
    """

    times: np.ndarray
    states: tuple[State, ...] | None
    target_mask: np.ndarray


@dataclass(slots=True)
class AbsorptionResult:
    """
    Absorption probabilities and expected absorption times

    Attributes
    ----------
    probabilities : np.ndarray
        Float array of shape (n_transient, n_absorbing) with entries B_ij.
    expected_steps : np.ndarray
        Float array of length n_transient containing E[time to absorption].
    transient_states : tuple[State, ...] | None
        Optional labels for transient states.
    absorbing_states : tuple[State, ...] | None
        Optional labels for absorbing states.
    """

    probabilities: np.ndarray
    expected_steps: np.ndarray
    transient_states: tuple[State, ...] | None
    absorbing_states: tuple[State, ...] | None


def stationary_distribution(
    chain_or_matrix: MarkovChain | np.ndarray,
    *,
    method: str = "lstsq",
    tol: float = 1e-12,
) -> StationaryResult:
    """
    Compute a stationary distribution π such that πP = π.

    Parameters
    ----------
    chain_or_matrix : MarkovChain | np.ndarray
        MarkovChain instance or row-stochastic transition matrix.
    method : {"lstsq"}
        Numerical method to solve the constrained linear system. The default
        uses a least-squares solve with the normalization Σπ_i = 1 appended
        as the final equation.
    tol : float
        Values with magnitude < tol are clipped to zero before final
        normalization to guard against tiny negative entries from round-off.
    Notes
    -----
    For reducible chains the stationary distribution need not be unique;
    this routine returns one solution of the linear system.
    """
    P, states = _extract_transition_matrix(chain_or_matrix)
    n = P.shape[0]

    if method != "lstsq":
        raise ValueError(f"Unsupported method {method!r}.")

    A = np.vstack([P.T - np.eye(n), np.ones(n)])
    b = np.zeros(n + 1)
    b[-1] = 1.0

    raw_dist, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    raw_residual = float(np.linalg.norm(raw_dist @ P - raw_dist, ord=1))

    dist = np.where(np.abs(raw_dist) < tol, 0.0, raw_dist)
    dist = np.clip(dist, 0.0, None)
    total = dist.sum()
    if total <= 0.0:
        raise RuntimeError(
            "Stationary solver returned a zero vector; check transition matrix."
        )
    dist /= total

    residual = float(np.linalg.norm(dist @ P - dist, ord=1))
    combined_residual = max(raw_residual, residual)

    if combined_residual > 1e-8:
        raise RuntimeError(
            "Failed to compute a stationary distribution with sufficient accuracy."
        )

    return StationaryResult(
        distribution=dist,
        states=states,
        residual=residual,
        method=method,
    )


def hitting_times(
    chain_or_matrix: MarkovChain | np.ndarray,
    targets: Sequence[int | State],
    *,
    state_space: StateSpace | None = None,
) -> HittingTimesResult:
    """
    Expected time to hit the target set for each starting state

    Targets can be specified as indices or state labels (when either a
    MarkovChain is provided or `state_space` is supplied).
    """
    if len(targets) == 0:
        raise ValueError("targets must contain at least one state.")

    P, states = _extract_transition_matrix(chain_or_matrix, state_space)
    n = P.shape[0]

    target_indices = _coerce_indices(targets, states, state_space, n)
    target_mask = np.zeros(n, dtype=bool)
    target_mask[target_indices] = True

    non_target = np.where(~target_mask)[0]

    if non_target.size == 0:
        return HittingTimesResult(
            times=np.zeros(n),
            states=states,
            target_mask=target_mask,
        )

    Q = P[np.ix_(non_target, non_target)]
    identity = np.eye(Q.shape[0])
    ones = np.ones(Q.shape[0])
    try:
        h_sub = np.linalg.solve(identity - Q, ones)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Hitting times do not exist; targets may be unreachable."
        ) from exc

    times = np.zeros(n)
    times[target_mask] = 0.0
    times[non_target] = h_sub

    return HittingTimesResult(
        times=times,
        states=states,
        target_mask=target_mask,
    )


def absorption_probabilities(
    chain_or_matrix: MarkovChain | np.ndarray,
    *,
    transient_states: Sequence[int | State],
    absorbing_states: Sequence[int | State],
    state_space: StateSpace | None = None,
) -> AbsorptionResult:
    """
    Absorption probabilities B = (I - Q)^{-1} R and expected steps to absorption.
    """
    if len(absorbing_states) == 0:
        raise ValueError("Must supply at least one absorbing state.")
    if len(transient_states) == 0:
        raise ValueError("Must supply at least one transient state.")

    P, states = _extract_transition_matrix(chain_or_matrix, state_space)
    n = P.shape[0]

    transient_idx = _coerce_indices(transient_states, states, state_space, n)
    absorbing_idx = _coerce_indices(absorbing_states, states, state_space, n)

    if set(transient_idx) & set(absorbing_idx):
        raise ValueError("Transient and absorbing states must be disjoint.")

    _validate_absorbing_rows(P, absorbing_idx)

    Q = P[np.ix_(transient_idx, transient_idx)]
    R = P[np.ix_(transient_idx, absorbing_idx)]

    identity = np.eye(Q.shape[0])
    try:
        B = np.linalg.solve(identity - Q, R)
        t = np.linalg.solve(identity - Q, np.ones(Q.shape[0]))
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Absorption probabilities undefined; (I - Q) is singular."
        ) from exc

    transient_labels = None
    absorbing_labels = None
    if states is not None:
        transient_labels = tuple(states[i] for i in transient_idx)
        absorbing_labels = tuple(states[i] for i in absorbing_idx)

    return AbsorptionResult(
        probabilities=B,
        expected_steps=t,
        transient_states=transient_labels,
        absorbing_states=absorbing_labels,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_transition_matrix(
    chain_or_matrix: MarkovChain | np.ndarray,
    state_space: StateSpace | None = None,
) -> tuple[np.ndarray, tuple[State, ...] | None]:
    if isinstance(chain_or_matrix, MarkovChain):
        P = np.asarray(chain_or_matrix.P, dtype=float)
        states = tuple(chain_or_matrix.state_space.states)
        return P, states

    P = np.asarray(chain_or_matrix, dtype=float)
    states_tuple = None
    if state_space is not None:
        states_tuple = tuple(state_space.states)
        if len(state_space) != P.shape[0]:
            raise ValueError(
                "StateSpace size does not match transition matrix dimension."
            )
    _validate_transition_matrix(P)
    return P, states_tuple


def _validate_transition_matrix(P: np.ndarray) -> None:
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("Transition matrix must be square.")

    if np.any(P < -1e-12):
        raise ValueError("Transition matrix must have non-negative entries.")

    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("Each row of the transition matrix must sum to 1.")


def _coerce_indices(
    items: Sequence[int | State],
    states_from_chain: tuple[State, ...] | None,
    explicit_state_space: StateSpace | None,
    n_states: int,
) -> list[int]:
    if len(items) == 0:
        raise ValueError("Sequence cannot be empty.")

    derived_state_space: StateSpace | None = None
    indices: list[int] = []
    for obj in items:
        if isinstance(obj, int):
            if obj < 0 or obj >= n_states:
                raise ValueError(
                    f"State index {obj} out of range for {n_states} states."
                )
            indices.append(obj)
            continue

        state_space = explicit_state_space
        if state_space is None and states_from_chain is not None:
            if derived_state_space is None:
                derived_state_space = StateSpace(list(states_from_chain))
            state_space = derived_state_space
        if state_space is None:
            raise ValueError(
                "State labels provided but no StateSpace or MarkovChain names available."
            )
        indices.append(state_space.index(obj))

    return indices


def _validate_absorbing_rows(P: np.ndarray, absorbing_idx: Sequence[int]) -> None:
    for idx in absorbing_idx:
        row = P[idx]
        diag = row[idx]
        off_diag = np.delete(row, idx)
        if not np.isclose(diag, 1.0, atol=1e-8) or not np.allclose(
            off_diag, 0.0, atol=1e-8
        ):
            raise ValueError(
                f"State index {idx} is not absorbing (row must equal e_{idx})."
            )
