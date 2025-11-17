"""Tests for the MarkovChain model."""

import numpy as np
import pytest

from stochlab.core.state_space import StateSpace
from stochlab.models.markov_chain import MarkovChain


def test_markov_chain_deterministic_alternation():
    """
    Simple deterministic Markov chain:

    states = ["A", "B"]
    P = [[0, 1],
         [1, 0]]

    Starting from "A", the chain alternates A, B, A, B, ...
    """
    ss = StateSpace(states=["A", "B"])
    P = np.array([[0.0, 1.0], [1.0, 0.0]])

    mc = MarkovChain(state_space=ss, P=P)

    path = mc.sample_path(T=5, x0="A")

    # times: 0,1,2,3,4,5
    assert list(path.times) == [0, 1, 2, 3, 4, 5]
    # states alternate deterministically
    assert list(path.states) == ["A", "B", "A", "B", "A", "B"]


def test_markov_chain_from_transition_matrix_absorbing():
    """
    Test the convenience constructor from_transition_matrix on a simple
    absorbing chain:

    A -> A with prob 1
    B -> B with prob 1
    """
    states = ["A", "B"]
    P = np.array([[1.0, 0.0], [0.0, 1.0]])

    mc = MarkovChain.from_transition_matrix(states=states, P=P)

    path_a = mc.sample_path(T=3, x0="A")
    path_b = mc.sample_path(T=3, x0="B")

    assert list(path_a.states) == ["A", "A", "A", "A"]
    assert list(path_b.states) == ["B", "B", "B", "B"]


def test_markov_chain_invalid_P_shape_raises():
    """P must be a square matrix with shape (n_states, n_states)."""
    ss = StateSpace(states=["A", "B"])
    bad_P = np.array(
        [[0.5, 0.5, 0.0], [0.3, 0.7, 0.0]]
    )  # shape (2, 3) instead of (2, 2)

    with pytest.raises(ValueError, match="must have shape"):
        MarkovChain(state_space=ss, P=bad_P)


def test_markov_chain_invalid_P_rowsum_raises():
    """Each row of P must sum to 1."""
    ss = StateSpace(states=["A", "B"])
    # rows do not sum to 1
    bad_P = np.array([[0.5, 0.4], [0.3, 0.3]])  # sums to 0.9  # sums to 0.6

    with pytest.raises(ValueError, match="Each row of P must sum to 1"):
        MarkovChain(state_space=ss, P=bad_P)


def test_markov_chain_invalid_initial_dist_length_raises():
    """initial_dist must have length equal to number of states."""
    ss = StateSpace(states=["A", "B"])
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    bad_init = np.array([1.0, 0.0, 0.0])  # wrong length

    with pytest.raises(ValueError, match="initial_dist must have shape"):
        MarkovChain(state_space=ss, P=P, initial_dist=bad_init)


def test_markov_chain_invalid_initial_dist_sum_raises():
    """initial_dist must sum to 1."""
    ss = StateSpace(states=["A", "B"])
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    bad_init = np.array([0.7, 0.7])  # sums to 1.4

    with pytest.raises(ValueError, match="initial_dist must sum to 1"):
        MarkovChain(state_space=ss, P=P, initial_dist=bad_init)


def test_markov_chain_sampling_uses_initial_dist_when_x0_none():
    """
    If x0 is None, X_0 should be drawn from initial_dist.

    We don't assert exact randomness, but we at least check that the code runs
    and the initial state is consistent with the support of initial_dist.
    """
    ss = StateSpace(states=["A", "B"])
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    # Only "B" has positive probability in initial_dist
    init = np.array([0.0, 1.0])

    rng = np.random.default_rng(123)
    mc = MarkovChain(state_space=ss, P=P, initial_dist=init)

    path = mc.sample_path(T=3, x0=None, rng=rng)

    # Because initial_dist is [0, 1], X_0 must be "B"
    assert path.states[0] == "B"
    assert len(path.states) == 4  # 0..3
