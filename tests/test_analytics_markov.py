"""Tests for analytics.markov."""

import numpy as np
import pytest

from stochlab.analytics import (
    AbsorptionResult,
    HittingTimesResult,
    StationaryResult,
    absorption_probabilities,
    hitting_times,
    stationary_distribution,
)
from stochlab.models import MarkovChain
from stochlab.core import StateSpace


def build_chain():
    states = ["A", "B", "C"]
    P = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    return MarkovChain.from_transition_matrix(states, P)


def test_stationary_distribution_matrix_input():
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    result = stationary_distribution(P)
    assert isinstance(result, StationaryResult)
    np.testing.assert_allclose(result.distribution @ P, result.distribution, atol=1e-10)
    assert abs(result.distribution.sum() - 1.0) < 1e-12


def test_stationary_distribution_chain_input():
    chain = build_chain()
    result = stationary_distribution(chain)
    assert result.states == tuple(chain.state_space.states)
    assert pytest.approx(result.distribution[-1]) == 1.0


def test_hitting_times_target_label():
    chain = build_chain()
    result = hitting_times(chain, targets=["C"])
    assert isinstance(result, HittingTimesResult)
    # From state B, expected one step to reach C (0.5 stay, 0.5 go to C)
    assert pytest.approx(result.times[1], abs=1e-10) == 2.0
    assert result.times[2] == 0.0


def test_absorption_probabilities():
    P = np.array(
        [
            [0.5, 0.5, 0.0],
            [0.2, 0.5, 0.3],
            [0.0, 0.0, 1.0],
        ]
    )
    ss = StateSpace(["T1", "T2", "A"])
    res = absorption_probabilities(
        P,
        transient_states=["T1", "T2"],
        absorbing_states=["A"],
        state_space=ss,
    )
    assert isinstance(res, AbsorptionResult)
    # Only one absorbing state → probabilities column sums to 1 for each transient
    np.testing.assert_allclose(res.probabilities, np.ones((2, 1)))
    assert res.expected_steps[0] > 0.0


def test_invalid_transition_matrix():
    with pytest.raises(ValueError):
        stationary_distribution(np.array([[0.2, 0.8, 0.0]]))


def test_hitting_times_unreachable():
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(RuntimeError, match="Hitting times do not exist"):
        hitting_times(P, targets=[1])


def test_absorption_requires_absorbing_rows():
    P = np.array(
        [
            [0.5, 0.5],
            [0.4, 0.6],
        ]
    )
    with pytest.raises(ValueError, match="not absorbing"):
        absorption_probabilities(
            P,
            transient_states=[0],
            absorbing_states=[1],
        )


def test_target_index_out_of_range():
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError, match="out of range"):
        hitting_times(P, targets=[5])


def test_state_space_matrix_mismatch():
    P = np.eye(2)
    ss = StateSpace([0])
    with pytest.raises(ValueError, match="StateSpace size"):
        hitting_times(P, targets=[0], state_space=ss)
