"""Tests for StateSpace class."""

import pytest

from stochlab.core import StateSpace


def test_state_space_basic():
    """Test basic StateSpace functionality."""
    ss = StateSpace(states=["A", "B", "C"])

    assert len(ss) == 3
    assert ss.n_states == 3
    assert "A" in ss
    assert "Z" not in ss

    assert ss.index("A") == 0
    assert ss.index("B") == 1
    assert ss.state(0) == "A"
    assert ss.state(2) == "C"


def test_state_space_validation():
    """Test StateSpace validation."""
    # Duplicate states should raise ValueError
    with pytest.raises(ValueError, match="States in StateSpace must be unique"):
        StateSpace(states=["A", "B", "A"])


def test_state_space_errors():
    """Test StateSpace error handling."""
    ss = StateSpace(states=["A", "B"])

    # Unknown state should raise KeyError
    with pytest.raises(KeyError, match="Unknown state"):
        ss.index("Z")

    # Out of range index should raise IndexError
    with pytest.raises(IndexError, match="State index out of range"):
        ss.state(5)
