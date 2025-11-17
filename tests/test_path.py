"""Tests for Path class."""

import numpy as np
import pytest

from stochlab.core import Path


def test_path_creation():
    """Test basic Path creation."""
    times = np.array([0, 1, 2])
    states = np.array(["A", "B", "A"])
    path = Path(times=times, states=states)
    
    assert len(path) == 3
    assert path[0] == "A"
    assert path[1] == "B"
    assert path[2] == "A"


def test_path_validation():
    """Test Path validation in __post_init__."""
    # Mismatched lengths should raise ValueError
    with pytest.raises(ValueError, match="times and states must have same length"):
        Path(times=np.array([0, 1]), states=np.array(["A"]))
    
    # Empty path should raise ValueError
    with pytest.raises(ValueError, match="Path cannot be empty"):
        Path(times=np.array([]), states=np.array([]))


def test_path_immutability():
    """Test that arrays become read-only."""
    times = np.array([0, 1])
    states = np.array(["A", "B"])
    path = Path(times=times, states=states)
    
    # Arrays should be read-only
    assert not path.times.flags.writeable
    assert not path.states.flags.writeable


def test_path_extras():
    """Test extras dictionary functionality."""
    path = Path(
        times=np.array([0, 1]),
        states=np.array(["A", "B"]),
        extras={"rewards": [10, 20]}
    )
    
    assert path.extras["rewards"] == [10, 20]