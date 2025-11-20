"""Tests for RandomWalk."""

import numpy as np
import pytest

from stochlab.models import RandomWalk


def test_random_walk_basic():
    """Test basic RandomWalk creation and path generation."""
    rw = RandomWalk(lower_bound=0, upper_bound=10, p=0.5)
    assert len(rw.state_space) == 11
    
    path = rw.sample_path(T=100, x0=5)
    assert len(path) == 101
    assert path[0] == 5
    assert all(0 <= s <= 10 for s in path.states)


def test_random_walk_reflecting_boundaries():
    """Test that boundaries reflect properly."""
    rw = RandomWalk(lower_bound=0, upper_bound=2, p=0.5)
    
    # At lower boundary, must move up
    path = rw.sample_path(T=1, x0=0)
    assert path[1] == 1
    
    # At upper boundary, must move down
    path = rw.sample_path(T=1, x0=2)
    assert path[1] == 1


def test_random_walk_biased():
    """Test biased random walk (p != 0.5)."""
    np.random.seed(42)
    rw = RandomWalk(lower_bound=-5, upper_bound=5, p=0.7)
    
    path = rw.sample_path(T=100, x0=0)
    # With p=0.7, expect upward drift
    assert path[-1] >= 0  # Likely to end positive


def test_random_walk_invalid_p():
    """Test that invalid p values raise errors."""
    with pytest.raises(ValueError, match="p must be in"):
        RandomWalk(0, 10, p=0.0)
    with pytest.raises(ValueError, match="p must be in"):
        RandomWalk(0, 10, p=1.0)
    with pytest.raises(ValueError, match="p must be in"):
        RandomWalk(0, 10, p=-0.1)


def test_random_walk_invalid_bounds():
    """Test that invalid bounds raise errors."""
    with pytest.raises(ValueError, match="lower_bound must be"):
        RandomWalk(10, 5)
    with pytest.raises(ValueError, match="lower_bound must be"):
        RandomWalk(5, 5)


def test_random_walk_invalid_x0():
    """Test that invalid initial state raises error."""
    rw = RandomWalk(0, 10)
    with pytest.raises(ValueError, match="not in state space"):
        rw.sample_path(T=10, x0=15)


def test_random_walk_simulate_paths():
    """Test multi-path simulation via StochasticProcess interface."""
    rw = RandomWalk(0, 10, p=0.5)
    result = rw.simulate_paths(n_paths=100, T=50, x0=5)
    
    assert len(result) == 100
    df = result.to_dataframe()
    assert len(df) == 100 * 51  # 100 paths × 51 time points
    assert all(0 <= s <= 10 for s in df['state'])
