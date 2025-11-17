"""Tests for SimulationResult class."""

import numpy as np
import pytest

from stochlab.core import Path, SimulationResult


def test_simulation_result_creation():
    """Test basic SimulationResult creation."""
    path1 = Path(times=np.array([0, 1]), states=np.array(["A", "B"]))
    path2 = Path(times=np.array([0, 1]), states=np.array(["A", "A"]))
    
    result = SimulationResult(paths=[path1, path2])
    assert len(result) == 2


def test_simulation_result_validation():
    """Test SimulationResult validation."""
    # Empty paths should raise ValueError
    with pytest.raises(ValueError, match="must contain at least one Path"):
        SimulationResult(paths=[])
    
    # Non-Path objects should raise TypeError
    with pytest.raises(TypeError, match="must be Path instances"):
        SimulationResult(paths=["not a path"])


def test_to_dataframe():
    """Test DataFrame conversion."""
    path1 = Path(times=np.array([0, 1]), states=np.array(["A", "B"]))
    path2 = Path(times=np.array([0, 1]), states=np.array(["A", "A"]))
    
    result = SimulationResult(paths=[path1, path2])
    df = result.to_dataframe()
    
    assert len(df) == 4  # 2 paths × 2 time points
    assert set(df.columns) == {"path_id", "t", "time", "state"}
    assert df["path_id"].tolist() == [0, 0, 1, 1]
    assert df["state"].tolist() == ["A", "B", "A", "A"]


def test_state_distribution():
    """Test empirical state distribution."""
    path1 = Path(times=np.array([0, 1]), states=np.array(["A", "B"]))
    path2 = Path(times=np.array([0, 1]), states=np.array(["A", "A"]))
    
    result = SimulationResult(paths=[path1, path2])
    
    # At t=0: both paths have "A"
    dist_t0 = result.state_distribution(t=0)
    assert dist_t0 == {"A": 1.0}
    
    # At t=1: one "A", one "B"
    dist_t1 = result.state_distribution(t=1)
    assert dist_t1 == {"A": 0.5, "B": 0.5}