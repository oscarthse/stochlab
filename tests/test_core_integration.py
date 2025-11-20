"""Integration test for core components."""

import numpy as np

from stochlab.core import Path, SimulationResult


def test_core_integration():
    """Test that all core components work together."""
    # Create some sample paths
    path1 = Path(times=np.array([0, 1, 2]), states=np.array(["A", "B", "C"]))
    path2 = Path(times=np.array([0, 1, 2]), states=np.array(["A", "A", "C"]))

    # Create simulation result
    result = SimulationResult(paths=[path1, path2])

    # Test basic functionality
    assert len(result) == 2
    assert len(path1) == 3
    assert path1[1] == "B"

    # Test DataFrame conversion
    df = result.to_dataframe()
    assert len(df) == 6  # 2 paths × 3 time points
    assert set(df.columns) == {"path_id", "t", "time", "state"}

    # Test state distribution
    dist_t1 = result.state_distribution(t=1)
    assert dist_t1["A"] == 0.5
    assert dist_t1["B"] == 0.5
    assert "C" not in dist_t1


if __name__ == "__main__":
    test_core_integration()
    print("✅ Core integration test passed!")
