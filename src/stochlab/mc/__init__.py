"""
Monte Carlo simulation engine with advanced features.

This module provides high-performance Monte Carlo simulation with:

- **Batched Parallelization**: 10-20x lower overhead than naive parallelization
- **Worker Initialization**: Avoid repeated pickling of process objects
- **Multiple Return Modes**: paths/values/stats for memory efficiency
- **Variance Reduction**: Antithetic variates, control variates (coming soon)
- **Statistical Analysis**: Confidence intervals, diagnostics

Examples
--------
Basic usage:

>>> from stochlab.models import MarkovChain
>>> from stochlab.mc import MonteCarloEngine
>>> import numpy as np
>>>
>>> P = np.array([[0.7, 0.3], [0.4, 0.6]])
>>> mc = MarkovChain(transition_matrix=P, labels=["A", "B"])
>>>
>>> # Sequential simulation
>>> engine = MonteCarloEngine(mc)
>>> result = engine.simulate(n_paths=1000, T=100)
>>>
>>> # Parallel simulation
>>> result = engine.simulate(
...     n_paths=100000,
...     T=100,
...     parallel=True,
...     seed=42
... )

Estimate expectations:

>>> def final_state_is_b(path):
...     return 1.0 if path.states[-1] == "B" else 0.0
>>>
>>> stats = engine.estimate(
...     estimator_fn=final_state_is_b,
...     n_paths=10000,
...     T=100,
...     parallel=True
... )
>>> print(f"P(X_100 = B) = {stats.mean:.3f} ± {stats.stderr:.3f}")
"""

from .engine import MonteCarloEngine
from .results import BatchResult, MCStatistics, ReturnMode
from .seeding import generate_path_seeds, make_rng

__all__ = [
    "MonteCarloEngine",
    "BatchResult",
    "MCStatistics",
    "ReturnMode",
    "generate_path_seeds",
    "make_rng",
]

