from __future__ import annotations

from importlib import metadata as _metadata

from . import analytics, mc
from .core import Path, SimulationResult, State, StateSpace, StochasticProcess
from .models import MM1Queue, MarkovChain, RandomWalk

try:
    __version__ = _metadata.version("stochlab")
except _metadata.PackageNotFoundError:  # pragma: no cover - dev installs
    __version__ = "0.0.0"

__all__ = [
    "State",
    "StateSpace",
    "Path",
    "SimulationResult",
    "StochasticProcess",
    "MarkovChain",
    "RandomWalk",
    "MM1Queue",
    "analytics",
    "mc",
    "__version__",
]
