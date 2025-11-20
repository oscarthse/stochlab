"""Parallel execution strategies for Monte Carlo simulation."""

from .parallel_base import ParallelExecutor
from .parallel_process import ProcessPoolParallelExecutor

__all__ = [
    "ParallelExecutor",
    "ProcessPoolParallelExecutor",
]

