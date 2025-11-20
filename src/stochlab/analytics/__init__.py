"""Analytical utilities for discrete-time stochastic processes."""

from .markov import (
    AbsorptionResult,
    HittingTimesResult,
    StationaryResult,
    absorption_probabilities,
    hitting_times,
    stationary_distribution,
)

__all__ = [
    "StationaryResult",
    "HittingTimesResult",
    "AbsorptionResult",
    "stationary_distribution",
    "hitting_times",
    "absorption_probabilities",
]
