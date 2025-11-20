"""Tests for the MM1Queue model."""

import numpy as np
import pytest

from stochlab.models import MM1Queue


def test_mm1_queue_generates_path():
    queue = MM1Queue(arrival_rate=2.0, service_rate=3.0, max_queue_length=20)
    # Use a seed to make test deterministic and avoid queue overflow
    rng = np.random.default_rng(42)
    path = queue.sample_path(T=100, x0=2, rng=rng)

    assert len(path) == 101
    assert path.times[0] == pytest.approx(0.0)
    assert np.all(np.diff(path.times) > 0.0)
    assert np.all((0 <= path.states) & (path.states <= 10))


def test_mm1_queue_invalid_parameters():
    with pytest.raises(ValueError, match="arrival_rate must be > 0"):
        MM1Queue(arrival_rate=0.0, service_rate=1.0, max_queue_length=5)
    with pytest.raises(ValueError, match="service_rate must be > 0"):
        MM1Queue(arrival_rate=1.0, service_rate=0.0, max_queue_length=5)
    with pytest.raises(ValueError, match="max_queue_length must be"):
        MM1Queue(arrival_rate=1.0, service_rate=1.0, max_queue_length=-1)


def test_mm1_queue_invalid_initial_state():
    queue = MM1Queue(arrival_rate=1.0, service_rate=1.0, max_queue_length=5)
    with pytest.raises(ValueError, match="not in queue state space"):
        queue.sample_path(T=5, x0=10)


def test_mm1_queue_rng_reproducibility():
    queue = MM1Queue(arrival_rate=1.5, service_rate=2.0, max_queue_length=10)
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    path1 = queue.sample_path(T=20, x0=1, rng=rng1)
    path2 = queue.sample_path(T=20, x0=1, rng=rng2)

    np.testing.assert_allclose(path1.times, path2.times)
    np.testing.assert_array_equal(path1.states, path2.states)


def test_mm1_queue_overflow_raises():
    queue = MM1Queue(arrival_rate=5.0, service_rate=1.0, max_queue_length=2)

    class ArrivalOnlyRNG:
        def exponential(self, scale):
            return 0.1

        def random(self):
            return 0.0

    rng = ArrivalOnlyRNG()

    with pytest.raises(RuntimeError, match="exceeded max_queue_length"):
        queue.sample_path(T=1, x0=2, rng=rng)


def test_mm1_queue_rejects_unused_kwargs():
    queue = MM1Queue(arrival_rate=1.0, service_rate=1.0, max_queue_length=5)
    with pytest.raises(TypeError, match="Unused keyword arguments"):
        queue.sample_path(T=1, foo="bar")
