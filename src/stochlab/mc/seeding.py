"""Random seed management for reproducible parallel Monte Carlo simulation."""

from __future__ import annotations

import numpy as np


def generate_path_seeds(base_seed: int, n_paths: int) -> list[int]:
    """
    Generate statistically independent seeds for parallel path simulation.

    Uses NumPy's SeedSequence to ensure:
    - Reproducibility: same base_seed always produces same path seeds
    - Independence: paths use statistically independent random streams
    - Quality: proper entropy distribution across workers

    Parameters
    ----------
    base_seed : int
        Base seed for reproducibility. Same base_seed always produces
        the same sequence of path seeds.
    n_paths : int
        Number of independent seeds to generate.

    Returns
    -------
    list[int]
        List of n_paths independent seeds, suitable for parallel workers.

    Examples
    --------
    >>> seeds = generate_path_seeds(base_seed=42, n_paths=5)
    >>> len(seeds)
    5
    >>> seeds_again = generate_path_seeds(base_seed=42, n_paths=5)
    >>> seeds == seeds_again
    True

    Notes
    -----
    This is the recommended approach for parallel random number generation
    in NumPy since version 1.17. It provides better statistical properties
    than naive approaches like sequential seeds or offset seeds.

    References
    ----------
    .. [1] NumPy Random Sampling
       https://numpy.org/doc/stable/reference/random/parallel.html
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")

    # Create base sequence
    ss = np.random.SeedSequence(base_seed)

    # Spawn independent child sequences
    child_sequences = ss.spawn(n_paths)

    # Extract integer seeds for compatibility
    seeds = [seq.generate_state(1)[0] for seq in child_sequences]

    return seeds


def generate_batch_seeds(base_seed: int, n_batches: int) -> list[int]:
    """
    Generate seeds for batch-level parallelization.

    This is identical to generate_path_seeds but named separately for clarity.
    Use when you're batching paths and need one seed per batch.

    Parameters
    ----------
    base_seed : int
        Base seed for reproducibility.
    n_batches : int
        Number of batch seeds to generate.

    Returns
    -------
    list[int]
        List of n_batches independent seeds.
    """
    return generate_path_seeds(base_seed, n_batches)


def make_rng(seed: int) -> np.random.Generator:
    """
    Create a new NumPy random number generator from a seed.

    Parameters
    ----------
    seed : int
        Seed for the generator.

    Returns
    -------
    np.random.Generator
        Fresh random number generator.

    Notes
    -----
    Uses the PCG64 bit generator (NumPy's default), which is:
    - Fast (~2x speedup over legacy MT19937)
    - High quality statistical properties
    - Suitable for parallel use
    """
    return np.random.default_rng(seed)
