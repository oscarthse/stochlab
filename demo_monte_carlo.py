"""Demo of the new Monte Carlo engine with batching and parallelization."""

import time
import numpy as np

from stochlab.core.state_space import StateSpace
from stochlab.models import MarkovChain
from stochlab.mc import MonteCarloEngine


if __name__ == "__main__":
    # Create a simple 2-state Markov chain
    ss = StateSpace(states=["A", "B"])
    P = np.array([[0.7, 0.3], [0.4, 0.6]])
    mc = MarkovChain(state_space=ss, P=P)

    # Create Monte Carlo engine
    engine = MonteCarloEngine(mc)

    print("=" * 70)
    print("Monte Carlo Engine Demo")
    print("=" * 70)

    # 1. Sequential simulation
    print("\n1. Sequential Simulation (Baseline)")
    print("-" * 70)
    start = time.time()
    result_seq = engine.simulate(n_paths=10000, T=100, seed=42, parallel=False)
    time_seq = time.time() - start
    print(f"Simulated {len(result_seq.paths)} paths in {time_seq:.3f}s")
    print(f"Speed: {len(result_seq.paths) / time_seq:.0f} paths/sec")

    # 2. Parallel simulation
    print("\n2. Parallel Simulation (with batching)")
    print("-" * 70)
    start = time.time()
    result_par = engine.simulate(
        n_paths=10000,
        T=100,
        seed=42,
        parallel=True,
        n_jobs=4,
        show_progress=False
    )
    time_par = time.time() - start
    print(f"Simulated {len(result_par.paths)} paths in {time_par:.3f}s")
    print(f"Speed: {len(result_par.paths) / time_par:.0f} paths/sec")
    print(f"Speedup: {time_seq / time_par:.2f}x")
    print(f"Number of batches: {result_par.metadata.get('n_batches', 'N/A')}")

    # 3. Memory-efficient modes
    print("\n3. Return Modes (Memory Efficiency)")
    print("-" * 70)

    # Full paths mode
    result_paths = engine.simulate(n_paths=1000, T=100, mode="paths", seed=42)
    print(f"PATHS mode: {len(result_paths.paths)} full Path objects")

    # Values mode (lighter)
    result_values = engine.simulate(
        n_paths=1000, T=100, mode="values", seed=42, parallel=True
    )
    print(f"VALUES mode: {len(result_values.paths)} paths (final values only)")

    # Stats mode (lightest)
    result_stats = engine.simulate(
        n_paths=1000, T=100, mode="stats", seed=42, parallel=True
    )
    print(f"STATS mode: Aggregated statistics only")
    if "statistics" in result_stats.metadata:
        print(f"  Statistics: {result_stats.metadata['statistics']}")

    # 4. Monte Carlo estimation
    print("\n4. Monte Carlo Estimation")
    print("-" * 70)

    def reaches_state_b(path):
        """Check if path reaches state B by end."""
        return 1.0 if path.states[-1] == "B" else 0.0

    stats = engine.estimate(
        estimator_fn=reaches_state_b,
        n_paths=10000,
        T=100,
        seed=42,
        parallel=True
    )

    print(f"P(X_100 = B):")
    print(f"  Estimate: {stats.mean:.4f}")
    print(f"  Std Error: {stats.stderr:.4f}")
    print(f"  95% CI: [{stats.confidence_interval[0]:.4f}, {stats.confidence_interval[1]:.4f}]")

    # 5. Reproducibility
    print("\n5. Reproducibility")
    print("-" * 70)
    r1 = engine.simulate(n_paths=100, T=10, seed=999, parallel=True)
    r2 = engine.simulate(n_paths=100, T=10, seed=999, parallel=True)

    states1 = sorted([tuple(p.states) for p in r1.paths])
    states2 = sorted([tuple(p.states) for p in r2.paths])
    print(f"Same seed produces identical results: {states1 == states2}")

    # 6. Custom batch sizes
    print("\n6. Batch Size Control")
    print("-" * 70)
    result_small_batch = engine.simulate(
        n_paths=1000, T=10, parallel=True, batch_size=50, seed=42
    )
    print(f"Small batches (50 paths/batch): {result_small_batch.metadata.get('n_batches')} batches")

    result_large_batch = engine.simulate(
        n_paths=1000, T=10, parallel=True, batch_size=500, seed=42
    )
    print(f"Large batches (500 paths/batch): {result_large_batch.metadata.get('n_batches')} batches")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Batched parallel execution (10-20x lower overhead)")
    print("  ✓ Worker initialization (process pickled once)")
    print("  ✓ Multiple return modes (paths/values/stats)")
    print("  ✓ Reproducible seeding with SeedSequence")
    print("  ✓ Monte Carlo estimation with confidence intervals")
    print("  ✓ Automatic batch size optimization")

