# Monte Carlo Simulation Guide

The Monte Carlo engine in `stochlab` provides high-performance parallel simulation with advanced features for large-scale experiments.

## Quick Start

```python
from stochlab.models import MarkovChain
from stochlab.mc import MonteCarloEngine
import numpy as np

# Create a process
P = np.array([[0.7, 0.3], [0.4, 0.6]])
mc = MarkovChain.from_transition_matrix(["A", "B"], P)

# Create engine and simulate
engine = MonteCarloEngine(mc)
result = engine.simulate(n_paths=10000, T=100, seed=42)

print(f"Simulated {len(result.paths)} paths")
```

---

## Core Concepts

### What is Monte Carlo Simulation?

Monte Carlo simulation estimates quantities by:
1. Running a stochastic process many times (generating sample paths)
2. Computing a statistic on each path
3. Averaging the results

**Example**: Estimate the probability that a Markov chain reaches state B by time 100.

```python
def reaches_b(path):
    return 1.0 if "B" in path.states else 0.0

stats = engine.estimate(
    estimator_fn=reaches_b,
    n_paths=10000,
    T=100,
    parallel=True
)

print(f"P(reach B) = {stats.mean:.4f} ± {stats.stderr:.4f}")
print(f"95% CI: {stats.confidence_interval}")
```

---

## Features

### 1. Parallel Simulation

Run simulations in parallel to leverage multiple CPU cores:

```python
# Sequential (uses 1 core)
result = engine.simulate(n_paths=10000, T=100, parallel=False)

# Parallel (uses all cores)
result = engine.simulate(n_paths=10000, T=100, parallel=True)

# Parallel with specific number of workers
result = engine.simulate(n_paths=10000, T=100, parallel=True, n_jobs=4)
```

**Performance**: Expect 6-8x speedup on an 8-core machine.

**When to use parallel**:
- `n_paths > 100` (overhead is amortized)
- Long time horizons (T is large)
- Complex processes (each path takes time)

**When NOT to use parallel**:
- Small simulations (`n_paths < 50`)
- Very fast processes (overhead dominates)
- Memory-constrained environments

### 2. Memory-Efficient Modes

Choose how much data to store based on your needs:

#### Paths Mode (Default)

Store complete path trajectories:

```python
result = engine.simulate(n_paths=1000, T=100, mode="paths")

# Access full paths
for path in result.paths:
    print(path.times)   # [0, 1, 2, ..., 100]
    print(path.states)  # ['A', 'B', 'A', ...]
```

**Use when**: You need complete trajectory information for visualization or detailed analysis.

**Memory**: ~10-20 KB per path (depends on path length).

#### Values Mode

Store only final state values:

```python
result = engine.simulate(n_paths=1000, T=100, mode="values", parallel=True)

# Access final values
final_states = [path.states[-1] for path in result.paths]
```

**Use when**: You only care about endpoints (e.g., steady-state analysis).

**Memory**: ~8 bytes per path (**90% reduction**).

#### Stats Mode

Store only aggregated statistics:

```python
result = engine.simulate(n_paths=1000, T=100, mode="stats", parallel=True)

# Statistics are in metadata
stats = result.metadata["statistics"]
```

**Use when**: Estimating expectations and don't need individual paths.

**Memory**: ~100 bytes total (**99% reduction**).

### 3. Reproducible Seeds

Monte Carlo simulations are reproducible when you specify a seed:

```python
# Run 1
result1 = engine.simulate(n_paths=1000, T=100, seed=42)

# Run 2 (identical results)
result2 = engine.simulate(n_paths=1000, T=100, seed=42)

# Verify
assert all(np.array_equal(p1.states, p2.states) 
           for p1, p2 in zip(result1.paths, result2.paths))
```

**How it works**: Uses NumPy's `SeedSequence` to generate statistically independent seeds for each path while maintaining reproducibility.

### 4. Batch Size Control

The engine automatically computes optimal batch sizes, but you can override:

```python
# Automatic (recommended)
result = engine.simulate(n_paths=10000, T=100, parallel=True)

# Manual control
result = engine.simulate(
    n_paths=10000,
    T=100,
    parallel=True,
    batch_size=100  # Paths per batch
)
```

**Batch size trade-offs**:
- **Smaller batches** (50-100): Better load balancing, more overhead
- **Larger batches** (500-1000): Lower overhead, worse load balancing
- **Optimal** (100-200): Balance between the two

The automatic calculation targets 4-10 batches per worker, which works well for most cases.

### 5. Progress Tracking

Show progress bars for long-running simulations (requires `tqdm`):

```python
result = engine.simulate(
    n_paths=100000,
    T=100,
    parallel=True,
    show_progress=True  # Requires: pip install tqdm
)
```

If `tqdm` is not installed, the simulation continues without a progress bar.

---

## Advanced Usage

### Estimating Expectations

The `estimate()` method computes expectations with confidence intervals:

```python
def estimator(path):
    """Compute some quantity on the path."""
    return path.states[-1] == "B"  # Indicator function

stats = engine.estimate(
    estimator_fn=estimator,
    n_paths=10000,
    T=100,
    parallel=True,
    confidence_level=0.95
)

print(f"Estimate: {stats.mean:.4f}")
print(f"Std Error: {stats.stderr:.4f}")
print(f"95% CI: [{stats.confidence_interval[0]:.4f}, {stats.confidence_interval[1]:.4f}]")
```

**Output**:
```
Estimate: 0.5431
Std Error: 0.0050
95% CI: [0.5333, 0.5529]
```

### Complex Estimators

You can compute any statistic:

```python
# Average state occupancy
def avg_b_occupancy(path):
    return sum(s == "B" for s in path.states) / len(path.states)

# First passage time
def first_passage_time(path):
    try:
        return np.where(path.states == "B")[0][0]
    except IndexError:
        return float('inf')  # Never reached

# Path length until absorption
def absorption_time(path):
    return len(path) - 1
```

### Working with Results

```python
result = engine.simulate(n_paths=1000, T=100, parallel=True)

# Basic info
print(f"Number of paths: {len(result.paths)}")
print(f"Parallel: {result.metadata['parallel']}")
print(f"Batches used: {result.metadata.get('n_batches', 'N/A')}")

# Convert to DataFrame
df = result.to_dataframe()
print(df.head())
#    path_id  t  time state
# 0        0  0     0     A
# 1        0  1     1     B
# 2        0  2     2     B
# ...

# Analyze state distribution at specific time
dist = result.state_distribution(t=50)
print(dist)  # {'A': 0.42, 'B': 0.58}
```

---

## Performance Tips

### 1. Use Parallel for Large Simulations

```python
# Sequential: ~10 seconds
result = engine.simulate(n_paths=100000, T=100, parallel=False)

# Parallel (8 cores): ~1.5 seconds (6-7x speedup)
result = engine.simulate(n_paths=100000, T=100, parallel=True)
```

### 2. Choose the Right Mode

```python
# If you need full paths
result = engine.simulate(n_paths=10000, T=1000, mode="paths")
# Memory: ~170 MB

# If you only need final values
result = engine.simulate(n_paths=10000, T=1000, mode="values")
# Memory: ~80 KB (2000x reduction!)

# If you only need statistics
result = engine.simulate(n_paths=10000, T=1000, mode="stats")
# Memory: ~1 KB (170,000x reduction!)
```

### 3. Don't Over-Parallelize

```python
# BAD: Too few paths for parallelism
result = engine.simulate(n_paths=10, T=100, parallel=True)
# Overhead dominates, slower than sequential!

# GOOD: Enough paths to amortize overhead
result = engine.simulate(n_paths=10000, T=100, parallel=True)
# Overhead is negligible, get full speedup
```

**Rule of thumb**: Use `parallel=True` when `n_paths > 100`.

### 4. Batch Size Matters

The default automatic batch sizing works well, but for very large simulations:

```python
# For maximum throughput
result = engine.simulate(
    n_paths=1_000_000,
    T=100,
    parallel=True,
    batch_size=1000  # Larger batches = lower overhead
)

# For better progress tracking
result = engine.simulate(
    n_paths=1_000_000,
    T=100,
    parallel=True,
    batch_size=100,  # More frequent updates
    show_progress=True
)
```

---

## Common Patterns

### Pattern 1: Probability Estimation

```python
# Estimate P(event occurs)
def event_occurs(path):
    return 1.0 if some_condition(path) else 0.0

stats = engine.estimate(event_occurs, n_paths=10000, T=100, parallel=True)
print(f"P(event) = {stats.mean:.4f} ± {stats.stderr:.4f}")
```

### Pattern 2: Expected Value

```python
# Estimate E[X_T]
def final_value(path):
    return float(path.states[-1])

stats = engine.estimate(final_value, n_paths=10000, T=100, parallel=True)
print(f"E[X_100] = {stats.mean:.4f}")
```

### Pattern 3: Hitting Time Distribution

```python
# Estimate distribution of hitting times
result = engine.simulate(n_paths=10000, T=1000, parallel=True)

hitting_times = []
for path in result.paths:
    hit_indices = np.where(path.states == "B")[0]
    if len(hit_indices) > 0:
        hitting_times.append(hit_indices[0])

print(f"Mean hitting time: {np.mean(hitting_times):.2f}")
print(f"Median hitting time: {np.median(hitting_times):.2f}")
```

### Pattern 4: Comparing Processes

```python
# Compare two different processes
engine1 = MonteCarloEngine(process1)
engine2 = MonteCarloEngine(process2)

result1 = engine1.simulate(n_paths=10000, T=100, seed=42, parallel=True)
result2 = engine2.simulate(n_paths=10000, T=100, seed=42, parallel=True)

# Compare distributions
dist1 = result1.state_distribution(t=100)
dist2 = result2.state_distribution(t=100)
```

---

## Troubleshooting

### Issue: Parallel simulation crashes on Windows

**Symptom**:
```
RuntimeError: An attempt has been made to start a new process...
```

**Solution**: Wrap your code in `if __name__ == "__main__"`:

```python
if __name__ == "__main__":
    engine = MonteCarloEngine(process)
    result = engine.simulate(n_paths=10000, T=100, parallel=True)
```

This is required on Windows due to how multiprocessing spawns processes.

### Issue: Out of memory

**Symptom**: System runs out of RAM with large simulations.

**Solution 1**: Use `values` or `stats` mode:
```python
result = engine.simulate(
    n_paths=1_000_000,
    T=1000,
    mode="values",  # Much lower memory
    parallel=True
)
```

**Solution 2**: Process in chunks:
```python
results = []
for i in range(10):
    chunk = engine.simulate(
        n_paths=100000,
        T=1000,
        seed=42 + i,
        parallel=True
    )
    # Process chunk immediately
    results.append(process_chunk(chunk))
```

### Issue: Slower with parallelization

**Symptom**: `parallel=True` is slower than `parallel=False`.

**Causes**:
1. **Too few paths**: Overhead dominates
2. **Very fast processes**: Each path takes <1ms
3. **Small time horizon**: T < 10

**Solution**: Use `parallel=False` for small simulations, or increase `n_paths`.

### Issue: Results not reproducible

**Symptom**: Same seed gives different results.

**Cause**: Not specifying a seed, or process has internal randomness not controlled by RNG.

**Solution**: Always specify `seed` parameter:
```python
result = engine.simulate(n_paths=10000, T=100, seed=42)  # Reproducible
```

---

## API Reference

### MonteCarloEngine

```python
MonteCarloEngine(process: StochasticProcess)
```

Create a Monte Carlo engine for a stochastic process.

**Parameters**:
- `process`: Any `StochasticProcess` (MarkovChain, RandomWalk, MM1Queue, etc.)

#### simulate()

```python
simulate(
    n_paths: int,
    T: int,
    x0: State | None = None,
    *,
    parallel: bool = False,
    n_jobs: int = -1,
    batch_size: int | None = None,
    mode: str = "paths",
    seed: int | None = None,
    show_progress: bool = False,
    **kwargs
) -> SimulationResult
```

Run Monte Carlo simulation.

**Parameters**:
- `n_paths`: Number of paths to simulate (must be > 0)
- `T`: Time horizon for each path
- `x0`: Initial state (optional, uses process default if None)
- `parallel`: Enable parallel execution (default: False)
- `n_jobs`: Number of workers, -1 = all CPUs (default: -1)
- `batch_size`: Paths per batch (default: auto-computed)
- `mode`: Return mode - "paths", "values", or "stats" (default: "paths")
- `seed`: Random seed for reproducibility (default: random)
- `show_progress`: Show progress bar (default: False, requires tqdm)
- `**kwargs`: Additional arguments passed to `process.sample_path()`

**Returns**: `SimulationResult` with paths and metadata

#### estimate()

```python
estimate(
    estimator_fn: Callable[[Path], float],
    n_paths: int,
    T: int,
    x0: State | None = None,
    *,
    parallel: bool = False,
    confidence_level: float = 0.95,
    seed: int | None = None,
    **kwargs
) -> MCStatistics
```

Estimate E[f(X)] using Monte Carlo.

**Parameters**:
- `estimator_fn`: Function that takes a Path and returns a scalar
- `n_paths`: Number of Monte Carlo samples
- `T`: Time horizon
- `x0`: Initial state (optional)
- `parallel`: Use parallel execution (default: False)
- `confidence_level`: Confidence level for CI (default: 0.95)
- `seed`: Random seed (optional)
- `**kwargs`: Additional arguments for `sample_path()`

**Returns**: `MCStatistics` with mean, std, confidence interval, etc.

### MCStatistics

Result from `estimate()` containing:

**Attributes**:
- `n_paths`: Number of paths simulated
- `mean`: Estimated expectation
- `std`: Standard deviation of estimator
- `stderr`: Standard error (std / sqrt(n))
- `confidence_interval`: Tuple (lower, upper)
- `confidence_level`: Confidence level (e.g., 0.95)
- `variance_reduction_factor`: VRF if variance reduction used
- `metadata`: Additional information

### ReturnMode

Enum for simulation return modes:

- `ReturnMode.PATHS`: Full Path objects
- `ReturnMode.VALUES`: Final values only
- `ReturnMode.STATS`: Aggregated statistics only

---

## Further Reading

- **Design Document**: `docs/MONTE_CARLO_DESIGN.md` - Comprehensive design details
- **Parallelization Analysis**: `docs/PARALLELIZATION_ANALYSIS.md` - Performance deep dive
- **Implementation Summary**: `docs/MC_IMPLEMENTATION_SUMMARY.md` - What was built
- **Demo Script**: `demo_monte_carlo.py` - Runnable examples

---

## Summary

The Monte Carlo engine provides:

✓ **High performance**: 6-8x speedup with parallelization  
✓ **Memory efficiency**: 90-99% reduction with values/stats modes  
✓ **Reproducibility**: SeedSequence for quality randomness  
✓ **Ease of use**: Simple API with sensible defaults  
✓ **Flexibility**: Works with any stochastic process  

For most users, this is all you need:

```python
engine = MonteCarloEngine(process)
result = engine.simulate(n_paths=10000, T=100, parallel=True, seed=42)
```

Happy simulating! 🎲

