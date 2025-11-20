# Monte Carlo Engine Design Document

## Overview

This document outlines the design for a high-performance, user-friendly Monte Carlo simulation engine for `stochlab`. The engine will provide advanced features like variance reduction, parallelization, and efficient memory management while maintaining a simple API.

---

## 1. Core Requirements

### 1.1 Functional Requirements
- **Variance Reduction**: Antithetic variates, control variates, importance sampling
- **Parallelization**: Multi-core simulation with proper seed management
- **Streaming**: Memory-efficient simulation for large experiments
- **Progress Tracking**: Real-time feedback for long-running simulations
- **Statistical Analysis**: Confidence intervals, convergence diagnostics
- **Reproducibility**: Deterministic results with seed control
- **Batching**: Efficient batch processing of simulations

### 1.2 Non-Functional Requirements
- **Performance**: 10-100x speedup with parallelization
- **Memory Efficiency**: O(1) memory for streaming mode
- **Backward Compatibility**: Existing `simulate_paths()` continues to work
- **Type Safety**: Full type hints for IDE support
- **Documentation**: Clear examples and API docs

---

## 2. Architecture Design

### 2.1 Module Structure

```
src/stochlab/mc/
├── __init__.py              # Public API exports
├── engine.py                # MonteCarloEngine (main entry point)
├── executor.py              # Parallel execution strategies
├── variance_reduction.py    # Variance reduction techniques
├── statistics.py            # Statistical analysis utilities
├── streaming.py             # Memory-efficient streaming
└── seeding.py               # Reproducible random number generation
```

### 2.2 Integration with Existing Architecture

```
┌─────────────────────────────────────────┐
│      StochasticProcess (ABC)            │
│  ┌───────────────────────────────────┐  │
│  │ sample_path(T, x0) -> Path        │  │
│  │ simulate_paths(n, T) -> Result    │  │ ← Keep for simple use
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
                    │
                    │ uses
                    ▼
┌─────────────────────────────────────────┐
│      MonteCarloEngine                    │
│  ┌───────────────────────────────────┐  │
│  │ simulate(...)                      │  │ ← New advanced API
│  │ with_variance_reduction(...)       │  │
│  │ with_parallel(...)                 │  │
│  │ with_streaming(...)                │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
                    │
                    │ returns
                    ▼
┌─────────────────────────────────────────┐
│      SimulationResult / MCResult         │
│  ┌───────────────────────────────────┐  │
│  │ paths: list[Path]                  │  │
│  │ statistics: MCStatistics           │  │ ← Enhanced result
│  │ metadata: dict                     │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

## 3. API Design

### 3.1 Simple API (Keep Existing)

```python
# Current API - remains unchanged
result = process.simulate_paths(n_paths=1000, T=100)
```

### 3.2 Advanced API (New)

```python
from stochlab.mc import MonteCarloEngine

# Basic usage - similar to existing
engine = MonteCarloEngine(process)
result = engine.simulate(n_paths=10000, T=100)

# With variance reduction
result = engine.simulate(
    n_paths=5000,
    T=100,
    variance_reduction="antithetic"  # reduces variance by using complementary paths
)

# With parallelization
result = engine.simulate(
    n_paths=100000,
    T=100,
    parallel=True,
    n_jobs=-1  # use all cores
)

# With streaming (memory-efficient)
result = engine.simulate_streaming(
    n_paths=1_000_000,
    T=100,
    chunk_size=1000,
    aggregate_fn=lambda paths: compute_statistic(paths)
)

# Fluent/Builder pattern (most flexible)
result = (
    MonteCarloEngine(process)
    .with_variance_reduction("antithetic")
    .with_parallel(n_jobs=4)
    .with_confidence_level(0.95)
    .with_seed(42)
    .simulate(n_paths=10000, T=100)
)
```

### 3.3 Progress Tracking

```python
from stochlab.mc import MonteCarloEngine
from tqdm import tqdm

# Built-in progress bar
result = engine.simulate(
    n_paths=100000,
    T=100,
    parallel=True,
    show_progress=True  # uses tqdm
)

# Custom progress callback
def progress_callback(n_completed, n_total):
    print(f"Progress: {n_completed}/{n_total}")

result = engine.simulate(
    n_paths=100000,
    T=100,
    progress_callback=progress_callback
)
```

---

## 4. Performance Optimizations

### 4.1 Parallelization Strategy

**Option 1: `concurrent.futures` (Recommended)**
- ✅ Standard library (no dependencies)
- ✅ Works with both threads and processes
- ✅ Easy to use
- ✅ Good for I/O and CPU-bound tasks

**Option 2: `multiprocessing`**
- ✅ True parallelism (bypasses GIL)
- ⚠️ Overhead for process spawning
- ⚠️ Pickle limitations

**Option 3: `joblib`**
- ✅ Optimized for NumPy arrays
- ✅ Better caching
- ❌ External dependency

**Decision**: Use `concurrent.futures.ProcessPoolExecutor` as default, allow `joblib` as optional dependency.

### 4.2 Seed Management for Parallelization

```python
import numpy as np

def generate_independent_seeds(base_seed: int, n_paths: int) -> list[int]:
    """
    Generate independent seeds for parallel workers using SeedSequence.
    
    This ensures reproducibility and statistical independence.
    """
    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(n_paths)
    return [s.generate_state(1)[0] for s in child_seeds]
```

### 4.3 Memory Optimization

**Streaming Mode**:
- Process paths in chunks
- Only keep aggregated statistics in memory
- Support for arbitrary reduction operations

**Path Compression**:
- For large state spaces, store indices instead of labels
- Lazy conversion to labels when needed

### 4.4 Batching Strategy

```python
# Optimal batch size calculation
def optimal_batch_size(n_paths: int, n_jobs: int) -> int:
    """
    Calculate optimal batch size to minimize overhead.
    
    - Too small: high communication overhead
    - Too large: poor load balancing
    """
    # Heuristic: aim for ~100 batches per worker
    min_batch_size = 10
    ideal_batches_per_worker = 100
    batch_size = max(min_batch_size, n_paths // (n_jobs * ideal_batches_per_worker))
    return batch_size
```

---

## 5. Variance Reduction Techniques

### 5.1 Antithetic Variates

**Concept**: For each random path, generate a "mirror" path using complementary random numbers.

```python
# Implementation strategy
def sample_path_with_antithetic(self, T: int, x0: State, rng: np.random.Generator):
    """Generate a pair of paths using antithetic variates."""
    # Sample original path with random uniforms U
    uniforms = rng.uniform(0, 1, size=T)
    path1 = self._sample_path_from_uniforms(T, x0, uniforms)
    
    # Sample antithetic path with 1-U
    path2 = self._sample_path_from_uniforms(T, x0, 1 - uniforms)
    
    return path1, path2
```

**Benefit**: Can reduce variance by up to 50% for estimating expectations.

**Applicability**: Works best when estimator is monotonic in random inputs.

### 5.2 Control Variates

**Concept**: Use a correlated variable with known expectation to reduce variance.

```python
# Example: Using analytical steady-state for control variate
def estimate_with_control_variate(
    paths: list[Path],
    control_mean: float,  # Known expectation of control
    estimate_fn: Callable,
    control_fn: Callable
):
    """Estimate with control variate correction."""
    estimates = [estimate_fn(p) for p in paths]
    controls = [control_fn(p) for p in paths]
    
    # Optimal coefficient
    cov = np.cov(estimates, controls)[0, 1]
    var_control = np.var(controls)
    c = -cov / var_control
    
    # Corrected estimate
    corrected = [est + c * (ctrl - control_mean) 
                 for est, ctrl in zip(estimates, controls)]
    
    return np.mean(corrected)
```

**Benefit**: Can dramatically reduce variance if good control variate exists.

**Applicability**: Need a correlated variable with known mean (e.g., steady-state distribution).

### 5.3 Importance Sampling

**Concept**: Sample from alternative distribution, reweight results.

**Status**: Defer to Phase 2 (requires more complex integration).

---

## 6. Statistical Analysis

### 6.1 Confidence Intervals

```python
@dataclass
class MCStatistics:
    """Statistical summary of Monte Carlo simulation."""
    mean: float
    std: float
    stderr: float  # std / sqrt(n)
    confidence_interval: tuple[float, float]
    confidence_level: float
    n_paths: int
    n_effective: int  # after variance reduction
    variance_reduction_factor: float  # VRF = var_baseline / var_reduced
```

### 6.2 Convergence Diagnostics

```python
def compute_convergence_diagnostic(
    estimates: np.ndarray,
    window_size: int = 100
) -> dict:
    """
    Assess convergence of Monte Carlo estimate.
    
    Returns
    -------
    dict
        - rolling_mean: running average
        - rolling_std: rolling standard error
        - is_converged: bool (based on relative tolerance)
    """
    ...
```

---

## 7. Implementation Plan

### Phase 1: Core Engine (Week 1-2)
1. ✅ Design document (this)
2. Create `mc` module structure
3. Implement `MonteCarloEngine` base class
4. Implement basic `simulate()` method
5. Add seed management with `SeedSequence`
6. Write unit tests

### Phase 2: Parallelization (Week 2-3)
1. Implement `ParallelExecutor` with `concurrent.futures`
2. Add proper seed spawning for workers
3. Implement progress tracking with `tqdm`
4. Add batch size optimization
5. Write parallel tests with different n_jobs
6. Benchmark performance

### Phase 3: Variance Reduction (Week 3-4)
1. Implement antithetic variates
2. Implement control variates
3. Add variance reduction factor calculation
4. Write tests for variance reduction effectiveness
5. Document when to use each technique

### Phase 4: Streaming & Memory (Week 4-5)
1. Implement `simulate_streaming()`
2. Add chunk-based processing
3. Support for custom aggregation functions
4. Memory profiling tests
5. Document memory-efficient patterns

### Phase 5: Statistics & Analysis (Week 5-6)
1. Implement `MCStatistics` dataclass
2. Add confidence interval calculation
3. Implement convergence diagnostics
4. Add statistical tests (e.g., Kolmogorov-Smirnov)
5. Create visualization helpers

### Phase 6: Documentation & Examples (Week 6)
1. Write comprehensive docstrings
2. Create tutorial notebooks
3. Add to Sphinx docs
4. Performance benchmarks
5. Best practices guide

---

## 8. Testing Strategy

### 8.1  pytests
- Test each variance reduction technique independently
- Verify seed reproducibility
- Test parallel execution with mock processes
- Test edge cases (n_paths=1, empty paths, etc.)

### 8.2 Integration Tests
- End-to-end with real processes (MarkovChain, RandomWalk)
- Compare parallel vs. sequential results
- Verify variance reduction effectiveness

### 8.3 Performance Tests
- Benchmark parallel speedup (strong scaling)
- Benchmark memory usage in streaming mode
- Profile hotspots with `cProfile`

### 8.4 Statistical Tests
- Verify unbiasedness of estimators
- Verify confidence interval coverage
- Test variance reduction claims

---

## 9. Dependencies

### Required (Already Have)
- `numpy` - random number generation, array operations
- `pandas` - result storage (optional)

### Optional (User Choice)
- `tqdm` - progress bars (graceful fallback)
- `joblib` - alternative parallelization (optional)
- `matplotlib`/`plotly` - visualization (Phase 5)

### Development
- `pytest` - testing
- `pytest-benchmark` - performance testing
- `memory_profiler` - memory testing

---

## 10. Example Usage Patterns

### Pattern 1: Quick Parallel Simulation

```python
from stochlab.models import MarkovChain
from stochlab.mc import MonteCarloEngine

P = np.array([[0.7, 0.3], [0.4, 0.6]])
mc = MarkovChain(transition_matrix=P, labels=["A", "B"])

# Old way (still works)
result = mc.simulate_paths(n_paths=1000, T=100)

# New way (parallel + stats)
engine = MonteCarloEngine(mc)
result = engine.simulate(n_paths=100000, T=100, parallel=True)

print(f"Mean: {result.statistics.mean:.4f}")
print(f"95% CI: {result.statistics.confidence_interval}")
```

### Pattern 2: Variance Reduction

```python
# Standard simulation
result_standard = engine.simulate(n_paths=10000, T=100)
print(f"Std Error: {result_standard.statistics.stderr:.4f}")

# With antithetic variates (same accuracy with fewer paths)
result_antithetic = engine.simulate(
    n_paths=5000,  # half as many paths
    T=100,
    variance_reduction="antithetic"
)
print(f"Std Error: {result_antithetic.statistics.stderr:.4f}")
print(f"VRF: {result_antithetic.statistics.variance_reduction_factor:.2f}x")
```

### Pattern 3: Memory-Efficient Streaming

```python
# Compute hitting time distribution without storing all paths
def compute_hitting_time(paths):
    """Compute hitting times for a chunk of paths."""
    times = []
    for path in paths:
        hit = np.where(path.states == "B")[0]
        if len(hit) > 0:
            times.append(hit[0])
    return times

all_hitting_times = []

for chunk_result in engine.simulate_streaming(
    n_paths=1_000_000,
    T=100,
    chunk_size=1000
):
    hitting_times = compute_hitting_time(chunk_result.paths)
    all_hitting_times.extend(hitting_times)

# Now analyze
mean_hitting_time = np.mean(all_hitting_times)
```

### Pattern 4: Fluent API

```python
result = (
    MonteCarloEngine(process)
    .with_seed(42)
    .with_parallel(n_jobs=4)
    .with_variance_reduction("antithetic")
    .with_confidence_level(0.99)
    .with_progress_bar()
    .simulate(n_paths=100000, T=100)
)
```

---

## 11. Performance Targets

### Baseline (Current)
- 1000 paths, T=100, MarkovChain: ~0.1s (sequential)

### Target (With Optimization)
- **Parallel (8 cores)**: 6-8x speedup → ~0.0125s
- **Antithetic**: 2x variance reduction (50% fewer paths for same accuracy)
- **Streaming**: O(chunk_size) memory instead of O(n_paths)
- **Large Scale**: 1M paths in <30s on 8-core machine

---

## 12. Future Extensions (Phase 7+)

### Advanced Variance Reduction
- Stratified sampling
- Importance sampling
- Quasi-Monte Carlo (low-discrepancy sequences)

### Adaptive Simulation
- Stop when target precision reached
- Sequential sampling with online variance estimation

### Rare Event Simulation
- Multilevel splitting
- Cross-entropy method

### GPU Acceleration
- CuPy/JAX for GPU-based simulation
- Batch matrix operations

---

## 13. Open Questions & Decisions Needed

1. **API Style**: Functional (`simulate()`) vs. Fluent (`.with_X()`) vs. Both?
   - **Recommendation**: Support both, fluent returns new engine instance (immutable)

2. **Progress Bar Dependency**: Require `tqdm` or make it optional?
   - **Recommendation**: Optional with graceful fallback to simple print

3. **Result Object**: Extend `SimulationResult` or create new `MCResult`?
   - **Recommendation**: Create `MCResult` that inherits from `SimulationResult`

4. **Variance Reduction**: Always calculate VRF or only on request?
   - **Recommendation**: Only when requested (requires baseline run)

5. **Parallelization**: Default to sequential or auto-detect optimal n_jobs?
   - **Recommendation**: Default to sequential, `parallel=True` uses `os.cpu_count()`

---

## 14. Success Metrics

### Code Quality
- ✅ >90% test coverage
- ✅ Type hints everywhere
- ✅ Docstrings for all public APIs
- ✅ No linter errors

### Performance
- ✅ 6x speedup on 8-core machine
- ✅ <1.1x memory overhead (vs. sequential)
- ✅ Antithetic achieves >1.5x variance reduction

### User Experience
- ✅ Simple cases use <5 lines of code
- ✅ Clear error messages
- ✅ Works with all existing process types
- ✅ Zero breaking changes to existing API

---

## Summary

This design prioritizes:
1. **Simplicity**: Keep existing API, add power features opt-in
2. **Performance**: Smart parallelization with minimal overhead
3. **Correctness**: Proper seed management, statistical rigor
4. **Flexibility**: Support common patterns without bloat
5. **Maintainability**: Clean abstractions, testable components

Next step: Begin Phase 1 implementation with `engine.py` and basic tests.

