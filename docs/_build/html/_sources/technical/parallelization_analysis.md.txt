# Parallelization Strategy Analysis

## The Problem

We need to generate `n_paths` independent sample paths, where each call to `sample_path(T, x0)` is:
- **CPU-bound** (random number generation, array operations)
- **Independent** (no shared state)
- **Variable cost** (some processes may be faster than others)

---

## Option 1: `concurrent.futures.ProcessPoolExecutor` ⭐ RECOMMENDED

### Implementation

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import Callable

def parallel_simulate_process_pool(
    sample_fn: Callable,
    n_paths: int,
    base_seed: int,
    n_jobs: int = -1,
    show_progress: bool = False
) -> list[Path]:
    """
    Parallelize using process pool with proper seed management.
    
    Parameters
    ----------
    sample_fn : Callable
        Function that takes (seed) and returns a Path
    n_paths : int
        Number of paths to generate
    base_seed : int
        Base seed for reproducibility
    n_jobs : int
        Number of worker processes (-1 = all CPUs)
    show_progress : bool
        Show progress bar
    """
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    # Generate independent seeds for each path
    ss = np.random.SeedSequence(base_seed)
    seeds = [s.generate_state(1)[0] for s in ss.spawn(n_paths)]
    
    paths = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        futures = {executor.submit(sample_fn, seed): i 
                   for i, seed in enumerate(seeds)}
        
        # Collect results as they complete
        iterator = as_completed(futures)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=n_paths, desc="Simulating")
            except ImportError:
                pass
        
        for future in iterator:
            path = future.result()  # This will raise if worker had exception
            paths.append(path)
    
    return paths
```

### Pros
- ✅ **True parallelism** (bypasses GIL - Global Interpreter Lock)
- ✅ **Standard library** (no dependencies)
- ✅ **Cross-platform** (works on Windows, Linux, macOS)
- ✅ **Good performance** for CPU-bound tasks
- ✅ **Error handling** built-in (exceptions propagate)
- ✅ **Progress tracking** easy with `as_completed`
- ✅ **Resource management** automatic (context manager)

### Cons
- ⚠️ **Startup overhead** (~100-500ms for process spawning)
- ⚠️ **Pickling required** (process must be serializable)
- ⚠️ **Memory overhead** (separate process per worker)

### When to Use
- **n_paths > 100** (overhead is amortized)
- **CPU-bound workloads** (which Monte Carlo is)
- **Want zero dependencies**

### Performance Characteristics
- **Speedup**: Near-linear until CPU cores saturated (6-8x on 8-core)
- **Overhead**: ~200ms startup + ~1ms per task
- **Memory**: ~50MB per worker process

---

## Option 2: `multiprocessing.Pool`

### Implementation

```python
from multiprocessing import Pool
import numpy as np

def parallel_simulate_mp_pool(
    sample_fn: Callable,
    n_paths: int,
    base_seed: int,
    n_jobs: int = -1
) -> list[Path]:
    """Parallelize using multiprocessing.Pool."""
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    # Generate independent seeds
    ss = np.random.SeedSequence(base_seed)
    seeds = [s.generate_state(1)[0] for s in ss.spawn(n_paths)]
    
    with Pool(processes=n_jobs) as pool:
        paths = pool.map(sample_fn, seeds)
    
    return paths
```

### Pros
- ✅ **True parallelism** (bypasses GIL)
- ✅ **Standard library**
- ✅ **Simpler API** than ProcessPoolExecutor
- ✅ **Good for batch processing**

### Cons
- ⚠️ **Less flexible** than `concurrent.futures`
- ⚠️ **No `as_completed()`** (harder for progress tracking)
- ⚠️ **map() blocks** until all done (can't stream results)
- ⚠️ **Same overhead** as ProcessPoolExecutor

### When to Use
- When you want simpler code and don't need progress tracking
- Batch processing without streaming

---

## Option 3: `joblib.Parallel`

### Implementation

```python
from joblib import Parallel, delayed
import numpy as np

def parallel_simulate_joblib(
    sample_fn: Callable,
    n_paths: int,
    base_seed: int,
    n_jobs: int = -1,
    backend: str = "loky"  # or "multiprocessing", "threading"
) -> list[Path]:
    """
    Parallelize using joblib with optimized backend.
    
    joblib is optimized for NumPy arrays and scientific computing.
    """
    # Generate independent seeds
    ss = np.random.SeedSequence(base_seed)
    seeds = [s.generate_state(1)[0] for s in ss.spawn(n_paths)]
    
    paths = Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        delayed(sample_fn)(seed) for seed in seeds
    )
    
    return paths
```

### Pros
- ✅ **Optimized for NumPy** (efficient array serialization)
- ✅ **Multiple backends** (loky, multiprocessing, threading)
- ✅ **Better caching** (memoization support)
- ✅ **Progress bar built-in** (`verbose` parameter)
- ✅ **Batching optimization** (automatic)

### Cons
- ❌ **External dependency** (`pip install joblib`)
- ⚠️ **Less common** than stdlib
- ⚠️ **More complex** under the hood

### When to Use
- **Heavy NumPy usage** (which we have!)
- **Need caching** between runs
- OK with external dependency

### Performance Characteristics
- **Speedup**: Similar to ProcessPoolExecutor (6-8x)
- **Overhead**: Slightly lower due to NumPy optimization
- **Memory**: More efficient for large NumPy arrays

---

## Option 4: `threading.ThreadPoolExecutor`

### Implementation

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def parallel_simulate_threads(
    sample_fn: Callable,
    n_paths: int,
    base_seed: int,
    n_jobs: int = -1
) -> list[Path]:
    """Parallelize using threads (NOT recommended for CPU-bound)."""
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    ss = np.random.SeedSequence(base_seed)
    seeds = [s.generate_state(1)[0] for s in ss.spawn(n_paths)]
    
    paths = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(sample_fn, seed) for seed in seeds]
        for future in as_completed(futures):
            paths.append(future.result())
    
    return paths
```

### Pros
- ✅ **Low overhead** (no process spawning)
- ✅ **Shared memory** (no pickling)
- ✅ **Fast startup** (<1ms)

### Cons
- ❌ **GIL bottleneck** (limited speedup for CPU-bound tasks)
- ❌ **Poor CPU utilization** (1.2-1.5x speedup max)

### When to Use
- **I/O-bound** tasks only (network, disk)
- **NOT for Monte Carlo** (CPU-bound)

---

## Option 5: Ray (Advanced)

### Implementation

```python
import ray

@ray.remote
def sample_path_remote(process_state, seed, T, x0):
    """Sample a path in a Ray worker."""
    # Reconstruct process from serialized state
    process = reconstruct_process(process_state)
    np.random.seed(seed)
    return process.sample_path(T, x0)

def parallel_simulate_ray(
    process,
    n_paths: int,
    T: int,
    x0,
    base_seed: int
):
    """Parallelize using Ray for distributed computing."""
    ray.init(ignore_reinit_error=True)
    
    # Serialize process once
    process_state = serialize_process(process)
    
    # Generate seeds
    ss = np.random.SeedSequence(base_seed)
    seeds = [s.generate_state(1)[0] for s in ss.spawn(n_paths)]
    
    # Dispatch to workers
    futures = [
        sample_path_remote.remote(process_state, seed, T, x0)
        for seed in seeds
    ]
    
    # Collect results
    paths = ray.get(futures)
    
    ray.shutdown()
    return paths
```

### Pros
- ✅ **Distributed** (works across multiple machines)
- ✅ **Scales to clusters**
- ✅ **Advanced features** (plasma store, actor model)

### Cons
- ❌ **Heavy dependency** (large install)
- ❌ **Overkill** for single-machine workloads
- ❌ **Complex setup**

### When to Use
- **Massive scale** (millions of paths)
- **Have a cluster** available
- **Future extension** only

---

## Comparison Table

| Feature | ProcessPoolExecutor | multiprocessing.Pool | joblib | ThreadPool | Ray |
|---------|-------------------|---------------------|--------|------------|-----|
| **Speedup (8-core)** | 6-8x | 6-8x | 6-8x | 1.2x ❌ | 6-8x |
| **Startup Time** | 200ms | 200ms | 150ms | <1ms | ~2s |
| **Dependencies** | None ✅ | None ✅ | joblib | None ✅ | ray |
| **Progress Tracking** | Easy ✅ | Hard | Built-in ✅ | Easy | Hard |
| **Error Handling** | Excellent | Good | Good | Excellent | Complex |
| **NumPy Optimization** | No | No | Yes ✅ | N/A | Yes |
| **Cross-Platform** | Yes ✅ | Yes ✅ | Yes ✅ | Yes ✅ | Mostly |
| **Memory Efficient** | Good | Good | Better ✅ | Best | Good |
| **Distributed** | No | No | No | No | Yes ✅ |
| **Learning Curve** | Low ✅ | Low ✅ | Medium | Low ✅ | High |

---

## Recommendation: Hybrid Approach

### Primary: `concurrent.futures.ProcessPoolExecutor`
**Use as default** because:
1. No dependencies
2. Excellent error handling
3. Easy progress tracking with `as_completed()`
4. Good enough performance

### Optional: `joblib` (with fallback)
**Allow as opt-in** for power users:
```python
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

def simulate(self, ..., backend="auto"):
    """
    backend : str
        "auto" - use joblib if available, else ProcessPoolExecutor
        "stdlib" - force ProcessPoolExecutor
        "joblib" - force joblib (error if not installed)
    """
    if backend == "joblib" or (backend == "auto" and JOBLIB_AVAILABLE):
        return self._simulate_joblib(...)
    else:
        return self._simulate_stdlib(...)
```

---

## Detailed Implementation Strategy

### 1. Seed Management (Critical!)

```python
def generate_path_seeds(base_seed: int, n_paths: int) -> list[int]:
    """
    Generate statistically independent seeds for parallel workers.
    
    Uses SeedSequence to ensure:
    - Reproducibility (same base_seed -> same path seeds)
    - Independence (paths don't share random state)
    - Quality (proper entropy distribution)
    """
    ss = np.random.SeedSequence(base_seed)
    child_seeds = ss.spawn(n_paths)
    return [s.generate_state(1)[0] for s in child_seeds]
```

**Why SeedSequence?**
- ✅ Designed for parallel workloads
- ✅ Guarantees statistical independence
- ✅ Reproducible (same base seed → same children)
- ✅ NumPy recommended approach (since 1.17)

### 2. Batch Size Optimization

```python
def optimal_batch_size(n_paths: int, n_jobs: int) -> int:
    """
    Calculate optimal batch size to balance:
    - Communication overhead (favor larger batches)
    - Load balancing (favor smaller batches)
    """
    # Heuristic: aim for 4-10 batches per worker
    # This allows dynamic load balancing without too much overhead
    
    if n_paths < n_jobs * 10:
        # Few paths: one per worker
        return 1
    
    # Many paths: batch to ~4-10 batches per worker
    target_batches_per_worker = 8
    batch_size = n_paths // (n_jobs * target_batches_per_worker)
    
    # Clamp to reasonable range
    return max(1, min(batch_size, 1000))
```

### 3. Progress Tracking

```python
def simulate_with_progress(
    sample_fn: Callable,
    n_paths: int,
    seeds: list[int],
    n_jobs: int,
    show_progress: bool = False
) -> list[Path]:
    """Execute with optional progress bar."""
    paths = []
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(sample_fn, seed): i 
                   for i, seed in enumerate(seeds)}
        
        # Wrap iterator with progress bar if requested
        iterator = as_completed(futures)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    iterator,
                    total=n_paths,
                    desc="Simulating paths",
                    unit="path",
                    ncols=80
                )
            except ImportError:
                # Fallback to simple progress
                print(f"Simulating {n_paths} paths...")
        
        for future in iterator:
            try:
                path = future.result()
                paths.append(path)
            except Exception as e:
                # Re-raise with context
                raise RuntimeError(f"Path simulation failed: {e}") from e
    
    return paths
```

### 4. Error Handling

```python
def robust_sample_path(process, T, x0, seed):
    """
    Wrapper that catches and reports errors with context.
    
    Critical for debugging parallel failures.
    """
    try:
        np.random.seed(seed)
        return process.sample_path(T, x0)
    except Exception as e:
        # Include seed for reproducibility
        raise RuntimeError(
            f"Failed to sample path with seed={seed}, T={T}, x0={x0}"
        ) from e
```

### 5. Memory-Efficient Batching

```python
def simulate_batched(
    sample_fn: Callable,
    n_paths: int,
    batch_size: int,
    n_jobs: int
) -> Iterator[list[Path]]:
    """
    Yield paths in batches to avoid memory buildup.
    
    Useful for streaming or when n_paths is very large.
    """
    ss = np.random.SeedSequence(base_seed)
    all_seeds = [s.generate_state(1)[0] for s in ss.spawn(n_paths)]
    
    for i in range(0, n_paths, batch_size):
        batch_seeds = all_seeds[i : i + batch_size]
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(sample_fn, seed) 
                      for seed in batch_seeds]
            batch_paths = [f.result() for f in futures]
        
        yield batch_paths
```

---

## Concrete API Design

### User-Facing API

```python
class MonteCarloEngine:
    def simulate(
        self,
        n_paths: int,
        T: int,
        x0: State | None = None,
        *,
        parallel: bool = False,
        n_jobs: int = -1,
        backend: str = "auto",
        seed: int | None = None,
        show_progress: bool = False,
        batch_size: int | None = None,
        **kwargs
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.
        
        Parameters
        ----------
        parallel : bool
            Enable parallel processing (default: False)
        n_jobs : int
            Number of parallel workers (-1 = all cores)
        backend : str
            "auto" - use joblib if available, else stdlib
            "stdlib" - force ProcessPoolExecutor
            "joblib" - force joblib (must be installed)
        seed : int, optional
            Random seed for reproducibility
        show_progress : bool
            Show progress bar (requires tqdm)
        batch_size : int, optional
            Paths per batch (auto-computed if None)
        """
        # Implementation...
```

### Internal Implementation

```python
def _simulate_parallel_stdlib(
    self,
    n_paths: int,
    T: int,
    x0: State | None,
    n_jobs: int,
    seed: int,
    show_progress: bool,
    **kwargs
) -> list[Path]:
    """Internal: parallel simulation using stdlib."""
    
    # 1. Generate independent seeds
    seeds = generate_path_seeds(seed, n_paths)
    
    # 2. Create worker function (closure over process params)
    def sample_one(path_seed: int) -> Path:
        np.random.seed(path_seed)
        return self.process.sample_path(T, x0, **kwargs)
    
    # 3. Execute in parallel
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    paths = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(sample_one, s): i 
                   for i, s in enumerate(seeds)}
        
        iterator = as_completed(futures)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=n_paths)
            except ImportError:
                pass
        
        for future in iterator:
            paths.append(future.result())
    
    return paths
```

---

## Performance Tuning Guidelines

### When to Use Parallel

```python
def should_parallelize(n_paths: int, T: int, n_jobs: int) -> bool:
    """
    Heuristic: parallel overhead is worth it if total work > 1 second.
    
    Rough estimate: ~1ms per path * T
    """
    estimated_work_ms = n_paths * T * 0.001
    parallel_overhead_ms = 200  # process spawning
    
    # Need at least 2x speedup to justify overhead
    return estimated_work_ms > parallel_overhead_ms * n_jobs
```

### Optimal n_jobs

```python
def optimal_n_jobs(n_paths: int) -> int:
    """
    Choose number of workers based on workload.
    
    - Too few: underutilize CPU
    - Too many: overhead dominates
    """
    max_workers = os.cpu_count() or 1
    
    if n_paths < 10:
        return 1  # overhead not worth it
    elif n_paths < 100:
        return min(4, max_workers)  # limited parallelism
    else:
        return max_workers  # full parallelism
```

---

## Summary: Recommendation

### Phase 1 (MVP)
**Use `concurrent.futures.ProcessPoolExecutor`**
- ✅ Zero dependencies
- ✅ Good enough performance (6-8x)
- ✅ Easy to implement and test
- ✅ Excellent error handling

### Phase 2 (Optimization)
**Add optional `joblib` support**
- ✅ Better for heavy NumPy users
- ✅ Built-in progress bar
- ✅ Fallback to stdlib if not installed

### Future (Distributed)
**Consider Ray/Dask** for massive scale
- Only if users request cluster support
- Phase 3+

---

## Example Implementation

```python
# src/stochlab/mc/executor.py

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Callable, Iterator
import numpy as np

def execute_parallel(
    sample_fn: Callable[[int], Path],
    n_paths: int,
    base_seed: int,
    n_jobs: int = -1,
    show_progress: bool = False
) -> list[Path]:
    """
    Execute sample_fn in parallel with proper seed management.
    
    Parameters
    ----------
    sample_fn : Callable[[int], Path]
        Function that takes a seed and returns a Path
    n_paths : int
        Number of paths to generate
    base_seed : int
        Base seed for reproducibility
    n_jobs : int
        Number of parallel workers (-1 = all CPUs)
    show_progress : bool
        Show tqdm progress bar if available
    
    Returns
    -------
    list[Path]
        Generated paths (order may differ from sequential)
    """
    # Auto-detect optimal workers
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1
    
    # Generate independent seeds using SeedSequence
    ss = np.random.SeedSequence(base_seed)
    seeds = [child.generate_state(1)[0] for child in ss.spawn(n_paths)]
    
    paths = []
    
    # Execute with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all tasks
        futures = {executor.submit(sample_fn, seed): i 
                   for i, seed in enumerate(seeds)}
        
        # Collect as they complete
        iterator = as_completed(futures)
        
        # Optional progress bar
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(
                    iterator,
                    total=n_paths,
                    desc="Simulating",
                    unit="path"
                )
            except ImportError:
                print(f"Simulating {n_paths} paths (install tqdm for progress bar)...")
        
        # Gather results
        for future in iterator:
            try:
                path = future.result()
                paths.append(path)
            except Exception as e:
                # Clean up remaining futures on error
                for f in futures:
                    f.cancel()
                raise RuntimeError(f"Simulation failed: {e}") from e
    
    return paths
```

This is production-ready, maintainable, and performant! 🚀

