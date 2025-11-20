# Getting Started

## Installation

### Prerequisites
- Python 3.11 or higher
- pip or uv package manager

### Install from Source

```bash
# Clone the repository
git clone https://github.com/oscarthse/stochlab.git
cd stochlab

# Install with uv (recommended)
uv sync

# Or install in development mode
uv pip install -e .
```

## Your First Simulation

Let's create and simulate a simple 2-state Markov chain representing market regimes:

```python
import numpy as np
from stochlab.core import StateSpace
from stochlab.models import MarkovChain

# Step 1: Define states
states = ["Bull", "Bear"]

# Step 2: Define transition probabilities
P = np.array([
    [0.7, 0.3],  # Bull -> Bull: 70%, Bull -> Bear: 30%
    [0.4, 0.6]   # Bear -> Bull: 40%, Bear -> Bear: 60%
])

# Step 3: Create the Markov chain
mc = MarkovChain.from_transition_matrix(states, P)

# Step 4: Simulate a single path
path = mc.sample_path(T=10, x0="Bull")
print(f"Path: {list(path.states)}")

# Step 5: Run Monte Carlo simulation
result = mc.simulate_paths(n_paths=1000, T=100)
print(f"Simulated {len(result)} paths of length {len(result.paths[0])}")
```

## Understanding the Output

### Single Path
A `Path` object contains:
- `times`: Array of time points [0, 1, 2, ..., T]
- `states`: Array of state values at each time
- `extras`: Dictionary for optional metadata

```python
print(f"Times: {path.times}")
print(f"States: {path.states}")
print(f"State at t=5: {path[5]}")
```

### Simulation Results
A `SimulationResult` contains multiple paths and analysis methods:

```python
# Convert to DataFrame for analysis
df = result.to_dataframe()
print(df.head())

# Analyze state distribution at specific time
dist_t50 = result.state_distribution(t=50)
print(f"Distribution at t=50: {dist_t50}")
```

## Core Concepts

### State Space
The foundation of all stochastic processes in stochlab:

```python
from stochlab.core import StateSpace

# Create state space
ss = StateSpace(["A", "B", "C"])

# Access properties
print(f"Number of states: {len(ss)}")
print(f"Index of 'B': {ss.index('B')}")
print(f"State at index 2: {ss.state(2)}")
print(f"Contains 'A': {'A' in ss}")
```

### Process Interface
All models implement the `StochasticProcess` interface:

```python
# Every process has a state space
print(f"State space: {mc.state_space.states}")

# Every process can generate paths
path = mc.sample_path(T=20)

# Every process supports Monte Carlo
result = mc.simulate_paths(n_paths=100, T=50)
```

## Monte Carlo Simulation

For advanced Monte Carlo features including parallel execution and memory optimization:

```python
from stochlab.mc import MonteCarloEngine

# Create engine
engine = MonteCarloEngine(mc)

# Simple parallel simulation
result = engine.simulate(
    n_paths=100000,
    T=100,
    parallel=True,  # Uses all CPU cores
    seed=42         # Reproducible
)

# Estimate expectations
def final_state_is_b(path):
    return 1.0 if path.states[-1] == "B" else 0.0

stats = engine.estimate(
    estimator_fn=final_state_is_b,
    n_paths=10000,
    T=100,
    parallel=True
)

print(f"P(X_100 = B) = {stats.mean:.4f} ± {stats.stderr:.4f}")
print(f"95% CI: {stats.confidence_interval}")
```

**Key Features**:
- **6-8x speedup** with parallel execution
- **90-99% memory reduction** with efficient modes
- **Reproducible** results with seed management
- **Progress tracking** for long simulations

See the [Monte Carlo Guide](guides/monte_carlo.md) for complete documentation.

## Next Steps

1. **Monte Carlo Simulation**: Learn about [high-performance parallel simulation](guides/monte_carlo.md)
2. **Analytics**: Explore [Markov chain analytics](guides/analytics.md) for computing stationary distributions and more
3. **Quick Reference**: See the [quick reference](quick_reference.md) for common operations
4. **API Reference**: Browse the complete [API Documentation](api/index.rst)