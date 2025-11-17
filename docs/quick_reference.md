# Quick Reference

## Essential Imports
```python
from stochlab.core import StateSpace, Path, SimulationResult
from stochlab.models import MarkovChain
import numpy as np
```

## Basic Workflow

### 1. Create State Space
```python
states = StateSpace(["A", "B", "C"])
```

### 2. Build Markov Chain
```python
P = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.4, 0.3], 
              [0.1, 0.1, 0.8]])
mc = MarkovChain.from_transition_matrix(["A", "B", "C"], P)
```

### 3. Simulate Paths
```python
# Single path
path = mc.sample_path(T=100, x0="A")

# Multiple paths
result = mc.simulate_paths(n_paths=1000, T=100)
```

### 4. Analyze Results
```python
# Convert to DataFrame
df = result.to_dataframe()

# State distribution at time t
dist = result.state_distribution(t=50)

# Access individual paths
first_path = result.paths[0]
final_state = first_path[-1]
```

## Key Methods

| Class | Method | Purpose |
|-------|--------|---------|
| `StateSpace` | `index(state)` | Get index of state |
| `StateSpace` | `state(idx)` | Get state at index |
| `MarkovChain` | `sample_path(T, x0)` | Generate single trajectory |
| `MarkovChain` | `simulate_paths(n_paths, T)` | Monte Carlo simulation |
| `Path` | `path[i]` | Get state at time i |
| `SimulationResult` | `to_dataframe()` | Convert to pandas DataFrame |
| `SimulationResult` | `state_distribution(t)` | Empirical distribution |