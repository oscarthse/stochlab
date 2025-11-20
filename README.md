# stochlab

*A compact library for discrete-time stochastic processes, Monte Carlo simulation, and analytics.*

---

## 🚀 What is stochlab?

`stochlab` is a Python library for **finite / countable discrete-time stochastic processes**.

It's designed to be:

* **Coherence** – built around clear Mathematical objects:
    * a finite **state space** $S = \{s_0, s_1, \ldots, s_{n-1}\}$,
    * a **discrete-time process** $(X_t)_{t=0}^T$ with values in $S$,
    * **sample paths** and **Monte Carlo experiments**.
* **OOP** - Models/Processes are defined as objects inheriting from the parent class StochasticProcess
* **Engineer-friendly** – clean abstractions, type hints, tests, and a modular layout ready to grow.
* **Practical** – aims at use cases like:
    * Markov chains (credit ratings, user journeys, regimes),
    * simple queueing models,
    * branching processes (viral growth, extinction),
    * Monte Carlo estimation + variance reduction,
    * DataFrame summaries and interactive plots (later phases).

Right now the project is in **early stages**: Phase 1 focuses on solid core abstractions with the first concrete model (MarkovChain) implemented and tested.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/oscarthse/stochlab.git
cd stochlab

# Install with uv (recommended)
uv sync

# Or install in development mode
uv pip install -e .
```

### Basic Usage

```python
import numpy as np
from stochlab.core import StateSpace
from stochlab.models import MarkovChain

# Define a simple 2-state Markov chain
states = ["Bull", "Bear"]
P = np.array([[0.7, 0.3],   # Bull -> Bull: 70%, Bull -> Bear: 30%
              [0.4, 0.6]])  # Bear -> Bull: 40%, Bear -> Bear: 60%

# Create the Markov chain
mc = MarkovChain.from_transition_matrix(states, P)

# Simulate a single path
path = mc.sample_path(T=100, x0="Bull")
print(f"Path length: {len(path)}")
print(f"Final state: {path[-1]}")

# Run Monte Carlo simulation
result = mc.simulate_paths(n_paths=1000, T=100)
df = result.to_dataframe()
print(f"Simulated {len(result)} paths")

# Analyze state distribution at time T=50
dist = result.state_distribution(t=50)
print(f"State distribution at t=50: {dist}")
```

---

## Mathematical Foundation

The library implements discrete-time stochastic processes with mathematical precision:

### State Space
Every process operates on a finite state space:
```math
S = \{s_0, s_1, \ldots, s_{n-1}\}
```

States can be any hashable Python objects: strings, numbers, tuples, or custom types. The `StateSpace` class provides a clean mapping between **labels** (like `"A"`, `"BBB"`, `(regime,vol)`) and **integer indices** $0, 1, \ldots, n-1$.

### Stochastic Process
A discrete-time stochastic process is a sequence of random variables:
```math
(X_t)_{t=0}^T, \quad X_t \in S
```

Each model implements the `sample_path()` method to generate realizations of length $T+1$.

### Sample Paths
A sample path represents one possible evolution of the process:
```math
\omega = (X_0(\omega), X_1(\omega), \ldots, X_T(\omega))
```

Represented by a `Path` object holding arrays of times and states, with optional metadata in the `extras` dictionary.

### Monte Carlo Simulation
Generate multiple independent paths to estimate:
- State probabilities: $P(X_t = s)$
- Path functionals: $E[f(X_0, \ldots, X_T)]$
- Hitting times, absorption probabilities, etc.

The `SimulationResult` class collects multiple paths and provides analysis methods like DataFrame conversion and empirical state distributions.

---

## 🧮 Analytical Solutions (Finite Markov Chains)

The `stochlab.analytics` module packages classical finite-state Markov chain formulas with validation and friendly outputs. Every routine accepts either a raw `numpy.ndarray` transition matrix or a `MarkovChain` instance (preserving state labels).

### Stationary Distributions

Find $\pi$ satisfying

```math
\pi P = \pi, \qquad \sum_{i=0}^{n-1} \pi_i = 1.
```

We solve

```math
\begin{bmatrix}
P^{\top} - I \\
\mathbf{1}^{\top}
\end{bmatrix} \pi =
\begin{bmatrix}
0 \\
1
\end{bmatrix}
```

via least squares, clip tiny negatives, renormalize, and report $\|\pi P - \pi\|_1$ so you can judge accuracy. Irreducible aperiodic chains yield a unique solution; reducible chains may have multiple stationary distributions, and the solver returns one valid vector.

```python
from stochlab.analytics import stationary_distribution

res = stationary_distribution(mc)
print(res.distribution)
print(res.residual)
print(res.states)
```

### Expected Hitting Times

For a target set $T$, the hitting time $\tau_T = \inf\{t \ge 0 : X_t \in T\}$ has expectation

```math
h = (I - Q)^{-1} \mathbf{1},
```

where $Q$ restricts $P$ to non-target states. Targets themselves have $h_i = 0$. If $(I - Q)$ is singular (targets unreachable) the implementation raises a clear `RuntimeError`.

```python
from stochlab.analytics import hitting_times
res = hitting_times(mc, targets=["Bear"])
print(res.times)
print(res.target_mask)
```

### Absorption Probabilities & Times

Partition states into transient $T$ and absorbing $A$ so that

```math
P = \begin{bmatrix}
Q & R \\
0 & I
\end{bmatrix}.
```

The fundamental matrix $N = (I - Q)^{-1}$ yields

- $B = N R$: probability of ending in each absorbing state.
- $t = N \mathbf{1}$: expected steps before absorption.

```python
from stochlab.analytics import absorption_probabilities

res = absorption_probabilities(
    mc.P,
    transient_states=["Active", "Dormant"],
    absorbing_states=["Churned"],
    state_space=mc.state_space,
)
print(res.probabilities)
print(res.expected_steps)
```

Rows supplied as absorbing are validated to ensure they are truly absorbing (identity rows), preventing silent misuse.

---

## 🏗️ Architecture

`stochlab` follows a **layered architecture** designed for extensibility and mathematical rigor:

### Core Layer (`stochlab.core`)
Fundamental building blocks that all models depend on:

- **`StateSpace`**: Manages finite state sets with bijective label ↔ index mapping
- **`Path`**: Represents a single trajectory $(X_0, X_1, \ldots, X_T)$ with validation and immutability
- **`StochasticProcess`**: Abstract base class defining the interface all models must implement
- **`SimulationResult`**: Collection of paths with analysis methods (DataFrame conversion, empirical distributions)

### Models Layer (`stochlab.models`)
Concrete implementations of stochastic processes:

- **`MarkovChain`**: Finite-state, time-homogeneous Markov chains with validation and convenience constructors
- **`RandomWalk`**: Reflecting random walk on bounded integers
- **`MM1Queue`**: Capped M/M/1 queue with exponential event clocks and overflow detection
- *Coming soon*: `GaltonWatsonProcess`, birth–death processes

### Analytics Layer (`stochlab.analytics`)
Finite-state analytical routines:

- `stationary_distribution`, `hitting_times`, `absorption_probabilities`
- Dataclass outputs with state labels, residual diagnostics, and model-safe validation

### Future Layers
- **`stochlab.mc`**: Advanced Monte Carlo engines with variance reduction
- **`stochlab.reporting`**: Visualization and reporting tools

---

## 🛠️ Development

### Development Setup

```bash
# Install development dependencies
make dev

# Run tests
make test

# Code quality checks
make format    # Format with black
make lint      # Lint with ruff
make typecheck # Type check with mypy
make all-checks # Run all quality checks
```

### Project Structure

```
stochlab/
├── src/stochlab/           # Main package
│   ├── core/              # Core abstractions
│   ├── models/            # Concrete models
│   └── __init__.py
├── tests/                 # Test suite
├── Makefile              # Development automation
├── pyproject.toml        # Project configuration
└── README.md
```

### Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/new-model`
3. **Implement** your changes following the architecture
4. **Add tests** for new functionality
5. **Run** `make all-checks` to ensure quality
6. **Submit** a pull request

---

## 🎯 Roadmap

### Phase 1: Core Foundation ✅
- [x] `StateSpace`, `Path`, `SimulationResult` classes
- [x] `StochasticProcess` abstract interface
- [x] `MarkovChain` implementation with comprehensive validation
- [x] Comprehensive test suite (19 tests passing)
- [x] Development workflow (Makefile, linting, formatting)

### Phase 2: Model Expansion
- [x] `RandomWalk` - symmetric and asymmetric random walks
- [x] `MM1Queue` - single-server queueing model
- [ ] `GaltonWatsonProcess` - branching processes
- [ ] Birth-death processes

### Phase 3: Advanced Monte Carlo
- [ ] `MonteCarloEngine` with variance reduction
- [ ] Antithetic variates, control variates
- [ ] Parallel simulation support
- [ ] Rare event simulation techniques

### Phase 4: Analytics & Theory
- [x] Markov chain analytics (stationary distributions, hitting times, absorption)
- [ ] Queueing metrics (L, W, Lq, Wq)
- [ ] Branching process extinction probabilities
- [ ] Spectral analysis tools

### Phase 5: Visualization & Reporting
- [ ] Interactive plots with Plotly
- [ ] Jupyter notebook integration
- [ ] Statistical summaries and reports
- [ ] Export to common formats (CSV, JSON, HDF5)

---

## 📚 Examples & Use Cases

### Financial Modeling
```python
# Credit rating transitions
ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
P = credit_transition_matrix()  # Your transition matrix
mc = MarkovChain.from_transition_matrix(ratings, P)

# Simulate rating paths
result = mc.simulate_paths(n_paths=10000, T=60)  # 5 years monthly
default_prob = result.state_distribution(t=60)["D"]
```

### Market Regime Modeling
```python
# Bull/Bear market transitions
states = ["Bull", "Bear", "Sideways"]
P = np.array([[0.85, 0.10, 0.05],
              [0.15, 0.70, 0.15],
              [0.20, 0.20, 0.60]])

mc = MarkovChain.from_transition_matrix(states, P)
result = mc.simulate_paths(n_paths=1000, T=252)  # 1 year daily

# Analyze regime persistence
df = result.to_dataframe()
regime_changes = df.groupby('path_id')['state'].apply(lambda x: (x != x.shift()).sum())
```

### Epidemiological Modeling
```python
# Simple SIR-like model (future implementation)
states = ["Susceptible", "Infected", "Recovered"]
# Custom transition probabilities based on infection/recovery rates
```

### Operations Research
```python
# Queue length analysis (future MM1Queue implementation)
queue = MM1Queue(arrival_rate=0.8, service_rate=1.0)
result = queue.simulate_paths(n_paths=1000, T=1000)
# Analyze steady-state queue length distribution
```

---

## 🔬 Technical Features

### Robust Validation
- **Transition matrices**: Validates stochastic properties (rows sum to 1, non-negative)
- **State spaces**: Ensures uniqueness and proper label-index mapping
- **Paths**: Enforces time-state array consistency and immutability
- **Initial distributions**: Validates probability distribution properties

### Performance Optimizations
- **Efficient numpy arrays**: Core simulation uses vectorized operations
- **Memory optimization**: `slots=True` on dataclasses reduces memory footprint
- **Immutable trajectories**: Read-only arrays prevent accidental modification

### Type Safety
- **Modern Python typing**: Full type hints with Python 3.11+ syntax
- **Abstract interfaces**: Clear contracts between components
- **Runtime validation**: Comprehensive error checking with informative messages

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🤝 Acknowledgments

Built with modern Python tools:
- [uv](https://github.com/astral-sh/uv) for dependency management
- [pytest](https://pytest.org/) for testing
- [black](https://black.readthedocs.io/) for code formatting
- [ruff](https://ruff.rs/) for linting
- [mypy](https://mypy.readthedocs.io/) for type checking
- [numpy](https://numpy.org/) for numerical computing
- [pandas](https://pandas.pydata.org/) for data analysis
