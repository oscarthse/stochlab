# Analytics Module

The `stochlab.analytics` package provides **closed-form results** for finite, discrete-time Markov chains. Every routine is implemented exactly as it appears in standard probability texts, but with careful validation and numerical safeguards so you can rely on the results even if you are still building intuition.

The functions described here live in `stochlab.analytics.markov` and are re-exported from `stochlab.analytics`.

---

## Stationary Distribution

### Definition

For a transition matrix $P \in \mathbb{R}^{n \times n}$ (row-stochastic: non-negative rows summing to one) a **stationary distribution** is a probability vector $\pi \in \mathbb{R}^n$ satisfying

```{math}
\pi P = \pi, \qquad \sum_{i=0}^{n-1} \pi_i = 1, \qquad \pi_i \ge 0.
```

Intuitively, if $X_0 \sim \pi$ then $X_t \sim \pi$ for all $t \ge 0$.

### Uniqueness

- If the chain is **irreducible and aperiodic**, $\pi$ exists and is unique.
- If the chain is **reducible**, there can be infinitely many stationary distributions; the solver returns one valid solution (minimum-norm least-squares solution) and reports the residual.

### Algorithm

`stationary_distribution()` solves the linear system

```{math}
\begin{bmatrix}
P^\top - I \\ 1^\top
\end{bmatrix} \pi =
\begin{bmatrix}
0 \\ 1
\end{bmatrix}
```

using `numpy.linalg.lstsq`, then clips tiny negative values (< `tol`) caused by numerical round-off and renormalizes. Residuals are reported so you can verify the solution quality.

### Usage

```python
import numpy as np
from stochlab.analytics import stationary_distribution

P = np.array([[0.7, 0.3],
              [0.4, 0.6]])
result = stationary_distribution(P)

print(result.distribution)  # array([4/7, 3/7])
print(result.residual)      # L1 norm ||πP - π||
```

When passing a `MarkovChain`, the returned object also includes state labels:

```python
from stochlab.models import MarkovChain

mc = MarkovChain.from_transition_matrix(["Bull", "Bear"], P)
result = stationary_distribution(mc)
print(result.states)        # ('Bull', 'Bear')
```

---

## Hitting Times

### Definition

Given a target set $T \subseteq S$, the **hitting time** is the random variable

```{math}
\tau_T = \inf \{ t \ge 0 : X_t \in T \}.
```

We are interested in its expectation $h_i = \mathbb{E}[\tau_T \mid X_0 = s_i]$.

### Formula

Partition the state space into target and non-target states. Let $Q$ be the submatrix of $P$ restricted to the non-target indices (i.e., transitions that stay outside $T$). The hitting time vector on those states solves

```{math}
h = (I - Q)^{-1} \mathbf{1}.
```

Target states have $h_i = 0$ by definition, since we have already “hit” $T$.

### Assumptions & Failure Modes

- The matrix $I - Q$ must be invertible. If target states are unreachable from some starting state, the solver raises `RuntimeError` (and you can inspect your transition graph).
- Targets do **not** need to be absorbing. The expectation is computed for the first entrance into $T$, regardless of whether the process can subsequently leave.

### Usage

```python
from stochlab.analytics import hitting_times

result = hitting_times(mc, targets=["Bear"])
print(result.times)          # Expected steps to reach "Bear" from each state
print(result.target_mask)    # Boolean array indicating which states are targets
```

---

## Absorption Probabilities & Times

### Setup

Split the states into **transient** indices $T$ and **absorbing** indices $A$. After optionally reordering states, the transition matrix takes the canonical block form

```{math}
P = \begin{bmatrix}
Q & R \\
0 & I
\end{bmatrix},
```

where:

- $Q$ captures transitions among transient states.
- $R$ captures transitions from transient to absorbing states.
- Absorbing states have identity rows ($P_{ii} = 1$, $P_{ij} = 0$ for $j \ne i$).

### Fundamental Matrix

The **fundamental matrix** is

```{math}
N = (I - Q)^{-1} = I + Q + Q^2 + \cdots,
```

and encodes the expected number of visits: $N_{ij}$ is the expected number of times the chain is in transient state $j$ starting from transient state $i$.

### Absorption Probabilities

The matrix of absorption probabilities is

```{math}
B = N R,
```

where $B_{ik}$ is the probability that a trajectory starting in transient state $i$ is eventually absorbed in absorbing state $k$.

### Expected Time to Absorption

The vector of expected absorption times is

```{math}
t = N \mathbf{1},
```

which gives $\mathbb{E}[\tau_A \mid X_0 = s_i]$ for each transient state $i$.

### Usage

```python
from stochlab.analytics import absorption_probabilities

result = absorption_probabilities(
    mc.P,
    transient_states=["Active", "Dormant"],
    absorbing_states=["Churned"],
    state_space=mc.state_space,
)

print(result.probabilities)   # Shape: (n_transient, n_absorbing)
print(result.expected_steps)  # Expected steps before absorption
```

The function checks that the supplied absorbing states really are absorbing (rows equal to a canonical basis vector) and raises informative errors otherwise.

---

## Numerical & Modeling Assumptions

1. **Row-Stochastic Matrices**  
   All matrices must have non-negative entries with each row summing to one. The validators allow tiny numerical noise (atol $=10^{-8}$) to accommodate floating point rounding.

2. **Finite State Spaces**  
   The analytics layer currently targets finite state spaces, matching the core library design. Infinite-state processes (e.g., unbounded queues) need to be truncated before applying these formulas.

3. **Singular Systems**  
   If $(I - Q)$ is singular (unreachable targets, non-absorbing structures), the routines raise `RuntimeError` so you can diagnose the Markov graph rather than silently returning misleading numbers.

4. **Label vs Index Inputs**  
   Every function accepts either integer indices or state labels. When you pass a `MarkovChain`, the labels come directly from its `StateSpace`. When you pass a raw matrix, supply `state_space=...` if you want to refer to states by label.

---

## API Cheatsheet

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `stationary_distribution(obj, *, method="lstsq", tol=1e-12)` | `obj` = `MarkovChain` or `np.ndarray` | `StationaryResult` | Solves $\pi P = \pi$ with normalization and residual check. |
| `hitting_times(obj, targets, *, state_space=None)` | `targets` indices or labels | `HittingTimesResult` | Computes $\mathbb{E}[\tau_T]$ via $(I-Q)^{-1} \mathbf{1}$. |
| `absorption_probabilities(obj, *, transient_states, absorbing_states, state_space=None)` | Partition states explicitly | `AbsorptionResult` | Returns $B = (I-Q)^{-1}R$ and $t = (I-Q)^{-1}\mathbf{1}$. |

Each result dataclass includes both the numeric arrays and human-readable metadata (state labels, masks), making it easy to inspect outputs or turn them into pandas objects for downstream analysis.

