# stochlab

*A small, serious library for discrete-time stochastic processes, Monte Carlo simulation, and analytics.*

---

## 🚀 What is stochlab?

`stochlab` is a Python library for **finite / countable discrete-time stochastic processes**.

It’s designed to be:

* **Mathematically coherent** – built around clear objects:
    * a finite **state space** $ S = \{s_0,\dots,s_{n-1}\} $,
    * a **discrete-time process** $(X_t)_{t=0}^T$ with values in $ S $,
    * **sample paths** and **Monte Carlo experiments**.
* **Engineer-friendly** – clean abstractions, type hints, tests, and a modular layout ready to grow.
* **Practical** – aims at use cases like:
    * Markov chains (credit ratings, user journeys, regimes),
    * simple queueing models,
    * branching processes (viral growth, extinction),
    * Monte Carlo estimation + variance reduction,
    * DataFrame summaries and interactive plots (later phases).

Right now the project is in **early stages**: Phase 1 focuses on solid core abstractions.

---

## 🧠 Core ideas (math, lightly)

The library is built around a few standard concepts:

* A **finite state space**:
    $$
    S = \{s_0, s_1, \dots, s_{n-1}\}
    $$
    represented in code by a `StateSpace` object, which provides a clean mapping between **labels** (like `"A"`, `"BBB"`, `(regime,vol)`) and **integer indices** $0, 1, \dots, n-1$.

* A **discrete-time stochastic process**:
    $$
    (X_t)_{t=0}^T,\quad X_t \in S,
    $$
    represented by subclasses of an abstract `StochasticProcess` base class. A process is required to be able to **simulate a sample path** of length $T+1$.

* A **sample path** (one realization of the process):
    $$
    \omega = (X_0(\omega), X_1(\omega), \dots, X_T(\omega)),
    $$
    represented by a `Path` object holding arrays of times and states.

* A **simulation result**: a collection of many paths, used to build empirical distributions, statistics, and tables, wrapped in a `SimulationResult` object.

In later phases, we’ll add:

* concrete models such as `MarkovChain`, `MM1Queue`, `GaltonWatsonProcess`,
* analytical tools (e.g. stationary distributions, absorption probabilities, queue metrics),
* Monte Carlo engines with variance reduction and reporting/visualization tools.
