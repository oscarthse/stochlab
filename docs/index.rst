stochlab Documentation
=====================

**stochlab** is a Python library for discrete-time stochastic processes, Monte Carlo simulation, and analytics.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   quick_reference
   architecture
   api/index
   examples/financial_modeling

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from stochlab.core import StateSpace
   from stochlab.models import MarkovChain

   # Create a simple 2-state Markov chain
   states = ["Bull", "Bear"]
   P = np.array([[0.7, 0.3], [0.4, 0.6]])
   mc = MarkovChain.from_transition_matrix(states, P)

   # Simulate paths
   result = mc.simulate_paths(n_paths=1000, T=100)
   df = result.to_dataframe()

Key Features
------------

* **Mathematical rigor**: Built around clean abstractions for finite state spaces and stochastic processes
* **Type safety**: Full type hints with modern Python 3.11+ syntax
* **Extensible**: Layered architecture ready for additional models and analytics
* **Performance**: Efficient numpy-based simulation with immutable trajectories

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`