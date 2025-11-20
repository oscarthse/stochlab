stochlab Documentation
========================

**stochlab** is a Python library for discrete-time stochastic processes, Monte Carlo simulation, and analytics.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   getting_started
   quick_reference

.. toctree::
   :maxdepth: 2
   :caption: User Guides:

   guides/monte_carlo
   guides/analytics

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Contributing:

   contributing/development_guide

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`