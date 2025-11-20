# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2024-11-20

### Added
- High-performance Monte Carlo simulation engine (`stochlab.mc`)
  - `MonteCarloEngine` class for batch simulations and estimations
  - Parallel execution support using `ProcessPoolExecutor`
  - Multiple return modes: `paths`, `values`, `stats` for memory efficiency
  - Task batching to minimize overhead
  - Worker initialization for optimal serialization
  - Reproducible seeding with `np.random.SeedSequence`
  - Progress bar support via `tqdm`
  - Confidence intervals for estimations (requires `scipy`)
- Comprehensive documentation
  - Monte Carlo Simulation Guide
  - Technical design documents
  - Parallelization analysis
  - Implementation summary
- Documentation badges and links in README
- Live documentation at https://oscarthse.github.io/stochlab/

### Changed
- Reorganized documentation into cleaner directory structure:
  - `docs/guides/` - User guides
  - `docs/technical/` - Technical documentation
  - `docs/contributing/` - Contributor documentation
- Updated all documentation links and references
- Improved README with prominent documentation section
- Updated pyproject.toml documentation URL to point to GitHub Pages

### Fixed
- GitHub Actions workflow deprecation warning (updated `upload-pages-artifact` to v3)
- Sequential simulation reproducibility in `MonteCarloEngine`

## [0.1.0] - 2024-11-01

### Added
- Core stochastic process abstractions
  - `StochasticProcess` abstract base class
  - `StateSpace` for discrete state management
  - `Path` for sample path representation
  - `SimulationResult` for collecting multiple paths
- Concrete models
  - `MarkovChain` with finite transition matrix
  - `RandomWalk` with reflecting boundaries
  - `MM1Queue` (M/M/1 queueing model)
- Analytics module for Markov chains
  - Stationary distribution calculation
  - Hitting time analysis
  - Absorption probabilities and times
- Basic Monte Carlo support via `simulate_paths()`
- Comprehensive test suite
- Sphinx documentation with Read the Docs theme
- Type hints throughout codebase
- Development tooling (black, ruff, mypy, pytest)

### Initial Release
- Python 3.11+ support
- NumPy and Pandas integration
- MIT License

