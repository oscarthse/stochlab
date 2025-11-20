# Stochlab Documentation

Complete documentation for the stochlab library.

## 📚 Documentation Structure

```
docs/
├── getting_started.md                 # Quick introduction
├── quick_reference.md                 # Common operations cheat sheet
├── guides/                            # User guides
│   ├── monte_carlo.md                 # Complete Monte Carlo guide ⭐
│   └── analytics.md                   # Markov chain analytics
├── api/                               # API reference
│   └── index.rst                      # Auto-generated documentation
├── technical/                         # Technical documentation
│   ├── monte_carlo_design.md          # MC engine design doc
│   ├── parallelization_analysis.md    # Parallel execution deep dive
│   └── implementation_summary.md      # What was built and why
└── contributing/                      # Developer documentation
    └── development_guide.md           # Contributing guidelines
```

### Quick Access

**Getting Started**:
- **[Getting Started](getting_started.md)** - Quick introduction and basic usage
- **[Quick Reference](quick_reference.md)** - Common operations cheat sheet

**User Guides**:
- **[Monte Carlo Guide](guides/monte_carlo.md)** ⭐ - Complete guide to parallel simulation
- **[Analytics Guide](guides/analytics.md)** - Markov chain analytics and theory

**API Documentation**:
- **[API Reference](api/index.rst)** - Complete API documentation (auto-generated from docstrings)

**Technical Documentation**:
- **[Monte Carlo Design](technical/monte_carlo_design.md)** - Comprehensive design document
- **[Parallelization Analysis](technical/parallelization_analysis.md)** - Deep dive into parallel execution
- **[Implementation Summary](technical/implementation_summary.md)** - What was built and why

**Contributing**:
- **[Development Guide](contributing/development_guide.md)** - Contributing guidelines and workflow

## 🚀 Quick Links

### For Users

**New to stochlab?** Start here:
1. Read [Getting Started](getting_started.md)
2. Try the examples in `demo_random_walk.py` and `demo_monte_carlo.py`
3. Browse the [Quick Reference](quick_reference.md)

**Want high-performance simulation?** 
- Read the [Monte Carlo Guide](guides/monte_carlo.md)
- Learn about parallel execution, memory modes, and optimization

**Need analytics?**
- Check out the [Analytics Guide](guides/analytics.md)
- Learn to compute stationary distributions, hitting times, and more

### For Developers

**Contributing to stochlab?**
1. Read the [Development Guide](contributing/development_guide.md)
2. Review the technical docs to understand the architecture
3. Run tests with `pytest`

**Understanding the codebase?**
- **Monte Carlo Engine**: See [monte_carlo_design.md](technical/monte_carlo_design.md) and [parallelization_analysis.md](technical/parallelization_analysis.md)
- **Core Architecture**: See docstrings in `src/stochlab/core/`
- **Models**: See docstrings in `src/stochlab/models/`

## 📖 Documentation By Topic

### Basic Usage

- Creating processes: [Getting Started § Models](getting_started.md#models)
- Simulating paths: [Getting Started § Process Interface](getting_started.md#process-interface)
- Working with results: [Quick Reference § Results](quick_reference.md)

### Monte Carlo Simulation

- **Quick start**: [Monte Carlo Guide § Quick Start](guides/monte_carlo.md#quick-start)
- **Parallel execution**: [Monte Carlo Guide § Parallel Simulation](guides/monte_carlo.md#1-parallel-simulation)
- **Memory optimization**: [Monte Carlo Guide § Memory-Efficient Modes](guides/monte_carlo.md#2-memory-efficient-modes)
- **Reproducibility**: [Monte Carlo Guide § Reproducible Seeds](guides/monte_carlo.md#3-reproducible-seeds)
- **Estimating expectations**: [Monte Carlo Guide § Estimating Expectations](guides/monte_carlo.md#estimating-expectations)
- **Performance tips**: [Monte Carlo Guide § Performance Tips](guides/monte_carlo.md#performance-tips)
- **Troubleshooting**: [Monte Carlo Guide § Troubleshooting](guides/monte_carlo.md#troubleshooting)

### Analytics

- Stationary distributions: [Analytics § Stationary Distribution](guides/analytics.md)
- Hitting times: [Analytics § Hitting Times](guides/analytics.md)
- Absorption probabilities: [Analytics § Absorption](guides/analytics.md)

### Technical Details

- **Why batching matters**: [Parallelization Analysis § Batching](technical/parallelization_analysis.md)
- **Worker initialization**: [Parallelization Analysis § Worker Init](technical/parallelization_analysis.md)
- **Seed management**: [Monte Carlo Design § Seeding](technical/monte_carlo_design.md)
- **Return modes**: [Monte Carlo Design § Return Modes](technical/monte_carlo_design.md)
- **Performance analysis**: [Implementation Summary § Performance](technical/implementation_summary.md)

## 🔍 Finding Documentation

### By Feature

| Feature | Documentation |
|---------|--------------|
| Basic simulation | [Getting Started](getting_started.md) |
| Parallel Monte Carlo | [Monte Carlo Guide](guides/monte_carlo.md) |
| Markov chain analytics | [Analytics](guides/analytics.md) |
| API reference | [API Docs](api/index.rst) |
| Contributing | [Development Guide](contributing/development_guide.md) |

### By Process Type

| Process | Documentation |
|---------|--------------|
| MarkovChain | [Getting Started](getting_started.md), [Analytics](guides/analytics.md) |
| RandomWalk | [Getting Started](getting_started.md) |
| MM1Queue | [Getting Started](getting_started.md) |

### By Use Case

| Use Case | Documentation |
|----------|--------------|
| "I want to run 1M simulations fast" | [Monte Carlo Guide § Parallel](guides/monte_carlo.md#1-parallel-simulation) |
| "I'm running out of memory" | [Monte Carlo Guide § Memory Modes](guides/monte_carlo.md#2-memory-efficient-modes) |
| "I need reproducible results" | [Monte Carlo Guide § Seeds](guides/monte_carlo.md#3-reproducible-seeds) |
| "How do I estimate probabilities?" | [Monte Carlo Guide § Estimating](guides/monte_carlo.md#estimating-expectations) |
| "I want steady-state distribution" | [Analytics § Stationary](guides/analytics.md) |
| "I need to contribute code" | [Development Guide](contributing/development_guide.md) |

## 📝 Documentation Quality

All user-facing code includes:

✅ **Comprehensive docstrings** - Every public class and function  
✅ **Type hints** - Full type coverage  
✅ **Examples** - Docstring examples for key functions  
✅ **User guides** - Step-by-step tutorials  
✅ **API reference** - Auto-generated from docstrings  
✅ **Technical docs** - Design decisions and architecture  

## 🛠️ Building Documentation

To build the HTML documentation locally:

```bash
cd docs
make html
open _build/html/index.html
```

Or using `sphinx-build` directly:

```bash
sphinx-build -b html docs docs/_build/html
```

## 📊 Documentation Coverage

- **Core modules**: 100% documented
- **Monte Carlo engine**: 100% documented  
- **Models**: 100% documented
- **Analytics**: 100% documented
- **User guides**: Complete
- **Technical docs**: Complete
- **Examples**: Complete

## 🎯 Documentation Goals

Our documentation aims to be:

1. **Accessible** - Clear for beginners, detailed for experts
2. **Complete** - Cover all features and edge cases
3. **Practical** - Plenty of examples and use cases
4. **Up-to-date** - Updated with every feature addition
5. **Searchable** - Easy to find what you need

## 💡 Documentation Principles

- **Show, don't just tell** - Every concept has a code example
- **Multiple levels** - Quick start, user guide, API reference, technical deep dive
- **Real use cases** - Examples solve real problems
- **Performance guidance** - When to use what
- **Troubleshooting** - Common issues and solutions

---

**Happy reading!** 📖

If you find any documentation gaps or errors, please [open an issue](https://github.com/oscarthse/stochlab/issues).

