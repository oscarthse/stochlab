# stochlab Roadmap

This document outlines planned features, ideas under consideration, and the long-term vision for stochlab.

---

## Current Version: 0.1.1

✅ Core abstractions (StateSpace, StochasticProcess, Path, SimulationResult)  
✅ Basic models (MarkovChain, RandomWalk, MM1Queue)  
✅ Analytics module (stationary distribution, hitting times, absorption)  
✅ High-performance Monte Carlo engine with parallelization  
✅ Comprehensive documentation

---

## Version 0.2.0 (Next Release)

**Focus**: Visualization & CI/CD

### Planned Features

- **Visualization Module** (`stochlab.viz`)
  - Path plotting (`plot_paths()`)
  - State distribution plots (`plot_distribution()`)
  - Transition diagram visualization
  - Integration with matplotlib and plotly
  
- **CI/CD Pipeline**
  - Automated testing on push/PR (GitHub Actions)
  - Code coverage reporting (codecov.io)
  - Automated PyPI releases on tags
  - Automated documentation deployment

- **Community Infrastructure**
  - Issue templates
  - PR templates
  - Contributing guidelines (root CONTRIBUTING.md)
  - Code of Conduct

---

## Version 0.3.0

**Focus**: More Models & Examples

### Additional Models

- **Birth-Death Processes**
  - Population dynamics
  - Queue length processes
  - Epidemic models (SIS, SIR)
  
- **Branching Processes**
  - Galton-Watson processes
  - Extinction probabilities
  - Multi-type branching

- **Additional Queueing Models**
  - M/M/c (multi-server)
  - M/M/∞ (infinite servers)
  - M/G/1 (general service times)

### Example Gallery

- `examples/` directory with real-world use cases:
  - Finance: Credit rating transitions
  - Biology: Population dynamics
  - Operations: Queueing systems
  - Marketing: Customer journey analysis

### Jupyter Tutorials

- "Getting Started with stochlab"
- "Advanced Monte Carlo Techniques"
- "Building Custom Models"
- "Performance Optimization Guide"

---

## Version 0.4.0

**Focus**: Advanced Monte Carlo

### Variance Reduction Techniques

Extend the Monte Carlo engine with:

- **Antithetic Variates**
  - Negative correlation for variance reduction
  - Automatic detection of suitability
  
- **Control Variates**
  - Use correlated variables with known expectation
  - Automatic coefficient estimation
  
- **Importance Sampling**
  - Change of measure for rare event simulation
  - Adaptive importance sampling
  
- **Stratified Sampling**
  - Partition state space for more efficient sampling
  - Proportional allocation

### API Extension

```python
from stochlab.mc import MonteCarloEngine
from stochlab.mc.variance_reduction import AntitheticVariates, ControlVariates

engine = MonteCarloEngine(process)
result = engine.estimate(
    estimator_fn=my_estimator,
    n_paths=10000,
    variance_reduction=AntitheticVariates(),
    parallel=True
)
```

---

## Version 0.5.0+

**Focus**: Advanced Features

### Continuous-Time Processes

- **Continuous-Time Markov Chains (CTMC)**
  - Uniformization method
  - Matrix exponential computation
  - Transient analysis
  
- **Poisson Processes**
  - Homogeneous Poisson process
  - Non-homogeneous Poisson process
  - Compound Poisson process

### Distributed Computing

- **Ray Backend**
  - True cluster computing
  - Distributed memory
  - Automatic scaling
  
- **Dask Backend**
  - Out-of-core computation
  - Parallel pandas integration
  - Cloud execution (AWS, GCP)

---

## Future Work: Major Features

> **📖 Related Reading**: See [The Inference Landscape](docs/technical/inference_landscape.md) for a comprehensive exploration of inference techniques for stochastic processes.

### 🎯 Hidden Markov Models (HMMs)

**Status**: Design phase  
**Estimated Effort**: 2-3 weeks  
**Target Version**: 0.6.0 - 0.7.0

#### Overview

Hidden Markov Models extend Markov chains with an observation layer, enabling inference of hidden states from observed data. Natural fit with stochlab's architecture.

#### Proposed API

```python
from stochlab.models import HiddenMarkovModel

# Define HMM
hmm = HiddenMarkovModel(
    hidden_states=["Sunny", "Rainy"],
    observations=["Happy", "Grumpy", "Neutral"],
    transition_matrix=P,      # P(X_t | X_{t-1})
    emission_matrix=E,        # P(Y_t | X_t)
    initial_distribution=pi0
)

# Three fundamental problems:

# 1. Evaluation: P(observations | model)
log_prob = hmm.score(observations)

# 2. Decoding: Most likely hidden state sequence
hidden_path = hmm.decode(observations, algorithm='viterbi')

# 3. Learning: Estimate parameters from data
hmm_fitted = HiddenMarkovModel.fit(
    observed_sequences,
    n_hidden_states=3,
    algorithm='baum-welch'
)
```

#### Implementation Phases

**Phase 1: Inference (v0.6.0)**
- `HiddenMarkovModel` class
- Forward algorithm (evaluation/scoring)
- Viterbi algorithm (decoding)
- Forward-Backward algorithm (smoothing)
- Sampling with observations
- Log-space arithmetic for numerical stability

**Phase 2: Learning (v0.7.0)**
- Baum-Welch algorithm (EM for parameter estimation)
- Maximum likelihood estimation
- `fit()` class method
- Cross-validation utilities
- Model selection criteria (AIC, BIC)

**Phase 3: Advanced Features (v0.8.0)**
- Continuous emissions (Gaussian HMMs)
- Semi-supervised learning
- Online learning algorithms
- Particle filtering integration with Monte Carlo engine

#### Use Cases

1. **Finance**: Market regime detection (bull/bear/sideways)
2. **Bioinformatics**: Gene finding, protein structure prediction
3. **NLP**: Part-of-speech tagging, speech recognition
4. **Anomaly Detection**: Identifying unusual patterns
5. **Operations**: Hidden state diagnosis in systems

#### Integration with Existing Features

```python
# Synergy with Monte Carlo engine
from stochlab.mc import MonteCarloEngine

# Use MC for particle filtering
engine = MonteCarloEngine(hmm)
posterior_samples = engine.simulate(
    n_paths=10000,
    condition_on_observations=observed_data
)

# Bootstrap confidence intervals for HMM parameters
hmm_bootstrap = HiddenMarkovModel.bootstrap_fit(
    observed_sequences,
    n_bootstrap=1000,
    parallel=True
)
```

#### Technical Considerations

**Pros**:
- Natural extension of MarkovChain
- High user demand across many fields
- Differentiates stochlab from other libraries
- Enables inference/learning workflows

**Cons**:
- Increases scope beyond simulation
- Non-trivial algorithms (Viterbi, Baum-Welch)
- Numerical stability challenges
- More testing/validation required

**Dependencies**:
- scipy (for optimization in Baum-Welch)
- Possibly optional sklearn integration

#### References & Resources

- Rabiner, L. R. (1989). "A tutorial on hidden Markov models and selected applications in speech recognition"
- Durbin, R. et al. (1998). "Biological Sequence Analysis"
- Stamp, M. (2004). "A Revealing Introduction to Hidden Markov Models"

---

### 🔍 Inference & Learning Module

**Status**: Design & exploration phase  
**Estimated Effort**: 4-6 weeks (incremental)  
**Target Version**: 0.6.0+

#### Overview

Expand stochlab from **simulation-only** (forward problem) to **simulation + inference** (forward + inverse problems). This creates a complete bi-directional workflow.

#### Core Inference Features

**Phase 1: Parameter Estimation (v0.6.0)**

```python
from stochlab.inference import MaximumLikelihoodEstimator, Bootstrap

# Estimate parameters from observed data
mle = MaximumLikelihoodEstimator(model_class=MarkovChain)
estimated_model = mle.fit(observed_paths)

# Bootstrap confidence intervals
bootstrap = BootstrapEstimator(mle, n_bootstrap=1000, parallel=True)
results = bootstrap.fit(observed_paths)
confidence_intervals = results.confidence_interval(0.95)
```

**Features**:
- Maximum Likelihood Estimation for all models
- Bootstrap methods (perfect synergy with MC engine)
- Standard errors and confidence intervals
- Method of moments

**Phase 2: Particle Filtering (v0.7.0)**

```python
from stochlab.inference import ParticleFilter

# Sequential state estimation
pf = ParticleFilter(
    process=markov_chain,
    observation_model=observation_fn,
    n_particles=10000
)

# Online filtering
for y_t in observations:
    pf.update(y_t)
    state_estimate = pf.get_estimate()
```

**Features**:
- Bootstrap particle filter
- Auxiliary particle filter
- Particle smoothing (offline)
- Integration with Monte Carlo engine

**Phase 3: Change Point Detection (v0.8.0)**

```python
from stochlab.inference import ChangePointDetector

# Detect regime changes
detector = ChangePointDetector(model_class=MarkovChain)
result = detector.detect_multiple(observed_path, max_changepoints=5)

change_points = result.change_points
segments = result.models  # Model for each segment
```

**Features**:
- Single and multiple change point detection
- Online change point detection
- Likelihood ratio tests
- CUSUM and PELT algorithms

**Phase 4: Model Selection & Diagnostics (v0.8.0)**

```python
from stochlab.inference import ModelSelection, Diagnostics

# Compare models
selector = ModelSelection(observed_paths)
results = selector.compare(models, criteria=['aic', 'bic'])
best_model = results.best_by('bic')

# Validate model
diagnostics = Diagnostics(model, observed_paths)
gof = diagnostics.goodness_of_fit()
residuals = diagnostics.residuals()
```

**Features**:
- AIC, BIC, DIC criteria
- K-fold cross-validation
- Goodness-of-fit tests
- Residual analysis

**Phase 5: Bayesian Inference (v0.9.0)**

```python
from stochlab.inference import BayesianEstimator

# MCMC for posterior inference
bayes = BayesianEstimator(
    model_class=MarkovChain,
    prior=DirichletPrior(alpha=[1, 1, 1])
)

posterior = bayes.fit(observed_paths, n_samples=10000)
P_credible_interval = posterior.quantiles([0.025, 0.975])
```

**Features**:
- Metropolis-Hastings sampler
- Gibbs sampler
- Prior specifications
- Posterior predictive distributions

#### Additional Inference Capabilities

- **Anomaly Detection**: Likelihood-based outlier detection
- **Process Identification**: Test for Markovianity, order selection
- **Causal Inference**: Treatment effects, counterfactuals (research-level)

#### Why This Matters

**1. Complete Toolkit**
```python
# Forward: Simulate from model
model = MarkovChain(states, P)
synthetic_data = model.simulate_paths(100, 50)

# Inverse: Learn model from data  
estimated_model = MarkovChain.fit(real_data)

# Validate
diagnostics = Diagnostics(estimated_model, real_data)

# Predict
future = estimated_model.predict(steps_ahead=20)
```

**2. Natural Synergy with Monte Carlo**

Your MC engine enables:
- Particle filtering (importance sampling)
- Bootstrap resampling
- Posterior sampling
- Prediction intervals
- Uncertainty quantification

**3. Competitive Differentiation**

Most libraries are either simulation-only OR inference-only. stochlab would be both.

**4. Educational Value**

Creates complete learning path:
1. Theory → 2. Simulation → 3. Analysis → 4. **Inference** → 5. Prediction

#### Technical Considerations

**Pros**:
- Leverages existing MC infrastructure
- High user demand
- Makes stochlab comprehensive
- Strong pedagogical value

**Cons**:
- Increases scope significantly
- More dependencies (scipy, possibly PyMC)
- More maintenance burden
- Testing complexity

#### References

See [The Inference Landscape](docs/technical/inference_landscape.md) for detailed exploration of:
- State estimation methods
- Parameter estimation techniques
- Model selection approaches
- Change point detection algorithms
- Causal inference methods

---

### 🎨 Interactive Jupyter Widgets

**Status**: Idea phase  
**Target Version**: TBD

```python
from stochlab.interactive import MarkovChainExplorer, ProcessVisualizer

# Interactive model builder
explorer = MarkovChainExplorer()
explorer.show()  # Widget to build/modify/explore chains

# Real-time simulation visualization
viz = ProcessVisualizer(markov_chain)
viz.animate(n_paths=100, T=50)
```

**Benefits**:
- Enhanced user experience
- Educational value
- Demos and presentations

**Dependencies**: ipywidgets, plotly

---

### 🧮 Bayesian Parameter Estimation

**Status**: Idea phase  
**Target Version**: TBD

```python
# Bayesian inference for Markov chain parameters
from stochlab.inference import BayesianMarkovChain

observed_path = [0, 1, 1, 0, 2, 1, ...]

# MCMC for posterior
mc_bayes = BayesianMarkovChain.from_data(
    observed_path,
    prior='dirichlet',
    n_samples=10000
)

# Posterior predictive
future_paths = mc_bayes.predict(steps=50, n_paths=1000)
```

**Benefits**:
- Uncertainty quantification
- Prior knowledge integration
- Robust to limited data

---

### 📊 Performance Benchmarks

**Status**: Idea phase  
**Target Version**: 0.2.0+

- Create `benchmarks/` directory
- Compare against SimPy, PyMC, hmmlearn
- Profile hot paths
- Consider Numba/Cython for critical code
- Publish benchmark results in docs

---

### 🌐 Advanced Backends

**Status**: Research phase  
**Target Version**: TBD

- **JAX Backend**: Automatic differentiation, GPU acceleration
- **Numba JIT**: Compile hot paths for speed
- **Cython**: C-level performance for critical algorithms
- **WebAssembly**: Run in browser for demos

---

## Long-Term Vision

### Core Philosophy

stochlab aims to be:
1. **Comprehensive**: Cover discrete-time and continuous-time stochastic processes
2. **Performant**: Best-in-class Monte Carlo simulation
3. **User-Friendly**: Intuitive API, excellent documentation
4. **Research-Ready**: Advanced techniques (variance reduction, HMMs, Bayesian inference)
5. **Production-Ready**: Robust, tested, well-maintained

### Target Audiences

1. **Students**: Learn stochastic processes
2. **Researchers**: Prototype and validate models
3. **Practitioners**: Apply to real-world problems (finance, operations, biology)
4. **Data Scientists**: Integrate with ML/stats pipelines

### Success Metrics

- Downloads: 10k+/month
- GitHub stars: 1k+
- Active contributors: 10+
- Academic citations
- Industry adoption

---

## How to Contribute Ideas

Have a feature idea? We'd love to hear it!

1. **Open a GitHub Discussion** in the "Ideas" category
2. **Create an Issue** with the "enhancement" label
3. **Submit a PR** updating this roadmap
4. **Email the maintainers** directly

---

## Changelog

- **2024-11-20**: Initial roadmap created (v0.1.1)
  - Added HMMs as major future feature
  - Outlined versions 0.2.0 - 0.5.0
  - Documented long-term vision

---

**Last Updated**: 2024-11-20  
**Version**: 0.1.1

