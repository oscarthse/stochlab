# The Inference Landscape for Stochastic Processes

**Date**: 2024-11-20  
**Status**: Exploration & Design Document

---

## Overview

This document explores the broader "inference ecosystem" for stochastic processes. While simulation (forward problem) is well-established in stochlab, inference (inverse problem) opens up a rich landscape of techniques and applications.

**Key Question**: Given observed data, what can we infer about:
1. The underlying process?
2. Hidden states?
3. Parameters?
4. Future behavior?
5. Anomalies?

---

## 🗺️ The Inference Landscape

```
STOCHASTIC PROCESS INFERENCE
│
├── 1. STATE ESTIMATION
│   ├── Filtering (online)
│   ├── Smoothing (offline)
│   └── Prediction (forecasting)
│
├── 2. PARAMETER ESTIMATION
│   ├── Point estimation (MLE, MoM)
│   ├── Bayesian inference (posterior)
│   └── Bootstrap methods
│
├── 3. MODEL SELECTION
│   ├── Information criteria
│   ├── Cross-validation
│   └── Hypothesis testing
│
├── 4. CHANGE POINT DETECTION
│   ├── Single change point
│   ├── Multiple change points
│   └── Online detection
│
├── 5. ANOMALY DETECTION
│   ├── Outlier detection
│   ├── Novelty detection
│   └── Rare event detection
│
├── 6. PROCESS IDENTIFICATION
│   ├── Testing Markovianity
│   ├── Order selection
│   └── Model diagnostics
│
└── 7. CAUSAL INFERENCE
    ├── Treatment effects
    ├── Counterfactuals
    └── Interventions
```

---

## 1. State Estimation

**Problem**: Estimate hidden/latent states from observations.

### 1.1 Filtering (Online Estimation)

Estimate current state given past observations: \( P(X_t | Y_{1:t}) \)

#### Particle Filtering (Sequential Monte Carlo)

**Status**: High Priority 🔥  
**Complexity**: Medium  
**Synergy with stochlab**: Perfect fit with MC engine

```python
from stochlab.inference import ParticleFilter

# Define process
process = MarkovChain(...)
observation_model = lambda x_t: np.random.normal(x_t, sigma)

# Create particle filter
pf = ParticleFilter(
    process=process,
    observation_model=observation_model,
    n_particles=10000
)

# Online filtering
for y_t in observations:
    pf.update(y_t)
    state_estimate = pf.get_estimate()
    effective_sample_size = pf.get_ess()
    
    if effective_sample_size < threshold:
        pf.resample()  # Importance resampling

# Get filtered trajectory
filtered_states = pf.get_trajectory()
```

**Use Cases**:
- Robotics: Localization
- Finance: Tracking hidden market regimes
- Epidemiology: Estimating true infection rates

**Algorithms**:
- Bootstrap filter
- Auxiliary particle filter
- Rao-Blackwellized particle filter

**Benefits**:
- Handles non-linear, non-Gaussian models
- Leverages existing Monte Carlo infrastructure
- Natural extension of current capabilities

#### Kalman Filtering

**Status**: Lower Priority (continuous state)  
**Complexity**: Low  
**Note**: Only works for linear-Gaussian systems

```python
from stochlab.inference import KalmanFilter

# For continuous-state linear systems
kf = KalmanFilter(
    transition_matrix=A,
    observation_matrix=H,
    process_noise=Q,
    observation_noise=R
)

for y_t in observations:
    kf.predict()
    kf.update(y_t)
```

**Extensions**:
- Extended Kalman Filter (EKF) - linearization
- Unscented Kalman Filter (UKF) - sigma points

### 1.2 Smoothing (Offline Estimation)

Estimate past states given all observations: \( P(X_t | Y_{1:T}) \)

```python
from stochlab.inference import ParticleSmoother

# After collecting all observations
ps = ParticleSmoother(process, observation_model)
smoothed_states = ps.smooth(observations)

# Better estimates than filtering (uses future info)
```

**Algorithms**:
- Forward-Backward algorithm (HMMs)
- Particle smoothing
- Rauch-Tung-Striebel smoother (Kalman)

### 1.3 Prediction/Forecasting

Predict future states: \( P(X_{t+k} | Y_{1:t}) \)

```python
# Multi-step ahead prediction with uncertainty
predictions = pf.predict(steps_ahead=10, n_paths=1000)

# Includes prediction intervals
lower, median, upper = predictions.quantiles([0.025, 0.5, 0.975])
```

---

## 2. Parameter Estimation

**Problem**: Estimate model parameters from observed data.

### 2.1 Maximum Likelihood Estimation (MLE)

**Status**: High Priority 🔥  
**Complexity**: Medium

```python
from stochlab.inference import MaximumLikelihoodEstimator

# Observed path(s)
observed_paths = [path1, path2, path3, ...]

# Estimate Markov chain transition matrix
mle = MaximumLikelihoodEstimator(model_class=MarkovChain)
estimated_model = mle.fit(observed_paths)

# Access results
P_hat = estimated_model.transition_matrix
standard_errors = mle.standard_errors
confidence_intervals = mle.confidence_intervals(level=0.95)
```

**For Different Models**:

```python
# Markov Chain: Count-based MLE
transition_counts = count_transitions(observed_paths)
P_mle = normalize_rows(transition_counts)

# Random Walk: Estimate drift and diffusion
rw_mle = MaximumLikelihoodEstimator(model_class=RandomWalk)
rw_hat = rw_mle.fit(observed_paths)
drift_estimate = rw_hat.drift
variance_estimate = rw_hat.variance

# M/M/1 Queue: Estimate arrival and service rates
queue_mle = MaximumLikelihoodEstimator(model_class=MM1Queue)
queue_hat = queue_mle.fit(observed_queue_lengths)
```

**Advanced**: EM algorithm for models with latent variables (HMMs, etc.)

### 2.2 Bayesian Inference

**Status**: Medium Priority  
**Complexity**: High

```python
from stochlab.inference import BayesianEstimator

# Specify prior
prior = DirichletPrior(alpha=[1, 1, 1])  # Uniform prior on simplex

# MCMC sampling
bayes = BayesianEstimator(
    model_class=MarkovChain,
    prior=prior,
    sampler='metropolis-hastings'  # or 'gibbs', 'hmc'
)

posterior = bayes.fit(
    observed_paths,
    n_samples=10000,
    burn_in=1000,
    thin=10
)

# Posterior samples
P_samples = posterior.samples['transition_matrix']
P_mean = posterior.mean()
P_credible_interval = posterior.quantiles([0.025, 0.975])

# Posterior predictive
future_paths = posterior.predict(T=50, n_paths=1000)
```

**Benefits**:
- Uncertainty quantification
- Incorporates prior knowledge
- Robust to limited data
- Natural for model comparison (Bayes factors)

**Challenges**:
- Computational cost (MCMC can be slow)
- Convergence diagnostics
- Prior specification

### 2.3 Method of Moments

**Status**: Low Priority  
**Complexity**: Low

```python
from stochlab.inference import MethodOfMoments

# Match empirical and theoretical moments
mom = MethodOfMoments(model_class=RandomWalk)
estimated_model = mom.fit(
    observed_paths,
    moments=['mean', 'variance', 'autocorrelation']
)
```

### 2.4 Bootstrap Methods

**Status**: Medium Priority  
**Complexity**: Low-Medium  
**Synergy**: Perfect with MC engine

```python
from stochlab.inference import BootstrapEstimator

# Bootstrap confidence intervals
bootstrap = BootstrapEstimator(
    estimator=MaximumLikelihoodEstimator(MarkovChain),
    n_bootstrap=1000,
    parallel=True  # Use MC engine!
)

results = bootstrap.fit(observed_paths)

# Bootstrap distribution of estimates
P_bootstrap_dist = results.bootstrap_distribution
P_confidence_interval = results.confidence_interval(level=0.95)
bias = results.bias
```

---

## 3. Model Selection

**Problem**: Choose the best model from candidates.

### 3.1 Information Criteria

**Status**: Medium Priority  
**Complexity**: Low

```python
from stochlab.inference import ModelSelection

models = [
    MarkovChain(n_states=2),
    MarkovChain(n_states=3),
    MarkovChain(n_states=4),
]

selector = ModelSelection(observed_paths)

# Compute information criteria
results = selector.compare(
    models,
    criteria=['aic', 'bic', 'dic']
)

print(results)
# Model | AIC    | BIC    | DIC
# ------|--------|--------|--------
# MC(2) | 145.3  | 152.1  | 144.8
# MC(3) | 138.2* | 149.7  | 137.5*  <- Best by AIC, DIC
# MC(4) | 142.1  | 158.3  | 141.9

best_model = results.best_by('bic')
```

**Criteria**:
- **AIC** (Akaike): \( -2\log L + 2k \)
- **BIC** (Bayesian): \( -2\log L + k\log n \)
- **DIC** (Deviance): For Bayesian models

### 3.2 Cross-Validation

**Status**: Medium Priority  
**Complexity**: Medium

```python
from stochlab.inference import CrossValidator

cv = CrossValidator(
    model_class=MarkovChain,
    n_folds=5,
    scoring='log_likelihood'
)

# K-fold cross-validation
scores = cv.validate(observed_paths, param_grid={
    'n_states': [2, 3, 4, 5],
    'regularization': [0, 0.01, 0.1]
})

best_params = scores.best_params
```

### 3.3 Hypothesis Testing

```python
from stochlab.inference import HypothesisTest

# Test if two Markov chains have same transition matrix
test = HypothesisTest.likelihood_ratio(
    model1=mc1,
    model2=mc2,
    data=observed_paths,
    alternative='two-sided'
)

print(f"Test statistic: {test.statistic}")
print(f"P-value: {test.pvalue}")
print(f"Reject H0: {test.reject}")
```

---

## 4. Change Point Detection

**Problem**: Detect when process parameters/dynamics change.

**Status**: High Priority 🔥  
**Complexity**: Medium  
**Applications**: Very broad

### 4.1 Single Change Point

```python
from stochlab.inference import ChangePointDetector

detector = ChangePointDetector(model_class=MarkovChain)

# Detect single change point
result = detector.detect_single(
    observed_path,
    method='likelihood-ratio'  # or 'cusum', 'bayesian'
)

change_point = result.change_point  # Time index
confidence = result.confidence
before_model = result.model_before
after_model = result.model_after

# Visualize
detector.plot(observed_path, change_point)
```

### 4.2 Multiple Change Points

```python
# Detect multiple change points
result = detector.detect_multiple(
    observed_path,
    max_changepoints=5,
    method='pelt'  # Pruned Exact Linear Time
)

change_points = result.change_points  # [t1, t2, t3, ...]
segments = result.segments  # Models for each segment
```

### 4.3 Online Change Point Detection

```python
# Real-time monitoring
online_detector = OnlineChangePointDetector(
    model_class=MarkovChain,
    baseline_model=mc_baseline,
    threshold=3.0  # Alert threshold
)

for t, x_t in enumerate(stream):
    alert = online_detector.update(x_t)
    
    if alert:
        print(f"Change point detected at t={t}")
        online_detector.reset()
```

**Use Cases**:
- **Finance**: Regime shifts in markets
- **Manufacturing**: Process control, quality shifts
- **Biology**: Evolutionary transitions
- **Climate**: Detecting climate changes
- **Medicine**: Disease progression stages

---

## 5. Anomaly Detection

**Problem**: Identify unusual observations or patterns.

**Status**: Medium Priority  
**Complexity**: Low-Medium

### 5.1 Likelihood-Based Anomaly Detection

```python
from stochlab.inference import AnomalyDetector

# Train on normal data
detector = AnomalyDetector(model_class=MarkovChain)
detector.fit(normal_paths)

# Score new observations
for x_t in new_observations:
    anomaly_score = detector.score(x_t)
    is_anomaly = anomaly_score > threshold
    
    if is_anomaly:
        print(f"Anomaly detected: {x_t} (score: {anomaly_score})")
```

### 5.2 Rare Event Detection

```python
# Detect rare events (e.g., system failures)
rare_event_detector = RareEventDetector(
    process=queue_model,
    event=lambda path: path.states[-1] > critical_threshold,
    rarity_threshold=1e-6
)

# Use importance sampling for efficiency
is_rare = rare_event_detector.estimate_probability(
    n_paths=100000,
    method='importance-sampling'
)
```

**Use Cases**:
- **Cybersecurity**: Intrusion detection
- **Finance**: Fraud detection
- **Healthcare**: Disease outbreak detection
- **IoT**: Sensor fault detection

---

## 6. Process Identification

**Problem**: Determine the type of stochastic process from data.

**Status**: Lower Priority  
**Complexity**: Medium

### 6.1 Testing for Markovianity

```python
from stochlab.inference import MarkovianityTest

# Test if observed process is Markovian
test = MarkovianityTest(observed_path)
result = test.test(method='likelihood-ratio')

print(f"Is Markovian: {result.is_markovian}")
print(f"P-value: {result.pvalue}")

# If not Markovian, suggest order
suggested_order = test.suggest_order(max_order=5)
```

### 6.2 Model Diagnostics

```python
from stochlab.inference import Diagnostics

# Fit model
model = MarkovChain.fit(observed_paths)

# Run diagnostics
diag = Diagnostics(model, observed_paths)

# Goodness-of-fit tests
gof = diag.goodness_of_fit()
print(f"Chi-square statistic: {gof.statistic}")
print(f"P-value: {gof.pvalue}")

# Residual analysis
residuals = diag.residuals()
diag.plot_residuals()  # QQ-plot, ACF plot, etc.

# Prediction accuracy
pred_accuracy = diag.prediction_accuracy(test_data)
```

---

## 7. Causal Inference

**Problem**: Infer causal relationships and treatment effects.

**Status**: Research Level  
**Complexity**: High

### 7.1 Treatment Effects in Stochastic Processes

```python
from stochlab.inference import CausalInference

# Observe process under different treatments
control_paths = observed_paths_control
treatment_paths = observed_paths_treatment

# Estimate average treatment effect
causal = CausalInference(model_class=MarkovChain)
ate = causal.estimate_treatment_effect(
    control_paths,
    treatment_paths,
    method='matching'  # or 'propensity-score', 'iv'
)

print(f"Average treatment effect: {ate.estimate}")
print(f"95% CI: [{ate.ci_lower}, {ate.ci_upper}]")
```

### 7.2 Counterfactual Analysis

```python
# What would have happened under different intervention?
counterfactual = causal.counterfactual(
    observed_path,
    intervention={'state': 0, 'time': 10}  # Force to state 0 at t=10
)

# Compare factual vs counterfactual
causal_effect = counterfactual - observed_path
```

---

## 🎯 Priority Matrix

Based on fit with stochlab's current capabilities and user value:

| Feature | Priority | Effort | Value | Synergy |
|---------|----------|--------|-------|---------|
| **Particle Filtering** | 🔥🔥🔥 | Medium | High | Perfect |
| **MLE for Parameters** | 🔥🔥🔥 | Medium | High | Good |
| **Change Point Detection** | 🔥🔥 | Medium | High | Good |
| **Bootstrap Methods** | 🔥🔥 | Low | Medium | Perfect |
| **Bayesian Inference** | 🔥 | High | High | Medium |
| **Model Selection** | 🔥 | Low | Medium | Good |
| **Anomaly Detection** | 🔥 | Low-Med | Medium | Good |
| **Process Identification** | 🟡 | Medium | Low-Med | Medium |
| **Causal Inference** | 🟡 | High | Medium | Low |

---

## 🚀 Recommended Roadmap

### Phase 1: Foundation (v0.6.0)
- HMM inference (as planned)
- Particle filtering
- Basic MLE for existing models

### Phase 2: Practical Inference (v0.7.0)
- HMM learning (Baum-Welch)
- Change point detection
- Bootstrap methods
- Model selection (AIC/BIC)

### Phase 3: Advanced Methods (v0.8.0)
- Bayesian inference (MCMC)
- Anomaly detection
- Cross-validation
- Process diagnostics

### Phase 4: Research Features (v0.9.0+)
- Causal inference
- Online learning
- Active learning
- Adaptive sampling

---

## 💡 Key Insights

### 1. Natural Synergy with Monte Carlo

Your existing MC engine is **perfect** for:
- Particle filtering (IS-based)
- Bootstrap methods (resampling)
- Bayesian posterior sampling
- Prediction intervals
- Uncertainty quantification

This is a **huge competitive advantage**.

### 2. Simulation + Inference = Complete Package

Right now: **Forward problem** (simulate from model)  
With inference: **Inverse problem** (learn model from data)  
Together: **Bi-directional workflow**

```python
# Forward: Simulation
model = MarkovChain(states, P)
data = model.simulate_paths(n_paths=100, T=50)

# Inverse: Inference
estimated_model = MarkovChain.fit(observed_data)
future_predictions = estimated_model.predict(steps_ahead=10)

# Validation
diagnostics = Diagnostics(estimated_model, observed_data)
goodness_of_fit = diagnostics.test()
```

### 3. Differentiates from Competition

Most libraries are either:
- **Simulation-only** (SimPy)
- **Inference-only** (statsmodels, PyMC)
- **Narrow focus** (hmmlearn)

**stochlab + inference = comprehensive toolkit**

### 4. Educational Value

Teaching stochastic processes typically covers:
1. Theory (math)
2. Simulation (forward)
3. **Inference (inverse)** ← Missing in most tools

stochlab could become **the** pedagogical tool.

---

## 🎓 Suggested Learning Path for Users

```python
# 1. Learn to simulate
mc = MarkovChain.from_transition_matrix(...)
paths = mc.simulate_paths(100, 50)

# 2. Learn to analyze
stats = mc.stationary_distribution()
hitting_times = mc.hitting_time_distribution(...)

# 3. Learn to infer
estimated_mc = MarkovChain.fit(observed_data)

# 4. Learn to validate
diagnostics = Diagnostics(estimated_mc, observed_data)

# 5. Learn to predict
predictions = estimated_mc.predict(steps_ahead=20)

# 6. Learn advanced techniques
pf = ParticleFilter(estimated_mc, observation_model)
changepoints = ChangePointDetector().detect(observed_data)
```

This creates a **complete learning journey**.

---

## 📚 Key References

### Particle Filtering
- Doucet, A., & Johansen, A. M. (2009). "A tutorial on particle filtering and smoothing"
- Arulampalam, M. S., et al. (2002). "A tutorial on particle filters"

### Parameter Estimation
- Cappé, O., Moulines, E., & Rydén, T. (2005). "Inference in Hidden Markov Models"
- Hamilton, J. D. (1994). "Time Series Analysis"

### Change Point Detection
- Killick, R., et al. (2012). "Optimal detection of changepoints"
- Tartakovsky, A. G., et al. (2014). "Sequential Analysis: Hypothesis Testing and Changepoint Detection"

### Causal Inference
- Pearl, J. (2009). "Causality: Models, Reasoning, and Inference"
- Hernán, M. A., & Robins, J. M. (2020). "Causal Inference: What If"

---

## 🤔 Questions for Strategic Decision

1. **Scope**: Should stochlab remain simulation-focused or expand to inference?
2. **Depth**: Full inference toolkit or just key methods?
3. **Dependencies**: OK to add scipy, PyMC, sklearn as dependencies?
4. **Audience**: Academic tool, professional tool, or both?
5. **Maintenance**: Can we sustain a larger feature set?

---

**Verdict**: Inference is a **natural and valuable** extension. Recommend:
- Start with HMMs + Particle Filtering (highest synergy)
- Add MLE and Bootstrap (low-hanging fruit)
- Gradually expand to change point and model selection
- Keep Bayesian and causal inference for later (high complexity)

This maintains focus while significantly expanding capabilities. 🎯

