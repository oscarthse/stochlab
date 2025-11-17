# Financial Modeling Examples

This section demonstrates how to use stochlab for financial applications.

## Credit Rating Transitions

Model how credit ratings evolve over time using historical transition data.

```python
import numpy as np
import pandas as pd
from stochlab.models import MarkovChain

# Standard credit rating states
ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]

# Historical annual transition matrix (example data)
P = np.array([
    [0.92, 0.06, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00],  # AAA
    [0.02, 0.90, 0.06, 0.01, 0.01, 0.00, 0.00, 0.00],  # AA
    [0.00, 0.03, 0.88, 0.07, 0.01, 0.01, 0.00, 0.00],  # A
    [0.00, 0.00, 0.05, 0.85, 0.07, 0.02, 0.01, 0.00],  # BBB
    [0.00, 0.00, 0.00, 0.06, 0.80, 0.10, 0.03, 0.01],  # BB
    [0.00, 0.00, 0.00, 0.00, 0.08, 0.75, 0.12, 0.05],  # B
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.65, 0.20],  # CCC
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],  # D (absorbing)
])

# Create the credit rating model
credit_model = MarkovChain.from_transition_matrix(ratings, P)

# Simulate rating paths for a portfolio
n_bonds = 1000
horizon_years = 5

result = credit_model.simulate_paths(
    n_paths=n_bonds,
    T=horizon_years,
    x0="BBB"  # Start all bonds at BBB rating
)

# Analyze default probability
df = result.to_dataframe()
final_ratings = df[df['t'] == horizon_years]
default_rate = (final_ratings['state'] == 'D').mean()

print(f"5-year default rate for BBB bonds: {default_rate:.2%}")

# Rating distribution over time
for year in range(horizon_years + 1):
    dist = result.state_distribution(t=year)
    print(f"Year {year}: {dist}")
```

## Market Regime Modeling

Model bull/bear market transitions with volatility regimes.

```python
import numpy as np
from stochlab.models import MarkovChain

# Define market regimes
regimes = [
    ("Bull", "Low Vol"),
    ("Bull", "High Vol"),
    ("Bear", "Low Vol"), 
    ("Bear", "High Vol")
]

# Transition matrix based on empirical data
P = np.array([
    [0.85, 0.10, 0.04, 0.01],  # Bull/Low -> ...
    [0.60, 0.25, 0.10, 0.05],  # Bull/High -> ...
    [0.15, 0.05, 0.70, 0.10],  # Bear/Low -> ...
    [0.10, 0.15, 0.25, 0.50],  # Bear/High -> ...
])

regime_model = MarkovChain.from_transition_matrix(regimes, P)

# Simulate market regimes for backtesting
trading_days = 252  # 1 year
n_scenarios = 1000

result = regime_model.simulate_paths(
    n_paths=n_scenarios,
    T=trading_days,
    x0=("Bull", "Low Vol")
)

# Analyze regime persistence
df = result.to_dataframe()

# Calculate average time in each regime
regime_durations = {}
for regime in regimes:
    regime_paths = df[df['state'] == regime]
    if len(regime_paths) > 0:
        # Calculate run lengths (simplified)
        avg_duration = len(regime_paths) / n_scenarios
        regime_durations[regime] = avg_duration

print("Average days per year in each regime:")
for regime, duration in regime_durations.items():
    print(f"  {regime}: {duration:.1f} days")
```

## Portfolio Risk Analysis

Use Markov chains to model correlated asset behavior.

```python
import numpy as np
from stochlab.models import MarkovChain

# Asset states: (Stock Performance, Bond Performance)
asset_states = [
    ("Up", "Up"),     # Both assets perform well
    ("Up", "Down"),   # Stock up, bonds down (rising rates)
    ("Down", "Up"),   # Stock down, bonds up (flight to quality)
    ("Down", "Down")  # Both assets perform poorly
]

# Transition probabilities reflecting market correlations
P = np.array([
    [0.60, 0.25, 0.10, 0.05],  # Up/Up -> ...
    [0.30, 0.40, 0.05, 0.25],  # Up/Down -> ...
    [0.20, 0.10, 0.50, 0.20],  # Down/Up -> ...
    [0.15, 0.20, 0.35, 0.30],  # Down/Down -> ...
])

portfolio_model = MarkovChain.from_transition_matrix(asset_states, P)

# Monte Carlo simulation for risk assessment
n_simulations = 10000
investment_horizon = 60  # 5 years monthly

result = portfolio_model.simulate_paths(
    n_paths=n_simulations,
    T=investment_horizon
)

# Risk metrics
df = result.to_dataframe()

# Calculate probability of adverse scenarios
adverse_states = [("Down", "Up"), ("Down", "Down")]
adverse_periods = df[df['state'].isin(adverse_states)]
adverse_probability = len(adverse_periods) / len(df)

print(f"Probability of adverse market conditions: {adverse_probability:.2%}")

# Stress testing: probability of extended downturns
final_states = df[df['t'] == investment_horizon]
severe_stress = (final_states['state'] == ("Down", "Down")).mean()

print(f"Probability of ending in severe stress: {severe_stress:.2%}")
```

## Key Financial Applications

### 1. Credit Risk Management
- **Default probability estimation**: Calculate PD curves for different rating classes
- **Portfolio loss modeling**: Simulate correlated defaults across portfolios
- **Regulatory capital**: Estimate economic capital requirements

### 2. Market Risk Modeling
- **Regime-dependent VaR**: Risk measures that adapt to market conditions
- **Stress testing**: Scenario generation for regulatory stress tests
- **Asset allocation**: Dynamic strategies based on regime probabilities

### 3. Operational Risk
- **Loss event modeling**: Frequency and severity of operational losses
- **Business continuity**: Model recovery times and operational states
- **Regulatory compliance**: Monte Carlo for operational risk capital

## Best Practices

### Model Validation
```python
# Always validate your transition matrix
def validate_transition_matrix(P):
    """Validate stochastic matrix properties."""
    # Check dimensions
    assert P.ndim == 2 and P.shape[0] == P.shape[1]
    
    # Check non-negativity
    assert np.all(P >= 0)
    
    # Check row sums
    row_sums = P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-8)
    
    print("Transition matrix validation passed")

validate_transition_matrix(P)
```

### Sensitivity Analysis
```python
# Test sensitivity to transition probabilities
base_default_prob = 0.05
sensitivity_range = np.linspace(0.01, 0.10, 10)

results = []
for prob in sensitivity_range:
    # Modify transition matrix
    P_modified = P.copy()
    P_modified[-2, -1] = prob  # Modify CCC -> D transition
    P_modified[-2, -2] = 1 - prob  # Adjust CCC -> CCC to maintain row sum
    
    model = MarkovChain.from_transition_matrix(ratings, P_modified)
    result = model.simulate_paths(n_paths=1000, T=5, x0="CCC")
    
    # Calculate default rate
    df = result.to_dataframe()
    final_states = df[df['t'] == 5]
    default_rate = (final_states['state'] == 'D').mean()
    
    results.append((prob, default_rate))

print("Sensitivity Analysis:")
for prob, default_rate in results:
    print(f"CCC->D prob: {prob:.3f}, 5-year default rate: {default_rate:.3f}")
```