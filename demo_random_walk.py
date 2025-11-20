"""Quick demo of RandomWalk."""

import numpy as np
from stochlab.models import RandomWalk

# Create a symmetric random walk
rw = RandomWalk(lower_bound=-10, upper_bound=10, p=0.5)
print(f"State space size: {len(rw.state_space)}")

# Single path
path = rw.sample_path(T=100, x0=0)
print(f"\nSingle path: start={path[0]}, end={path[-1]}")

# Monte Carlo simulation
np.random.seed(42)
result = rw.simulate_paths(n_paths=1000, T=100, x0=0)
print(f"\nSimulated {len(result)} paths")

# Analyze final distribution
final_dist = result.state_distribution(t=100)
print(f"\nFinal state distribution (top 5):")
for state, prob in sorted(final_dist.items(), key=lambda x: -x[1])[:5]:
    print(f"  State {state:3d}: {prob:.3f}")

# Biased walk
rw_biased = RandomWalk(lower_bound=0, upper_bound=20, p=0.7)
result_biased = rw_biased.simulate_paths(n_paths=1000, T=50, x0=10)
final_biased = result_biased.state_distribution(t=50)
avg_final = sum(s * p for s, p in final_biased.items())
print(f"\nBiased walk (p=0.7): average final position = {avg_final:.2f}")
