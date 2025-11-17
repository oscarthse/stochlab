# Architecture Overview

## Design Philosophy

stochlab follows a **layered architecture** that separates concerns and enables extensibility:

```mermaid
graph TD
    A[User Code] --> B[Models Layer]
    A --> C[Core Layer]
    B --> C
    D[Future: MC Engine] --> B
    D --> C
    E[Future: Analytics] --> C
    F[Future: Reporting] --> C
    F --> E
    
    subgraph "Core Layer (stochlab.core)"
        C1[StateSpace]
        C2[StochasticProcess]
        C3[Path]
        C4[SimulationResult]
    end
    
    subgraph "Models Layer (stochlab.models)"
        M1[MarkovChain]
        M2[RandomWalk - Future]
        M3[MM1Queue - Future]
    end
    
    C --> C1
    C --> C2
    C --> C3
    C --> C4
    B --> M1
    B --> M2
    B --> M3
```

## Module Interactions

### Core Components Flow

```mermaid
sequenceDiagram
    participant User
    participant StateSpace
    participant MarkovChain
    participant Path
    participant SimulationResult
    
    User->>StateSpace: Create finite state set
    User->>MarkovChain: Initialize with StateSpace + P matrix
    User->>MarkovChain: sample_path(T=100)
    MarkovChain->>StateSpace: index(state) / state(idx)
    MarkovChain->>Path: Create trajectory
    Path-->>MarkovChain: Return immutable path
    MarkovChain-->>User: Return Path
    
    User->>MarkovChain: simulate_paths(n_paths=1000, T=100)
    loop n_paths times
        MarkovChain->>MarkovChain: sample_path()
    end
    MarkovChain->>SimulationResult: Collect all paths
    SimulationResult-->>User: Return collection with analysis
```

## Layer Responsibilities

### Core Layer (`stochlab.core`)
**Purpose**: Fundamental abstractions used by all models

- **StateSpace**: Finite state set with bijective label ↔ index mapping
- **StochasticProcess**: Abstract interface all models must implement
- **Path**: Single trajectory with validation and immutability
- **SimulationResult**: Collection of paths with analysis methods

**Dependencies**: Only numpy, pandas
**Rules**: No model-specific logic, no I/O, no plotting

### Models Layer (`stochlab.models`)
**Purpose**: Concrete stochastic process implementations

- **MarkovChain**: Time-homogeneous finite Markov chains
- *Future*: RandomWalk, MM1Queue, GaltonWatsonProcess

**Dependencies**: Core layer + numpy for mathematical operations
**Rules**: Must implement StochasticProcess interface

### Future Layers

#### Monte Carlo Engine (`stochlab.mc`)
**Purpose**: Advanced simulation capabilities
- Variance reduction techniques
- Parallel processing
- Seeding strategies

#### Analytics (`stochlab.analytics`)
**Purpose**: Analytical solutions and numerical methods
- Stationary distributions
- Hitting times
- Queueing metrics

#### Reporting (`stochlab.reporting`)
**Purpose**: Visualization and data export
- Interactive plots
- DataFrame utilities
- Export formats

## Data Flow Architecture

```mermaid
graph LR
    subgraph "Input"
        I1[State Labels]
        I2[Parameters]
        I3[Simulation Config]
    end
    
    subgraph "Core Processing"
        P1[StateSpace Creation]
        P2[Model Initialization]
        P3[Path Generation]
        P4[Result Collection]
    end
    
    subgraph "Output"
        O1[Individual Paths]
        O2[Simulation Results]
        O3[DataFrames]
        O4[Analysis]
    end
    
    I1 --> P1
    I2 --> P2
    I3 --> P3
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P3 --> O1
    P4 --> O2
    O2 --> O3
    O2 --> O4
```

## Key Design Patterns

### Immutability Strategy
- **StateSpace**: Completely immutable (`frozen=True`)
- **Path**: Mutable container with immutable arrays
- **SimulationResult**: Mutable collection of immutable paths

### Validation Boundaries
- Input validation at object creation (`__post_init__`)
- Type checking with comprehensive hints
- Mathematical validation (stochastic matrices, probability distributions)

### Interface Contracts
- **State = Hashable**: Any hashable type can be a state
- **StochasticProcess**: Abstract base requiring `state_space` and `sample_path()`
- **Path indexing**: `path[i]` returns state at time i

## Extension Points

### Adding New Models
1. Inherit from `StochasticProcess`
2. Implement required abstract methods
3. Add comprehensive validation
4. Include thorough tests

### Adding Analytics
1. Create pure functions in `analytics/` module
2. Accept core objects as inputs
3. Return mathematical results
4. No side effects or global state

### Adding Visualization
1. Create functions in `reporting/` module
2. Accept `SimulationResult` or DataFrames
3. Return plot objects or save files
4. Optional dependencies for plotting libraries