# stochlab – Developer Guide

This document describes how to work on **stochlab** in a clean, consistent way:

- Git/GitHub workflow
- When and how to write tests
- Documentation expectations
- Code quality tools (`black`, `ruff`, `mypy`, `pytest`)

---

## 1. Git & Branching Workflow

**Main rules:**

`main` must always be:
- ✅ Installable
- ✅ Tests passing
- ✅ No obviously broken APIs

### 1.1 Branches

For any non-trivial change, use a feature branch:

```bash
git checkout -b feature/markov-chain-from-sequences
# or
git checkout -b fix/state-space-bug
```

**Keep branches small and focused:**
- One feature / refactor / bugfix per branch

### 1.2 Commit Style

Aim for small, logical commits:
- Each commit should represent one coherent change

**Suggested format:**

```
type: short description

[optional longer explanation]
```

**Examples:**

```
feat: add MarkovChain.from_transition_matrix
test: add unit tests for SimulationResult.state_distribution
refactor: simplify Path __post_init__ validation
fix: handle empty sequences in from_sequences
```

**Avoid** huge "mixed" commits (docs + refactor + new features all in one).

### 1.3 Pull Requests (even if it's just you)

When using GitHub:

1. Open a PR from `feature/...` → `main`
2. In the PR description include:
   - What you changed
   - Why you changed it
   - Any breaking changes / TODOs
3. Use the PR checklist from section 6 before merging

---

## 2. Testing Protocol (pytest)

### 2.1 Where Tests Live

All tests go in `tests/`:

```
tests/test_state_space.py
tests/test_path.py
tests/test_results.py
tests/test_markov_chain.py
tests/test_core_integration.py
```

**Naming convention:**
- **File:** `test_<module_or_feature>.py`
- **Functions:** `test_<something_descriptive>()`

### 2.2 When to Write/Update Tests

Always add or update tests when:

- You add a new core class or method (MarkovChain, from_sequences, new process types)
- You change behaviour (e.g. new validation rules)
- You fix a bug → add a test that would have caught the bug

**Minimal rule:**

> "No new feature without at least one test that hits it."

### 2.3 How Often to Run Tests

**During development:**

After implementing a new feature or refactor:

```bash
make test
# or
uv run pytest
```

**Before pushing or merging a PR:**

Run full test suite:

```bash
make all-checks
# which runs: lint + typecheck + tests
```

If something feels risky (big refactor, touching core), run:

```bash
make test-verbose
```

to see more detailed output.

---

## 3. Code Quality: Formatting, Linting, Types

**Tools** (installed as dev deps):

- `black` – formatting
- `ruff` – linting
- `mypy` – static type checking
- `pytest` – tests

Use the **Makefile shortcuts:**

### 3.1 Formatting

```bash
make format
# runs: uv run black src/ tests/
```

**Run this:**
- Before commits when you've touched multiple files
- Anytime pytest complains about style

### 3.2 Linting

```bash
make lint
# runs: uv run ruff check src/ tests/
```

**Run this:**
- Before PRs
- When adding new modules: ensures imports, unused vars, etc. are clean

### 3.3 Type-checking

```bash
make typecheck
# runs: uv run mypy src/
```

**Run this:**
- After adding new classes / signatures
- When refactoring core types (e.g. changing `State`, `Path`, `StochasticProcess`)

### 3.4 Full Check

```bash
make all-checks
# runs: lint + typecheck + test
```

**Use this as your pre-push / pre-merge command.**

---

## 4. Documentation Expectations

We want **lightweight but useful** docs:

### 4.1 Docstrings

Every public-facing class and method should have a docstring with:

- Short summary line
- `Parameters` section
- `Returns` (and `Raises` if relevant)

**Example:**

```python
def sample_path(
    self,
    T: int,
    x0: State | None = None,
    **kwargs: Any,
) -> Path:
    """
    Generate a single sample path (X_0, ..., X_T) from the Markov chain.

    Parameters
    ----------
    T : int
        Horizon (path length will be T+1).
    x0 : State, optional
        Initial state label. If None, X_0 is sampled from `initial_dist`.
    rng : np.random.Generator, optional
        Random number generator passed via kwargs.

    Returns
    -------
    Path
        Simulated trajectory as a Path object.
    """
```

**Minimum requirement:**

All core types in `stochlab.core` and public models in `stochlab.models` have docstrings like this.

### 4.2 README and Examples

Keep `README.md` up to date with:

- Brief description of the library's purpose
- Simple usage example, e.g.:

```python
from stochlab import MarkovChain

states = ["A", "B"]
P = [[0.9, 0.1],
     [0.2, 0.8]]

mc = MarkovChain.from_transition_matrix(states, P)
path = mc.sample_path(T=10, x0="A")
print(path.states)
```

Whenever you add a big new concept (e.g. new model type, Monte Carlo utilities), consider adding:

- A new example under README
- Or eventually an `examples/` directory (later)

---

## 5. Adding a New Feature: Checklist

When you add a new model / core concept (e.g. `from_sequences`, new process type, new Monte Carlo utility), follow this mini checklist:

### 1. Design First

Write a short comment or TODO at the top of the new file/module:
- What the class/function does
- Inputs/outputs, at a high level

### 2. Implement in `src/stochlab/...`

- Put core abstractions in `stochlab/core`
- Put concrete models in `stochlab/models`
- Keep functions small and focused

### 3. Add Tests in `tests/`

- At least one unit test per major method / behaviour
- If you fix a bug → add a regression test

### 4. Run Tools

```bash
make format
make lint
make typecheck
make test
# or just
make all-checks
```

### 5. Update Docs

- Add/adjust docstrings
- If it's user-facing, consider adding it to README or a short note in the dev guide

### 6. Commit & Push

- Use a clear commit message
- Open a PR if using GitHub; include a short description

---

## 6. PR / Pre-merge Checklist

Before merging to `main`, mentally tick these off:

- [ ] All tests pass: `make test`
- [ ] Lint & types clean: `make lint`, `make typecheck`
- [ ] Public classes/functions have docstrings
- [ ] No debug prints / dead code
- [ ] README / comments not obviously outdated
- [ ] Changes are focused (not a huge "do everything" PR)

---

**Happy coding! 🚀**
