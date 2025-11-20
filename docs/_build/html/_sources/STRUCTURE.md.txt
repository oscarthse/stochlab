# Documentation Structure

This document describes the organization of the stochlab documentation.

## Directory Structure

```
docs/
├── README.md                          # Documentation index and navigation
├── getting_started.md                 # Quick introduction (start here!)
├── quick_reference.md                 # Cheat sheet for common operations
│
├── guides/                            # User guides
│   ├── monte_carlo.md                 # Complete Monte Carlo guide ⭐
│   └── analytics.md                   # Markov chain analytics
│
├── api/                               # API reference
│   └── index.rst                      # Auto-generated from docstrings
│
├── technical/                         # Technical & design documentation
│   ├── monte_carlo_design.md          # MC engine design document
│   ├── parallelization_analysis.md    # Parallel execution deep dive
│   └── implementation_summary.md      # Implementation overview
│
├── contributing/                      # Developer documentation
│   └── development_guide.md           # Contributing guidelines
│
├── index.rst                          # Sphinx main entry point
└── conf.py                            # Sphinx configuration
```

## File Organization Principles

### 1. **Separation by Audience**

- **Root level** (`getting_started.md`, `quick_reference.md`): First-time users
- **guides/**: Regular users learning features
- **api/**: Users needing API reference
- **technical/**: Developers and advanced users
- **contributing/**: Contributors

### 2. **Consistent Naming**

- **Lowercase with underscores**: `monte_carlo.md`, `development_guide.md`
- **Descriptive names**: Clear what each document contains
- **No abbreviations**: `monte_carlo` not `mc`, `analytics` not `ana`

### 3. **Logical Grouping**

- **User-facing docs** stay in root or `guides/`
- **Implementation details** go in `technical/`
- **Process docs** go in `contributing/`

## Quick Navigation

### For New Users

Start here:
1. `getting_started.md` - Basics
2. `quick_reference.md` - Quick lookup
3. `guides/monte_carlo.md` - Deep dive

### For Regular Users

Reference:
- `guides/monte_carlo.md` - Monte Carlo features
- `guides/analytics.md` - Analytics features
- `api/index.rst` - API reference

### For Advanced Users

Deep dives:
- `technical/parallelization_analysis.md` - How parallelization works
- `technical/monte_carlo_design.md` - Design decisions
- `technical/implementation_summary.md` - What was built

### For Contributors

Development:
- `contributing/development_guide.md` - How to contribute
- `technical/` - Understand the architecture

## File Descriptions

### Root Level

**`README.md`**
- Purpose: Documentation index and navigation guide
- Audience: Anyone looking for documentation
- Links to: All other docs with descriptions

**`getting_started.md`**
- Purpose: Quick introduction to stochlab
- Audience: First-time users
- Content: Basic concepts, simple examples, next steps

**`quick_reference.md`**
- Purpose: Cheat sheet for common operations
- Audience: Users who know the basics
- Content: Quick lookup table of operations

**`index.rst`**
- Purpose: Sphinx documentation entry point
- Audience: Sphinx build system
- Content: Table of contents for generated HTML docs

### guides/

**`monte_carlo.md`** (~850 lines)
- Purpose: Complete guide to Monte Carlo simulation
- Audience: Users of Monte Carlo features
- Content:
  - Quick start
  - All features explained
  - Performance tips
  - Troubleshooting
  - API reference

**`analytics.md`**
- Purpose: Guide to Markov chain analytics
- Audience: Users doing analytical computations
- Content:
  - Stationary distributions
  - Hitting times
  - Absorption probabilities

### api/

**`index.rst`**
- Purpose: API reference documentation
- Audience: Users needing detailed API info
- Content: Auto-generated from docstrings

### technical/

**`monte_carlo_design.md`** (~591 lines)
- Purpose: Comprehensive design document for MC engine
- Audience: Developers, contributors, architects
- Content:
  - Requirements
  - Architecture design
  - API design
  - Performance optimizations
  - Implementation plan
  - Testing strategy

**`parallelization_analysis.md`** (~738 lines)
- Purpose: Deep dive into parallelization strategies
- Audience: Performance engineers, advanced users
- Content:
  - Comparison of 5 approaches
  - Detailed implementation
  - Performance analysis
  - Best practices

**`implementation_summary.md`** (~400 lines)
- Purpose: Overview of what was built
- Audience: Stakeholders, maintainers
- Content:
  - Features implemented
  - Performance characteristics
  - Design decisions
  - Success metrics

### contributing/

**`development_guide.md`**
- Purpose: Guidelines for contributing
- Audience: Contributors
- Content:
  - Development setup
  - Code style
  - Testing requirements
  - PR process

## Finding What You Need

### By Use Case

| Need | Start Here |
|------|-----------|
| First-time user | `getting_started.md` |
| Learn Monte Carlo | `guides/monte_carlo.md` |
| Quick lookup | `quick_reference.md` |
| API details | `api/index.rst` |
| Understand internals | `technical/` |
| Contribute code | `contributing/development_guide.md` |

### By Expertise Level

| Level | Read |
|-------|------|
| **Beginner** | `getting_started.md`, `quick_reference.md` |
| **Intermediate** | `guides/`, `api/` |
| **Advanced** | `technical/`, `contributing/` |

### By Topic

| Topic | Documentation |
|-------|--------------|
| **Monte Carlo basics** | `getting_started.md` § Monte Carlo |
| **Monte Carlo advanced** | `guides/monte_carlo.md` |
| **Monte Carlo internals** | `technical/monte_carlo_design.md` |
| **Parallelization** | `technical/parallelization_analysis.md` |
| **Analytics** | `guides/analytics.md` |
| **API** | `api/index.rst` |
| **Contributing** | `contributing/development_guide.md` |

## Building Documentation

### HTML (Sphinx)

```bash
cd docs
make html
open _build/html/index.html
```

### PDF (if configured)

```bash
cd docs
make latexpdf
```

## Maintenance

### When Adding New Features

1. **Update user guide**: Add to appropriate file in `guides/`
2. **Update docstrings**: Ensure API docs stay current
3. **Update getting started**: If it's a major feature
4. **Update quick reference**: If it's commonly used

### When Changing API

1. **Update docstrings**: Keep them in sync
2. **Update user guides**: Reflect API changes
3. **Update examples**: Ensure they still work

### When Adding Technical Content

- Add to `technical/` directory
- Link from `README.md`
- Update this `STRUCTURE.md` if needed

## Why This Structure?

### Benefits

1. **Clear separation**: Users vs developers vs contributors
2. **Easy navigation**: Logical grouping helps find docs
3. **Scalable**: Easy to add new docs in right place
4. **Discoverable**: Structure mirrors typical user journey
5. **Maintainable**: Clear ownership and purpose per file

### Design Principles

1. **Progressive disclosure**: Simple → advanced
2. **Audience-focused**: Docs organized by who needs them
3. **Searchable**: Clear naming and structure
4. **DRY**: Link between docs rather than duplicate
5. **Current**: Keep close to code for easy updates

## Summary

The documentation is organized into:

- **2 quick-start docs** at root (getting started, quick ref)
- **2 user guides** in `guides/` (Monte Carlo, analytics)
- **1 API reference** in `api/` (auto-generated)
- **3 technical docs** in `technical/` (design, analysis, summary)
- **1 contributor guide** in `contributing/` (development)

Total: **9 main documents** + README + this structure doc

This structure provides clear paths for every audience from beginners to contributors.

