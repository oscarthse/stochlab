# CI/CD Implementation Plan for stochlab

**Date**: 2024-11-20  
**Status**: Design Document  
**Target**: Complete automation of testing, quality checks, releases, and documentation

---

## 📋 Table of Contents

1. [Overview & Goals](#overview--goals)
2. [Architecture](#architecture)
3. [Workflow Breakdown](#workflow-breakdown)
4. [Implementation Details](#implementation-details)
5. [External Services](#external-services)
6. [Step-by-Step Setup](#step-by-step-setup)
7. [Testing the Setup](#testing-the-setup)
8. [Best Practices](#best-practices)

---

## Overview & Goals

### What is CI/CD?

**CI (Continuous Integration)**:
- Automatically run tests on every push/PR
- Check code quality (linting, type checking)
- Ensure code always works before merging

**CD (Continuous Deployment/Delivery)**:
- Automatically build and publish to PyPI on releases
- Automatically deploy documentation to GitHub Pages
- Reduce manual errors and save time

### Goals for stochlab

1. **Quality Assurance**
   - Every PR must pass all tests
   - Code must pass linting and type checking
   - Coverage must not decrease

2. **Automated Releases**
   - Tag a version → automatically build and publish to PyPI
   - No manual `python -m build` or `twine upload`

3. **Documentation**
   - Every push to `main` → rebuild and deploy docs
   - Always up-to-date documentation

4. **Developer Experience**
   - Fast feedback (tests run in < 5 minutes)
   - Clear error messages
   - Easy to debug failures

---

## Architecture

### Workflow Structure

```
GitHub Repository
│
├── On Push/PR → CI Workflow
│   ├── Test Suite (pytest)
│   ├── Code Quality (ruff, mypy)
│   ├── Coverage Report (pytest-cov)
│   └── Upload Coverage (codecov)
│
├── On Push to main → Docs Workflow
│   ├── Build Sphinx Docs
│   └── Deploy to GitHub Pages
│
└── On Tag (v*) → Release Workflow
    ├── Build Package (wheel + sdist)
    ├── Run Tests (sanity check)
    ├── Publish to PyPI
    └── Create GitHub Release
```

### Workflow Files

We'll create **3 GitHub Actions workflows**:

1. **`.github/workflows/ci.yml`** - Continuous Integration
   - Runs on: push, pull_request
   - Tests on: Python 3.11, 3.12, 3.13
   - Checks: tests, linting, type checking, coverage

2. **`.github/workflows/docs.yml`** - Documentation Deployment
   - Runs on: push to main
   - Builds: Sphinx HTML
   - Deploys: GitHub Pages

3. **`.github/workflows/release.yml`** - PyPI Release
   - Runs on: tag push (v*)
   - Builds: distribution packages
   - Publishes: PyPI (and TestPyPI for testing)

---

## Workflow Breakdown

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Purpose**: Ensure code quality on every change

**Triggers**:
- Push to any branch
- Pull request (opened, synchronize, reopened)
- Manual dispatch (for testing)

**Jobs**:

#### Job 1: Test Suite
```yaml
- Matrix: Python 3.11, 3.12, 3.13
- OS: ubuntu-latest, macos-latest, windows-latest
- Steps:
  1. Checkout code
  2. Set up Python
  3. Install dependencies (uv sync)
  4. Install package in dev mode
  5. Run pytest with coverage
  6. Upload coverage to codecov
```

#### Job 2: Code Quality
```yaml
- OS: ubuntu-latest
- Steps:
  1. Checkout code
  2. Set up Python
  3. Install dependencies
  4. Run ruff (linting)
  5. Run mypy (type checking)
  6. Run black --check (formatting check)
```

#### Job 3: Build Check
```yaml
- OS: ubuntu-latest
- Steps:
  1. Checkout code
  2. Set up Python
  3. Install build tools
  4. Build wheel and sdist
  5. Verify package structure
```

**Success Criteria**:
- ✅ All tests pass on all Python versions
- ✅ No linting errors
- ✅ No type errors
- ✅ Package builds successfully
- ✅ Coverage uploaded

---

### 2. Documentation Workflow (`.github/workflows/docs.yml`)

**Purpose**: Keep documentation up-to-date

**Triggers**:
- Push to `main` branch
- Manual dispatch

**Jobs**:

#### Job: Build and Deploy Docs
```yaml
- OS: ubuntu-latest
- Steps:
  1. Checkout code
  2. Set up Python
  3. Install dependencies (including docs extras)
  4. Build Sphinx documentation
  5. Deploy to GitHub Pages
```

**Success Criteria**:
- ✅ Docs build without errors
- ✅ Deployed to `https://oscarthse.github.io/stochlab/`

**Note**: We already have a `docs.yml` file, but it might need updates.

---

### 3. Release Workflow (`.github/workflows/release.yml`)

**Purpose**: Automate PyPI releases

**Triggers**:
- Tag push matching `v*` (e.g., `v0.1.2`, `v0.2.0`)
- Manual dispatch (for testing)

**Jobs**:

#### Job 1: Build and Test
```yaml
- OS: ubuntu-latest
- Steps:
  1. Checkout code
  2. Set up Python
  3. Install dependencies
  4. Run tests (sanity check)
  5. Build distribution packages
  6. Upload artifacts
```

#### Job 2: Publish to TestPyPI
```yaml
- Depends on: Build and Test
- Steps:
  1. Download artifacts
  2. Publish to TestPyPI using twine
  3. Verify installation from TestPyPI
```

#### Job 3: Publish to PyPI
```yaml
- Depends on: Publish to TestPyPI
- Manual approval: Required (for safety)
- Steps:
  1. Download artifacts
  2. Publish to PyPI using twine
  3. Create GitHub Release
```

**Success Criteria**:
- ✅ Package builds successfully
- ✅ Tests pass
- ✅ Published to TestPyPI
- ✅ Published to PyPI (after approval)
- ✅ GitHub Release created

---

## Implementation Details

### Dependencies Needed

We'll need to add a few dev dependencies:

```toml
[project.optional-dependencies]
dev = [
  # ... existing ...
  "pytest-cov>=4.0.0",      # Coverage reporting
  "coverage[toml]>=7.0.0",  # Coverage configuration
]
```

### Configuration Files

#### 1. `.github/workflows/ci.yml`

**Key Features**:
- Matrix strategy for multiple Python versions
- Caching for faster builds (uv cache, pip cache)
- Parallel jobs for speed
- Coverage reporting with codecov
- Artifact uploads for test results

**Estimated Runtime**: 5-8 minutes

#### 2. `.github/workflows/docs.yml`

**Key Features**:
- Build Sphinx docs
- Deploy to GitHub Pages using `actions/upload-pages-artifact@v3`
- Only runs on main branch (saves CI minutes)

**Estimated Runtime**: 2-3 minutes

#### 3. `.github/workflows/release.yml`

**Key Features**:
- Tag-based triggering
- Two-stage release (TestPyPI → PyPI)
- Manual approval for production
- GitHub Release creation
- Version extraction from tag

**Estimated Runtime**: 5-10 minutes

### Secrets Required

**GitHub Secrets** (Settings → Secrets and variables → Actions):

1. **`PYPI_API_TOKEN`** (required for release workflow)
   - Get from: https://pypi.org/manage/account/token/
   - Scope: Entire account or project-specific
   - Used for: Publishing to PyPI

2. **`TEST_PYPI_API_TOKEN`** (optional, for testing)
   - Get from: https://test.pypi.org/manage/account/token/
   - Used for: Publishing to TestPyPI first

**Note**: These are only needed for the release workflow. CI and docs workflows don't need secrets.

### Coverage Configuration

Create `pyproject.toml` section or `.coveragerc`:

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/_build/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
```

### Code Quality Configuration

Already configured in `pyproject.toml`:
- `ruff` for linting
- `mypy` for type checking
- `black` for formatting (check mode in CI)

---

## External Services

### 1. Codecov (Coverage Reporting)

**What it does**:
- Tracks coverage over time
- Shows coverage diff on PRs
- Prevents coverage decreases (optional)

**Setup**:
1. Sign up at https://codecov.io (free for open source)
2. Connect GitHub repository
3. Get upload token (auto-configured via GitHub integration)
4. Add badge to README

**Benefits**:
- Visual coverage reports
- PR comments with coverage changes
- Historical tracking

**Alternative**: GitHub's built-in coverage (simpler, less features)

### 2. PyPI (Package Distribution)

**What it does**:
- Hosts Python packages
- Makes packages installable via `pip install stochlab`

**Setup**:
- Already have account (you uploaded 0.1.1)
- Need API token for automation

### 3. TestPyPI (Testing Releases)

**What it does**:
- Test package uploads before production
- Verify installation works
- Catch issues early

**Setup**:
- Create account at https://test.pypi.org
- Get API token
- Optional but recommended

### 4. GitHub Pages (Documentation)

**What it does**:
- Hosts static HTML (our Sphinx docs)
- Free for public repos
- Custom domain support

**Setup**:
- Enable in repository settings
- Configure workflow to deploy
- Already partially set up (we have `docs.yml`)

---

## Step-by-Step Setup

### Phase 1: Prepare Configuration Files

#### Step 1.1: Add Coverage Configuration

Add to `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if TYPE_CHECKING:",
]
```

#### Step 1.2: Update Dev Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
dev = [
  # ... existing ...
  "pytest-cov>=4.0.0",
  "coverage[toml]>=7.0.0",
]
```

#### Step 1.3: Create `.coveragerc` (Alternative)

If you prefer a separate file:

```ini
[run]
source = src
omit = 
    */tests/*
    */__pycache__/*

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    if TYPE_CHECKING:
```

### Phase 2: Create CI Workflow

#### Step 2.1: Create Directory

```bash
mkdir -p .github/workflows
```

#### Step 2.2: Create `.github/workflows/ci.yml`

This will be a comprehensive workflow with:
- Matrix testing (Python 3.11, 3.12, 3.13)
- Multiple OS (Ubuntu, macOS, Windows)
- Code quality checks
- Coverage reporting
- Build verification

### Phase 3: Update Documentation Workflow

#### Step 3.1: Review Existing `docs.yml`

Check if it needs updates:
- Use latest action versions
- Proper permissions
- Correct paths

#### Step 3.2: Update if Needed

Ensure it:
- Builds on push to main
- Uses `actions/upload-pages-artifact@v3`
- Has proper permissions

### Phase 4: Create Release Workflow

#### Step 4.1: Create `.github/workflows/release.yml`

This will:
- Trigger on version tags
- Build packages
- Test before release
- Publish to TestPyPI
- Publish to PyPI (with approval)
- Create GitHub Release

### Phase 5: Set Up Secrets

#### Step 5.1: Get PyPI API Token

1. Go to https://pypi.org/manage/account/token/
2. Create new token
3. Name: "stochlab-ci" or similar
4. Scope: Entire account (or project-specific)
5. Copy token (starts with `pypi-...`)

#### Step 5.2: Add to GitHub Secrets

1. Go to repository Settings
2. Secrets and variables → Actions
3. New repository secret
4. Name: `PYPI_API_TOKEN`
5. Value: Your token

#### Step 5.3: Optional - TestPyPI Token

Same process for TestPyPI:
- https://test.pypi.org/manage/account/token/
- Secret name: `TEST_PYPI_API_TOKEN`

### Phase 6: Set Up Codecov (Optional)

#### Step 6.1: Sign Up

1. Go to https://codecov.io
2. Sign in with GitHub
3. Add repository: `oscarthse/stochlab`

#### Step 6.2: Get Badge

Add to README.md:

```markdown
[![codecov](https://codecov.io/gh/oscarthse/stochlab/branch/main/graph/badge.svg)](https://codecov.io/gh/oscarthse/stochlab)
```

### Phase 7: Enable GitHub Pages

#### Step 7.1: Configure Settings

1. Repository Settings → Pages
2. Source: GitHub Actions
3. Save

#### Step 7.2: Verify Workflow

The docs workflow should automatically deploy.

### Phase 8: Test Everything

#### Step 8.1: Test CI Workflow

1. Make a small change
2. Push to a branch
3. Open PR
4. Watch CI run
5. Verify all checks pass

#### Step 8.2: Test Docs Workflow

1. Merge PR to main
2. Watch docs workflow run
3. Verify docs deploy

#### Step 8.3: Test Release Workflow (Dry Run)

1. Create a test tag: `v0.1.1-test`
2. Push tag
3. Watch release workflow
4. Verify it builds (don't approve PyPI publish)

---

## Testing the Setup

### Test Checklist

**CI Workflow**:
- [ ] Tests run on push
- [ ] Tests run on PR
- [ ] All Python versions pass
- [ ] Linting works
- [ ] Type checking works
- [ ] Coverage uploads
- [ ] Build check passes

**Docs Workflow**:
- [ ] Builds on push to main
- [ ] Deploys to GitHub Pages
- [ ] Docs are accessible

**Release Workflow**:
- [ ] Triggers on tag
- [ ] Builds packages
- [ ] Tests pass
- [ ] Publishes to TestPyPI (if configured)
- [ ] Ready for PyPI approval

### Common Issues & Solutions

**Issue**: Tests fail on Windows
- **Solution**: Check path separators, line endings

**Issue**: Coverage not uploading
- **Solution**: Check codecov token, verify upload step

**Issue**: Docs not deploying
- **Solution**: Check GitHub Pages settings, permissions

**Issue**: Release workflow not triggering
- **Solution**: Verify tag format (`v*`), check workflow file

---

## Best Practices

### 1. Workflow Organization

- **Keep workflows focused**: One workflow per purpose
- **Use matrix strategies**: Test multiple versions efficiently
- **Cache dependencies**: Speed up builds
- **Fail fast**: Run quick checks first

### 2. Security

- **Never commit secrets**: Use GitHub Secrets
- **Limit token scope**: Use project-specific tokens when possible
- **Review permissions**: Only grant necessary permissions
- **Rotate tokens**: Periodically regenerate API tokens

### 3. Performance

- **Parallel jobs**: Run independent checks in parallel
- **Caching**: Cache `uv` packages, pip dependencies
- **Skip unnecessary runs**: Use `paths` filters when appropriate
- **Optimize test suite**: Group fast tests, slow tests separately

### 4. Developer Experience

- **Clear error messages**: Use descriptive step names
- **Fast feedback**: Run critical checks first
- **Artifact uploads**: Save test results, logs
- **Status badges**: Show CI status in README

### 5. Release Process

- **TestPyPI first**: Always test before production
- **Manual approval**: Require approval for PyPI publish
- **Version validation**: Verify version matches tag
- **Release notes**: Auto-generate from CHANGELOG

---

## Workflow File Templates

### Complete CI Workflow

See next section for full `.github/workflows/ci.yml` implementation.

### Complete Release Workflow

See next section for full `.github/workflows/release.yml` implementation.

---

## Success Metrics

After implementation, you should have:

✅ **Automated Testing**
- Every PR tested automatically
- Multiple Python versions verified
- Code quality enforced

✅ **Automated Releases**
- Tag → PyPI in minutes
- No manual build/upload steps
- Consistent release process

✅ **Automated Documentation**
- Always up-to-date docs
- Deployed on every main push
- Accessible at GitHub Pages

✅ **Developer Confidence**
- Know immediately if code breaks
- Coverage tracking
- Clear quality gates

---

## Timeline Estimate

**Phase 1-2**: Configuration (30 minutes)
- Add coverage config
- Update dependencies

**Phase 3-4**: Workflow Creation (1-2 hours)
- Write CI workflow
- Write release workflow
- Update docs workflow

**Phase 5-6**: External Setup (30 minutes)
- Set up secrets
- Configure codecov
- Enable GitHub Pages

**Phase 7-8**: Testing (1 hour)
- Test all workflows
- Fix any issues
- Verify end-to-end

**Total**: **3-4 hours** for complete setup

---

## Next Steps

1. **Review this plan** - Make sure it aligns with your needs
2. **Decide on scope** - Start with CI, add release later?
3. **Set up secrets** - Get PyPI tokens ready
4. **Create workflows** - I'll help you write the actual YAML files
5. **Test incrementally** - Start with CI, then docs, then release

---

**Ready to implement?** Let me know and I'll create the actual workflow files! 🚀

