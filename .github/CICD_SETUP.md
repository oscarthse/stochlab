# CI/CD Setup Summary

This document summarizes the CI/CD implementation for stochlab.

## ✅ What's Been Created

### 1. Workflow Files

- **`.github/workflows/ci.yml`** - Continuous Integration
  - Tests on Python 3.11, 3.12, 3.13
  - Tests on Ubuntu, macOS, Windows
  - Code quality checks (black, ruff, mypy)
  - Coverage reporting
  - Package build verification

- **`.github/workflows/docs.yml`** - Documentation Deployment
  - Already existed, verified and working
  - Builds Sphinx docs on push to main
  - Deploys to GitHub Pages

- **`.github/workflows/release.yml`** - Automated Releases
  - Triggers on version tags (v*)
  - Builds distribution packages
  - Publishes to TestPyPI first
  - Publishes to PyPI (with manual approval)
  - Creates GitHub Release

### 2. Configuration Updates

- **`pyproject.toml`**:
  - Added `pytest-cov` and `coverage[toml]` to dev dependencies
  - Added coverage configuration section
  - Configured coverage exclusions and reporting

### 3. Documentation

- **`docs/technical/cicd_plan.md`** - Comprehensive plan document
  - Architecture overview
  - Step-by-step setup guide
  - Best practices
  - Troubleshooting

## 🔧 Setup Required

### Step 1: Install New Dependencies

```bash
uv sync --extra dev
```

This will install `pytest-cov` and `coverage[toml]` needed for coverage reporting.

### Step 2: Set Up GitHub Secrets

Go to: **Repository Settings → Secrets and variables → Actions**

#### Required Secret:
- **`PYPI_API_TOKEN`**
  - Get from: https://pypi.org/manage/account/token/
  - Create new token with "Entire account" or project scope
  - Copy token (starts with `pypi-...`)
  - Add as repository secret

#### Optional Secret (for TestPyPI):
- **`TEST_PYPI_API_TOKEN`**
  - Get from: https://test.pypi.org/manage/account/token/
  - Same process as above
  - Only needed if you want to test releases on TestPyPI first

### Step 3: Enable GitHub Pages

1. Go to **Repository Settings → Pages**
2. Source: **GitHub Actions**
3. Save

The docs workflow will automatically deploy.

### Step 4: Set Up Codecov (Optional but Recommended)

1. Sign up at https://codecov.io (free for open source)
2. Connect GitHub account
3. Add repository: `oscarthse/stochlab`
4. Get badge code and add to README.md:

```markdown
[![codecov](https://codecov.io/gh/oscarthse/stochlab/branch/main/graph/badge.svg)](https://codecov.io/gh/oscarthse/stochlab)
```

### Step 5: Configure Release Environment (Manual Approval)

1. Go to **Repository Settings → Environments**
2. Create environment: **`pypi`**
3. Enable **Required reviewers** (optional but recommended)
4. This ensures manual approval before publishing to PyPI

## 🧪 Testing the Setup

### Test CI Workflow

1. Make a small change (e.g., add a comment)
2. Push to a branch
3. Open a Pull Request
4. Watch the CI workflow run
5. Verify:
   - ✅ Tests pass
   - ✅ Code quality checks pass
   - ✅ Coverage uploads (if codecov set up)

### Test Docs Workflow

1. Merge a PR to `main`
2. Watch the docs workflow run
3. Verify docs deploy to GitHub Pages
4. Check: https://oscarthse.github.io/stochlab/

### Test Release Workflow (Dry Run)

1. Create a test tag:
   ```bash
   git tag v0.1.1-test
   git push origin v0.1.1-test
   ```
2. Watch the release workflow run
3. Verify:
   - ✅ Build succeeds
   - ✅ Tests pass
   - ✅ Packages are built
   - ⚠️ Don't approve PyPI publish for test tag

## 📊 What Each Workflow Does

### CI Workflow (`ci.yml`)

**Triggers**: Push, PR, manual

**Jobs**:
1. **Test** - Runs pytest on multiple Python versions and OS
2. **Quality** - Runs black, ruff, mypy
3. **Build** - Verifies package builds correctly
4. **Import Check** - Verifies all imports work

**Duration**: ~5-8 minutes

### Docs Workflow (`docs.yml`)

**Triggers**: Push to `main`

**Jobs**:
1. **Build** - Builds Sphinx documentation
2. **Deploy** - Deploys to GitHub Pages

**Duration**: ~2-3 minutes

### Release Workflow (`release.yml`)

**Triggers**: Tag push (v*), manual

**Jobs**:
1. **Build and Test** - Builds packages, runs tests
2. **Publish to TestPyPI** - Tests release process
3. **Publish to PyPI** - Production release (requires approval)
4. **Create Release** - Creates GitHub Release with notes

**Duration**: ~5-10 minutes

## 🚀 Release Process

Once set up, releasing is simple:

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Commit and push**:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "chore: bump version to 0.1.2"
   git push
   ```
4. **Create and push tag**:
   ```bash
   git tag v0.1.2
   git push origin v0.1.2
   ```
5. **Watch workflow run**:
   - Builds packages
   - Publishes to TestPyPI
   - Waits for approval
   - Publishes to PyPI
   - Creates GitHub Release

**That's it!** No manual `python -m build` or `twine upload` needed.

## 📈 Monitoring

### CI Status Badge

Add to README.md:

```markdown
[![CI](https://github.com/oscarthse/stochlab/actions/workflows/ci.yml/badge.svg)](https://github.com/oscarthse/stochlab/actions/workflows/ci.yml)
```

### Coverage Badge

If using Codecov:

```markdown
[![codecov](https://codecov.io/gh/oscarthse/stochlab/branch/main/graph/badge.svg)](https://codecov.io/gh/oscarthse/stochlab)
```

## 🔍 Troubleshooting

### CI Fails

- Check workflow logs in Actions tab
- Common issues:
  - Import errors → Check dependencies
  - Test failures → Fix tests
  - Linting errors → Run `ruff check` locally

### Docs Don't Deploy

- Check GitHub Pages settings (must be "GitHub Actions")
- Check workflow permissions
- Verify docs build locally: `make docs`

### Release Fails

- Verify tag format: `v0.1.2` (not `0.1.2`)
- Check version matches `pyproject.toml`
- Verify PyPI token is set correctly
- Check if version already exists (workflow skips existing)

## 📚 Further Reading

- [CI/CD Plan](docs/technical/cicd_plan.md) - Detailed technical documentation
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [PyPI Publishing Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)

## ✅ Checklist

- [ ] Install new dependencies (`uv sync --extra dev`)
- [ ] Set up `PYPI_API_TOKEN` secret
- [ ] (Optional) Set up `TEST_PYPI_API_TOKEN` secret
- [ ] Enable GitHub Pages (GitHub Actions source)
- [ ] (Optional) Set up Codecov
- [ ] (Optional) Configure `pypi` environment with approval
- [ ] Test CI workflow (open a PR)
- [ ] Test docs workflow (merge to main)
- [ ] Add CI badge to README
- [ ] (Optional) Add coverage badge to README

---

**Status**: ✅ Workflows created and ready to use  
**Next Step**: Set up secrets and test!

