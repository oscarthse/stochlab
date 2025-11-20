# CI/CD Complete Tutorial

**A comprehensive guide to understanding and using stochlab's CI/CD pipeline**

---

## 📚 Table of Contents

1. [What is CI/CD?](#what-is-cicd)
2. [What We've Built](#what-weve-built)
3. [How It Works - Step by Step](#how-it-works---step-by-step)
4. [The Three Workflows Explained](#the-three-workflows-explained)
5. [Real-World Usage Examples](#real-world-usage-examples)
6. [Understanding the Output](#understanding-the-output)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## What is CI/CD?

### CI = Continuous Integration

**The Problem It Solves:**
- You write code on your computer
- You push it to GitHub
- But does it actually work? Does it break anything?
- You'd have to manually test every time

**The Solution:**
- **Automatically** test your code every time you push
- Run tests on multiple Python versions
- Check code quality (formatting, linting)
- Verify the package builds correctly
- All without you doing anything!

**Analogy:** Like having a robot assistant that checks your homework before you turn it in.

### CD = Continuous Deployment/Delivery

**The Problem It Solves:**
- You want to release a new version
- You have to manually:
  1. Build the package
  2. Run tests
  3. Upload to PyPI
  4. Update documentation
  5. Create a release
- Easy to make mistakes or forget steps

**The Solution:**
- **Automatically** do all of this when you create a version tag
- No manual steps
- Consistent, repeatable process

**Analogy:** Like having a robot that packages, ships, and delivers your product automatically.

---

## What We've Built

### The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR WORKFLOW                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  1. You write code locally        │
        │  2. You commit and push           │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  CI Workflow (Automatic)          │
        │  • Runs tests                     │
        │  • Checks code quality            │
        │  • Verifies package builds         │
        │  • Reports coverage                │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  Docs Workflow (Automatic)         │
        │  • Builds documentation            │
        │  • Deploys to GitHub Pages         │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  Release Workflow (On Tag)        │
        │  • Builds distribution packages   │
        │  • Publishes to PyPI               │
        │  • Creates GitHub Release          │
        └───────────────────────────────────┘
```

---

## How It Works - Step by Step

### Scenario 1: You Make a Code Change

#### Step 1: You Write Code

```bash
# On your computer
cd stochlab
# Edit some file
vim src/stochlab/models/markov_chain.py
```

#### Step 2: You Commit and Push

```bash
git add src/stochlab/models/markov_chain.py
git commit -m "feat: add new method to MarkovChain"
git push origin main
```

#### Step 3: GitHub Detects the Push

- GitHub sees: "New code pushed to `main`"
- GitHub checks: "Do I have workflows that trigger on push to main?"
- GitHub finds: `ci.yml` and `docs.yml`
- GitHub starts both workflows

#### Step 4: CI Workflow Runs (5-8 minutes)

**What happens:**

1. **GitHub spins up virtual machines**
   - Ubuntu, macOS, Windows
   - Python 3.11, 3.12, 3.13

2. **Each machine:**
   - Checks out your code
   - Installs dependencies
   - Runs tests
   - Checks code quality
   - Reports results

3. **You see results:**
   - Go to: https://github.com/oscarthse/stochlab/actions
   - See green checkmarks ✅ or red X's ❌
   - Click to see detailed logs

#### Step 5: Docs Workflow Runs (2-3 minutes)

**What happens:**

1. **GitHub spins up a virtual machine**
2. **Builds Sphinx documentation:**
   - Converts Markdown to HTML
   - Generates API reference
   - Creates search index
3. **Deploys to GitHub Pages:**
   - Uploads HTML files
   - Makes them available at: https://oscarthse.github.io/stochlab/

#### Step 6: You See the Results

- ✅ All tests passed → Code is good!
- ❌ Tests failed → Fix the issues
- ✅ Docs updated → Documentation is live!

---

### Scenario 2: You Open a Pull Request

#### Step 1: You Create a Feature Branch

```bash
git checkout -b feature/new-model
# Make changes
git commit -m "feat: add birth-death process model"
git push origin feature/new-model
```

#### Step 2: You Open a Pull Request

- Go to GitHub
- Click "New Pull Request"
- Select: `main` ← `feature/new-model`
- Click "Create Pull Request"

#### Step 3: CI Workflow Runs Automatically

**What happens:**

1. **GitHub detects:** "New PR opened"
2. **GitHub triggers:** CI workflow
3. **CI runs on the PR branch:**
   - Tests your new code
   - Checks code quality
   - Verifies it builds
   - Compares coverage

#### Step 4: You See Status Checks

On the PR page, you see:

```
✅ CI / Test (Python 3.11 on ubuntu-latest)
✅ CI / Test (Python 3.12 on ubuntu-latest)
✅ CI / Code Quality
✅ CI / Build Package
✅ CI / Import Check
```

#### Step 5: You Review and Merge

- If all checks pass ✅ → Safe to merge
- If checks fail ❌ → Fix issues first
- Merge when ready → Code goes to `main`

**Key Benefit:** You know the code works **before** it's merged!

---

### Scenario 3: You Release a New Version

#### Step 1: You Update Version

```bash
# Edit pyproject.toml
version = "0.1.2"  # Changed from 0.1.1

# Update CHANGELOG.md
# Add release notes for 0.1.2
```

#### Step 2: You Commit and Tag

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.1.2"
git tag v0.1.2
git push origin main
git push origin v0.1.2  # This triggers the release!
```

#### Step 3: Release Workflow Runs Automatically

**What happens:**

1. **GitHub detects:** "Tag `v0.1.2` was pushed"
2. **GitHub triggers:** Release workflow

3. **Job 1: Build and Test**
   - Verifies version matches tag
   - Runs tests (sanity check)
   - Builds wheel and sdist
   - Uploads artifacts

4. **Job 2: Publish to TestPyPI**
   - Downloads build artifacts
   - Publishes to TestPyPI
   - Verifies installation works

5. **Job 3: Publish to PyPI** (requires approval)
   - Waits for manual approval
   - You click "Approve" in GitHub
   - Publishes to production PyPI
   - Users can now `pip install stochlab==0.1.2`

6. **Job 4: Create GitHub Release**
   - Extracts release notes from CHANGELOG
   - Creates GitHub Release
   - Attaches distribution files

#### Step 4: You Verify

```bash
# Wait a few minutes, then:
pip install --upgrade stochlab
python -c "import stochlab; print(stochlab.__version__)"
# Output: 0.1.2
```

**Key Benefit:** One command (`git push origin v0.1.2`) does everything!

---

## The Three Workflows Explained

### 1. CI Workflow (`.github/workflows/ci.yml`)

**When it runs:**
- Push to any branch
- Pull request (opened, updated)
- Manual trigger

**What it does:**

#### Job 1: Test Suite

```yaml
Matrix Strategy:
  Python: [3.11, 3.12, 3.13]
  OS: [ubuntu, macos, windows]
  
For each combination:
  1. Checkout code
  2. Install Python
  3. Install dependencies (uv sync)
  4. Run pytest with coverage
  5. Upload coverage to Codecov
```

**Result:** 9 test jobs (3 Python × 3 OS, but reduced matrix)

**Why multiple versions?**
- Ensures compatibility
- Catches version-specific bugs
- Gives confidence

#### Job 2: Code Quality

```yaml
Steps:
  1. Check formatting (black --check)
  2. Lint code (ruff check)
  3. Type check (mypy)
```

**Result:** One job that ensures code quality

**Why?**
- Consistent code style
- Catch bugs early
- Maintainable codebase

#### Job 3: Build Package

```yaml
Steps:
  1. Build wheel and sdist
  2. Verify files exist
  3. Test installation
```

**Result:** One job that verifies packaging

**Why?**
- Ensures package can be distributed
- Catches packaging errors
- Verifies setup.py/pyproject.toml

#### Job 4: Import Check

```yaml
Steps:
  1. Install package
  2. Test all imports work
```

**Result:** One job that verifies imports

**Why?**
- Ensures public API works
- Catches import errors
- Verifies package structure

**Total Runtime:** ~5-8 minutes

---

### 2. Docs Workflow (`.github/workflows/docs.yml`)

**When it runs:**
- Push to `main` branch
- Manual trigger

**What it does:**

```yaml
Steps:
  1. Checkout code
  2. Install Python and dependencies
  3. Build Sphinx documentation
  4. Deploy to GitHub Pages
```

**Result:** Updated documentation at https://oscarthse.github.io/stochlab/

**Why only on main?**
- Saves CI minutes (docs don't need to build on every branch)
- Only deploy "official" documentation
- Faster workflow

**Total Runtime:** ~2-3 minutes

---

### 3. Release Workflow (`.github/workflows/release.yml`)

**When it runs:**
- Tag push matching `v*` (e.g., `v0.1.2`)
- Manual trigger

**What it does:**

#### Job 1: Build and Test

```yaml
Steps:
  1. Extract version from tag
  2. Verify version matches pyproject.toml
  3. Run tests (sanity check)
  4. Build distribution packages
  5. Upload artifacts
```

**Why verify version?**
- Prevents mistakes (tag v0.1.2 but pyproject.toml says 0.1.1)
- Ensures consistency

#### Job 2: Publish to TestPyPI

```yaml
Steps:
  1. Download build artifacts
  2. Publish to TestPyPI
  3. Verify installation
```

**Why TestPyPI first?**
- Test the release process
- Catch issues before production
- Verify installation works

#### Job 3: Publish to PyPI

```yaml
Steps:
  1. Wait for manual approval
  2. Download artifacts
  3. Publish to PyPI
  4. Verify installation
```

**Why manual approval?**
- Safety check
- Prevents accidental releases
- Gives you control

#### Job 4: Create GitHub Release

```yaml
Steps:
  1. Extract release notes from CHANGELOG.md
  2. Create GitHub Release
  3. Attach distribution files
```

**Why?**
- Professional releases
- Easy to find downloads
- Release notes for users

**Total Runtime:** ~5-10 minutes

---

## Real-World Usage Examples

### Example 1: Daily Development

**You:** "I want to add a new feature"

```bash
# 1. Create feature branch
git checkout -b feature/stationary-distribution

# 2. Write code
vim src/stochlab/analytics/markov.py
# Add new function

# 3. Write tests
vim tests/test_analytics_markov.py
# Add test for new function

# 4. Commit
git add .
git commit -m "feat: add stationary distribution calculation"
git push origin feature/stationary-distribution
```

**CI automatically:**
- ✅ Runs all tests (including your new test)
- ✅ Checks code quality
- ✅ Verifies imports work
- ✅ Reports coverage

**You:**
- Open PR
- See all checks pass ✅
- Get code review
- Merge with confidence

**Result:** Feature is added, tested, and documented automatically!

---

### Example 2: Fixing a Bug

**You:** "User reported a bug in RandomWalk"

```bash
# 1. Create bugfix branch
git checkout -b fix/random-walk-boundary

# 2. Write failing test (reproduce bug)
vim tests/test_random_walk.py
# Add test that fails

# 3. Fix the bug
vim src/stochlab/models/random_walk.py
# Fix the issue

# 4. Verify test passes
pytest tests/test_random_walk.py

# 5. Commit
git add .
git commit -m "fix: correct boundary handling in RandomWalk"
git push origin fix/random-walk-boundary
```

**CI automatically:**
- ✅ Runs all tests (your new test now passes)
- ✅ Ensures you didn't break anything else
- ✅ Verifies code quality

**You:**
- Open PR
- See bug is fixed ✅
- Merge

**Result:** Bug fixed, regression test added, users happy!

---

### Example 3: Releasing Version 0.2.0

**You:** "Time to release the new version"

```bash
# 1. Update version
vim pyproject.toml
# Change: version = "0.2.0"

# 2. Update changelog
vim CHANGELOG.md
# Add section for 0.2.0 with all changes

# 3. Commit
git add pyproject.toml CHANGELOG.md
git commit -m "chore: prepare release 0.2.0"

# 4. Push to main
git push origin main

# 5. Create and push tag
git tag v0.2.0
git push origin v0.2.0
```

**Release workflow automatically:**
- ✅ Builds packages
- ✅ Runs tests
- ✅ Publishes to TestPyPI
- ⏸️ Waits for your approval
- ✅ Publishes to PyPI (after approval)
- ✅ Creates GitHub Release

**You:**
- Go to GitHub Actions
- See release workflow running
- Click "Approve" when TestPyPI looks good
- Wait for PyPI publish
- Done! 🎉

**Result:** Version 0.2.0 is live, users can install it, release notes are published!

---

## Understanding the Output

### CI Workflow Output

When you go to: https://github.com/oscarthse/stochlab/actions

You'll see:

```
┌─────────────────────────────────────┐
│ CI #123                            │
│ test/ci-workflow → main            │
│ 5 jobs, 8m 23s                    │
└─────────────────────────────────────┘
```

Click to see:

```
Jobs:
  ✅ Test (Python 3.11 on ubuntu-latest) - 27s
  ✅ Test (Python 3.12 on ubuntu-latest) - 29s
  ✅ Test (Python 3.11 on macos-latest) - 38s
  ✅ Code Quality - 11s
  ✅ Build Package - 20s
  ✅ Import Check - 9s
```

Click a job to see detailed logs:

```
Run tests with coverage
  uv run pytest tests/ --cov=src/stochlab ...
  
  tests/test_markov_chain.py::test_basic ... PASSED
  tests/test_markov_chain.py::test_transition ... PASSED
  ...
  
  ========== 98 passed in 5.23s ==========
  
  Coverage: 87%
```

### What Each Status Means

- ✅ **Green checkmark**: Job passed, everything good
- ❌ **Red X**: Job failed, something needs fixing
- 🟡 **Yellow circle**: Job is running
- ⏸️ **Paused**: Waiting for approval (release workflow)

### Understanding Failures

If a job fails, click it to see:

```
❌ Code Quality
  ❌ Check code formatting with black
    Error: 17 files would be reformatted
```

**What to do:**
1. Read the error message
2. Fix the issue locally
3. Commit and push
4. CI will run again

---

## Troubleshooting

### Problem: CI Fails with "Import Error"

**Symptoms:**
```
❌ Import Check
  Error: No module named 'stochlab'
```

**Solution:**
- Check that `__init__.py` files exist
- Verify package structure
- Check `pyproject.toml` configuration

### Problem: Tests Fail on Windows Only

**Symptoms:**
```
✅ Test (Python 3.11 on ubuntu-latest)
✅ Test (Python 3.11 on macos-latest)
❌ Test (Python 3.11 on windows-latest)
```

**Solution:**
- Check for path separators (`/` vs `\`)
- Check for line ending issues
- Test locally on Windows if possible

### Problem: Release Workflow Doesn't Trigger

**Symptoms:**
- You pushed a tag, but nothing happened

**Solution:**
- Check tag format: Must be `v*` (e.g., `v0.1.2`, not `0.1.2`)
- Check workflow file exists: `.github/workflows/release.yml`
- Check Actions tab for any errors

### Problem: Docs Don't Deploy

**Symptoms:**
- Docs workflow runs but site doesn't update

**Solution:**
- Check GitHub Pages settings (must be "GitHub Actions")
- Check workflow permissions
- Wait a few minutes (deployment can be slow)

---

## Best Practices

### 1. Always Check CI Before Merging

**Don't:**
- Merge PRs with failing CI
- "It works on my machine" mentality

**Do:**
- Wait for CI to pass
- Fix failures before merging
- Review CI output

### 2. Write Tests for New Features

**Don't:**
- Add code without tests
- Skip testing "simple" changes

**Do:**
- Write tests first (TDD)
- Test edge cases
- Aim for high coverage

### 3. Keep PRs Small

**Don't:**
- Huge PRs with 50+ files
- Multiple unrelated changes

**Do:**
- One feature per PR
- Small, focused changes
- Easier to review and test

### 4. Use Descriptive Commit Messages

**Don't:**
```
git commit -m "fix"
git commit -m "update"
```

**Do:**
```
git commit -m "fix: correct boundary handling in RandomWalk"
git commit -m "feat: add stationary distribution calculation"
```

### 5. Test Locally First

**Don't:**
- Push broken code and let CI find issues
- Use CI as your primary testing

**Do:**
- Run tests locally: `pytest tests/`
- Check formatting: `black --check src/`
- Lint code: `ruff check src/`
- Then push

---

## Summary

### What You Get

✅ **Automated Testing**
- Every push is tested
- Multiple Python versions
- Multiple operating systems
- Fast feedback

✅ **Code Quality**
- Consistent formatting
- Linting catches bugs
- Type checking
- High coverage

✅ **Automated Releases**
- Tag → PyPI automatically
- No manual steps
- Consistent process

✅ **Always Up-to-Date Docs**
- Deploy on every main push
- No manual deployment
- Professional documentation

### What You Don't Have to Do

❌ Manually run tests
❌ Manually check code quality
❌ Manually build packages
❌ Manually upload to PyPI
❌ Manually deploy documentation
❌ Worry about forgetting steps

### The Bottom Line

**Before CI/CD:**
- Push code → Hope it works
- Release → 10+ manual steps, easy to make mistakes
- Docs → Manual build and deploy

**After CI/CD:**
- Push code → Automatically tested ✅
- Release → One command, everything automated ✅
- Docs → Automatically deployed ✅

**You focus on writing code. CI/CD handles the rest!** 🚀

---

**Questions?** Check the [CI/CD Setup Guide](../.github/CICD_SETUP.md) or the [Detailed Plan](cicd_plan.md).

