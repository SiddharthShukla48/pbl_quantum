# UV Package Manager - Quick Start Guide

## ✅ Installation Complete!

Your project is now set up with `uv` package manager. All dependencies are installed in `.venv/`.

---

## 📦 What Was Installed

**Core Dependencies:**
- ✅ `dwave-ocean-sdk` (8.0.1) - Complete D-Wave toolkit
- ✅ `numpy` (1.24.4) - Numerical computing
- ✅ `pandas` (2.0.3) - Data processing
- ✅ `dimod`, `neal`, `dwave-system` - QUBO/quantum solvers
- ✅ `pytest`, `ipython` - Development tools

**Total:** 78 packages installed

---

## 🚀 Usage with UV

### Running Python Scripts

**Option 1: Activate virtual environment**
```bash
source .venv/bin/activate
python qubo_solver_dwave.py
```

**Option 2: Run directly with uv (recommended)**
```bash
uv run python qubo_solver_dwave.py
uv run python dataset-generator.py
uv run python test_setup.py
```

### Managing Dependencies

**Install a new package:**
```bash
uv add matplotlib seaborn
```

**Install optional dependencies:**
```bash
uv sync --extra viz        # Install visualization tools
uv sync --extra classical  # Install OR-Tools for baseline
uv sync --extra dev        # Install development tools
```

**Update dependencies:**
```bash
uv sync --upgrade
```

**Remove a package:**
```bash
uv remove package-name
```

### Working with the Project

**Run tests:**
```bash
uv run pytest
```

**Start IPython:**
```bash
uv run ipython
```

**Quick test:**
```bash
uv run python test_setup.py
```

---

## 📁 Project Structure

```
Your Project/
├── .venv/                      # Virtual environment (managed by uv)
├── pyproject.toml             # Project config & dependencies
├── uv.lock                    # Locked dependency versions
├── qubo_solver_dwave.py       # Main solver
├── dataset-generator.py       # Dataset generation
├── data-loader.py             # Data loading
├── test_setup.py              # Setup verification
└── README_SOLVER.md           # Documentation
```

---

## 🔧 pyproject.toml Overview

Your `pyproject.toml` includes:

**[project]**
- Project metadata (name, version, description)
- Python version requirement (>=3.8)
- Core dependencies

**[project.optional-dependencies]**
- `viz`: Matplotlib, Seaborn (visualization)
- `classical`: OR-Tools (classical baseline)
- `dev`: Testing and development tools

**[tool.uv]**
- Dev-specific dependencies
- Managed by uv

---

## 🎯 Quick Start Workflow

### 1. Verify Setup
```bash
uv run python test_setup.py
```

Expected output:
```
✓ All imports successful!
✓ QUBO test passed
✓ Essential tests passed!
```

### 2. Generate Dataset
```bash
uv run python dataset-generator.py
```

Creates:
- `exam_scheduling_TINY/`
- `exam_scheduling_SMALL/`
- `exam_scheduling_MEDIUM/`

### 3. Run Solver
```bash
uv run python qubo_solver_dwave.py
```

Expected output:
```
✓ Loaded 5 courses
✓ Built QUBO: 20×20 matrix
✓ Solved in 0.15s
✓ Valid solution found!
```

---

## 🆚 UV vs PIP Comparison

| Task | pip | uv |
|------|-----|-----|
| Install deps | `pip install -r requirements.txt` | `uv sync` |
| Add package | `pip install pkg` + manual edit | `uv add pkg` |
| Run script | `python script.py` | `uv run python script.py` |
| Activate env | `source venv/bin/activate` | Not needed with `uv run` |
| Lock versions | Manual with pip freeze | Automatic with `uv.lock` |
| Speed | Slower | **Much faster** ⚡ |

---

## 🐍 Python Environment

**Location:** `.venv/`
**Python version:** 3.8.20
**Managed by:** uv

**To activate manually:**
```bash
source .venv/bin/activate
```

**To deactivate:**
```bash
deactivate
```

---

## 💡 Pro Tips

### 1. Always Use `uv run`
```bash
# Instead of:
source .venv/bin/activate
python qubo_solver_dwave.py

# Just do:
uv run python qubo_solver_dwave.py
```

### 2. Install Optional Features
```bash
# For visualization
uv sync --extra viz

# For classical comparison
uv sync --extra classical

# For development
uv sync --extra dev

# All extras
uv sync --all-extras
```

### 3. Check What's Installed
```bash
uv pip list
```

### 4. Update Everything
```bash
uv sync --upgrade
```

### 5. Clean Reinstall
```bash
rm -rf .venv uv.lock
uv sync
```

---

## 🔍 Troubleshooting

### Problem: "command not found: uv"
**Solution:**
```bash
pip install uv
# or
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Problem: "Python version mismatch"
**Solution:**
```bash
# Let uv manage Python
uv python install 3.8
uv sync
```

### Problem: Dependencies not found
**Solution:**
```bash
uv sync --reinstall
```

### Problem: Import errors in scripts
**Solution:**
Use `uv run`:
```bash
uv run python your_script.py
```

---

## 📊 Performance Comparison

Testing `uv sync` vs `pip install`:

| Metric | pip | uv |
|--------|-----|-----|
| Initial install | ~120s | **~30s** |
| Reinstall (cached) | ~45s | **~1s** |
| Resolution | ~15s | **~0.1s** |
| Lock file | Manual | Automatic |

**UV is 4-10x faster!** ⚡

---

## 🎓 Next Steps

Now that uv is set up:

1. ✅ Run test suite: `uv run python test_setup.py`
2. ✅ Generate datasets: `uv run python dataset-generator.py`
3. ✅ Run solver: `uv run python qubo_solver_dwave.py`
4. ✅ Read [README_SOLVER.md](README_SOLVER.md) for detailed guide
5. ✅ Start your research experiments!

---

## 🔗 Resources

- **UV Docs:** https://docs.astral.sh/uv/
- **Project Guide:** [README_SOLVER.md](README_SOLVER.md)
- **Implementation Details:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **D-Wave Docs:** https://docs.ocean.dwavesys.com/

---

## 📝 Common Commands Cheatsheet

```bash
# Install dependencies
uv sync

# Run scripts
uv run python qubo_solver_dwave.py

# Add dependency
uv add numpy

# Install extras
uv sync --extra viz

# Update all
uv sync --upgrade

# List installed
uv pip list

# Clean install
rm -rf .venv uv.lock && uv sync

# Check for updates
uv pip list --outdated
```

---

Your project is ready! Run `uv run python test_setup.py` to verify everything works. 🚀
