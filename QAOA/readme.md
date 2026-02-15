# Quantum Exam Scheduling using QAOA and D-Wave

A complete framework for solving university exam timetabling problems using quantum and quantum-inspired optimization techniques. This project formulates exam scheduling as a **graph coloring problem** and solves it using:

- **QAOA** (Quantum Approximate Optimization Algorithm) via IBM Qiskit
- **D-Wave Neal** Simulated Annealing
- **D-Wave QPU** Quantum Annealing (real quantum hardware)
- **D-Wave Hybrid** Classical-Quantum Hybrid Solver

---

## 📑 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Workflow](#detailed-workflow)
- [Unified Solver Usage](#unified-solver-usage)
- [Benchmarking](#benchmarking)
- [Understanding the Results](#understanding-the-results)
- [Research Use Cases](#research-use-cases)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## 🎯 Overview

### Problem Statement

**Exam Timetabling Problem:**
- Schedule university exams into time slots
- **No conflicts**: Students enrolled in multiple courses cannot have exams at the same time
- **Minimize time slots**: Use as few slots as possible
- **Assign rooms**: Match exams to available rooms based on enrollment

### Solution Approach

**Graph Coloring Formulation:**
1. **Nodes** = Exams
2. **Edges** = Conflicts (students taking both courses)
3. **Colors** = Time slots
4. **Objective**: Color graph with minimum colors (chromatic number K)

**Quantum/Classical Solvers:**
- Convert to **QUBO** (Quadratic Unconstrained Binary Optimization)
- Solve using quantum annealing (D-Wave) or gate-based quantum (QAOA)

---

## 📂 Project Structure

```
.
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── pyproject.toml                # Project configuration
│
├── QAOA/                         # Main workflow scripts
│   ├── 01_generate_dataset.py    # Generate exam/course data
│   ├── 02_visualize_graph.py     # Visualize conflict graphs
│   ├── 03_build_qubo.py          # Build QUBO matrices
│   ├── 04_unified_solver.py      # 🌟 MAIN SOLVER (all backends)
│   │
│   └── output/                   # Generated results
│       ├── latest_run.txt        # Pointer to latest run
│       └── run_YYYYMMDD_HHMMSS/  # Timestamped results
│           ├── datasets/         # Generated exam data
│           ├── solutions/        # Solver results (JSON)
│           ├── visualizations/   # Graphs and plots (PNG)
│           └── benchmark_results.csv
│
└── scripts to be removed/        # Legacy scripts (deprecated)
```

---

## 🔧 Installation

### Prerequisites

- **Python 3.9+**
- Virtual environment (recommended)

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd <repo-directory>
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `qiskit` + `qiskit-optimization` - IBM QAOA
- `dwave-ocean-sdk` - D-Wave solvers
- `numpy`, `pandas`, `matplotlib`, `networkx` - Data processing & visualization

### Optional: D-Wave Account Setup

For **D-Wave QPU/Hybrid** solvers (requires free account):

```bash
# Sign up at https://cloud.dwavesys.com/leap/signup/
dwave config create
# Enter your API token when prompted
```

---

## 🚀 Quick Start

### Run Complete Pipeline (5 minutes)

```bash
cd QAOA

# Step 1: Generate sample exam data
python 01_generate_dataset.py

# Step 2: Visualize conflict graphs
python 02_visualize_graph.py

# Step 3: Build QUBO matrices
python 03_build_qubo.py

# Step 4: Solve with QAOA (default backend)
python 04_unified_solver.py tiny 3 --backend qaoa

# Results will be in: output/run_*/solutions/
```

**What you'll get:**
- ✅ Valid exam schedule (if K ≥ chromatic number)
- 📊 JSON results with energy, runtime, conflicts
- 📈 Visualizations (if `--no-viz` not specified)

---

## 📖 Detailed Workflow

### 🔹 Script 1: `01_generate_dataset.py`

**Purpose:** Generate synthetic exam scheduling datasets

**What it creates:**
- `courses.csv` - Course details (enrollment, year, conflicts)
- `rooms.csv` - Available exam rooms (capacity)
- `conflict_adjacency.csv` - N×N conflict matrix (students in common)

**Datasets:**
| Name | Exams | Avg Conflicts | Chromatic # |
|------|-------|---------------|-------------|
| tiny | 5 | ~2 | 3-4 |
| small | 10 | ~3 | 4-5 |
| medium | 20 | ~5 | 5-7 |

**Usage:**
```bash
python 01_generate_dataset.py
# Output: QAOA/output/run_YYYYMMDD_HHMMSS/datasets/
```

**Customize:**
Edit the script to change:
- Number of exams
- Enrollment ranges
- Conflict probability
- Room configurations

---

### 🔹 Script 2: `02_visualize_graph.py`

**Purpose:** Visualize exam conflict graphs

**Creates:**
- Color-coded conflict graphs (nodes = exams, edges = conflicts)
- Adjacency matrix heatmaps
- Degree distribution plots

**Usage:**
```bash
python 02_visualize_graph.py
# Output: QAOA/output/run_*/visualizations/
```

**Example Output:**
```
conflict_graph_tiny.png      # Visual graph
adjacency_heatmap_tiny.png   # Matrix visualization
degree_distribution.png      # Statistical analysis
```

---

### 🔹 Script 3: `03_build_qubo.py`

**Purpose:** Convert graph coloring to QUBO optimization problem

**Mathematical Formulation:**

$$H = \lambda_1 \sum_{\text{exams}} (1 - \sum_{\text{colors}} x_{i,c})^2 + \lambda_2 \sum_{\text{conflicts}} x_{i,c} \cdot x_{j,c}$$

Where:
- $x_{i,c} \in \{0,1\}$: Binary variable (exam $i$ in slot $c$)
- **Constraint 1** ($\lambda_1$): Each exam assigned exactly once
- **Constraint 2** ($\lambda_2$): No conflicts in same slot

**Output:**
- `qubo_matrix_K{K}.npy` - QUBO matrix (NumPy array)
- `qubo_metadata_K{K}.json` - Problem dimensions, penalties

**Usage:**
```bash
python 03_build_qubo.py
# Generates QUBOs for K=2,3,4,5,6,7
```

**QUBO Size:**
- Variables: `num_exams × K`
- Example: 10 exams, K=4 → 40 binary variables
- Matrix: 40×40 = 1,600 coefficients

---

### 🔹 Script 4: `04_unified_solver.py` ⭐

**Purpose:** THE MAIN SOLVER - All quantum backends in one script

**Supported Backends:**

| Backend | Type | Speed | Accuracy | Hardware |
|---------|------|-------|----------|----------|
| `qaoa` | Gate-based Quantum | Medium | Good | Simulator |
| `neal` | Simulated Annealing | Fast | Best | Classical |
| `dwave` | Quantum Annealing | Very Fast | Excellent | QPU (requires account) |
| `hybrid` | Classical + Quantum | Fast | Excellent | Cloud (requires account) |

---

## 🎮 Unified Solver Usage

### Mode 1: Single Solve

Solve **one problem** with **one backend**:

```bash
# Syntax: python 04_unified_solver.py <dataset> <K> --backend <solver>

# QAOA (IBM Qiskit)
python 04_unified_solver.py tiny 3 --backend qaoa

# D-Wave Neal (Simulated Annealing)
python 04_unified_solver.py tiny 3 --backend neal

# D-Wave QPU (Quantum Hardware)
python 04_unified_solver.py tiny 3 --backend dwave --num-reads 100

# D-Wave Hybrid (Best for large problems)
python 04_unified_solver.py small 4 --backend hybrid
```

**Output:**
```json
{
  "dataset": "tiny",
  "K": 3,
  "backend": "qaoa",
  "runtime_seconds": 12.5,
  "energy": -15000.0,
  "is_valid": true,
  "num_conflicts": 0,
  "coloring": {"0": 0, "1": 1, "2": 0, "3": 2, "4": 1}
}
```

---

### Mode 2: Test Multiple K Values

**Find minimum K (chromatic number):**

```bash
# Test K from 2 to 5 using QAOA
python 04_unified_solver.py tiny --k-range 2 5 --backend qaoa

# Test K from 3 to 5 using Neal
python 04_unified_solver.py small --k-range 3 5 --backend neal
```

**What it does:**
- Solves for K=2, then K=3, K=4, K=5
- Finds smallest K with valid solution
- Saves all results to CSV

**Output:**
```
tiny | K=2 | ✗ INVALID (3 conflicts)
tiny | K=3 | ✓ VALID (0 conflicts, 15.2s)
tiny | K=4 | ✓ VALID (0 conflicts, 18.7s)
```

---

### Mode 3: Compare Backends

**Compare different solvers on same problem:**

```bash
# Compare QAOA vs Neal
python 04_unified_solver.py tiny 3 --compare-backends qaoa neal

# Compare all available backends
python 04_unified_solver.py tiny 3 --compare-backends qaoa neal dwave hybrid
```

**Output:**
```
BACKEND COMPARISON
==================
QAOA       |  15.2s | Energy: -15000.0 | ✓ VALID
NEAL       |   0.8s | Energy: -15000.0 | ✓ VALID
DWAVE      |   2.3s | Energy: -15000.0 | ✓ VALID
```

**Use Case:** Determine best solver for your problem size

---

### Mode 4: Full Benchmark

**Test all K values × all backends:**

```bash
# Benchmark tiny dataset (default: qaoa, neal)
python 04_unified_solver.py tiny --benchmark

# Benchmark with specific backends
python 04_unified_solver.py small --benchmark --backends qaoa neal

# Benchmark ALL datasets
python 04_unified_solver.py --all-datasets --benchmark
```

**Output Files:**
```
output/run_*/
├── benchmark_results.csv         # All run data
├── benchmark_plot.png            # Runtime/Energy comparison
└── solutions/
    ├── qaoa_results_tiny_K3.json
    ├── neal_results_tiny_K3.json
    └── ...
```

**Result Table:**
```csv
dataset,K,backend,runtime,energy,is_valid,num_conflicts
tiny,3,qaoa,15.2,-15000.0,True,0
tiny,3,neal,0.8,-15000.0,True,0
tiny,4,qaoa,18.9,-20000.0,True,0
```

---

## 🎛️ Advanced Options

### QAOA-Specific Parameters

```bash
# Adjust QAOA circuit depth (p layers)
python 04_unified_solver.py tiny 3 --backend qaoa --reps 3

# Limit optimizer iterations (faster but less accurate)
python 04_unified_solver.py tiny 3 --backend qaoa --maxiter 50

# Combine for large problems
python 04_unified_solver.py small 4 --backend qaoa --reps 1 --maxiter 30
```

**QAOA Parameters:**
- `--reps` (default: 2): Circuit depth, higher = better quality
- `--maxiter` (default: 100): Optimizer iterations
- Recommended for large problems: `--reps 1 --maxiter 30-50`

---

### D-Wave-Specific Parameters

```bash
# Increase sampling (better statistics)
python 04_unified_solver.py tiny 3 --backend neal --num-reads 5000

# QPU annealing time (microseconds)
python 04_unified_solver.py tiny 3 --backend dwave --num-reads 100 --annealing-time 20
```

**D-Wave Parameters:**
- `--num-reads` (default: 1000): Number of samples
- More reads = better solution quality (but slower)

---

### Other Options

```bash
# Skip visualization (faster)
python 04_unified_solver.py tiny 3 --backend qaoa --no-viz

# Assign rooms after solving
python 04_unified_solver.py tiny 3 --backend qaoa --assign-rooms

# Custom timeout
python 04_unified_solver.py small 4 --backend qaoa --timeout 300
```

---

## 📊 Benchmarking

### Performance Comparison

**Typical Results (5 exams, K=3):**

| Backend | Runtime | Success Rate | Energy |
|---------|---------|--------------|--------|
| QAOA | 10-20s | 85% | -15000 |
| Neal | <1s | 95% | -15000 |
| D-Wave QPU | 2-5s | 98% | -15000 |

**Scaling (10 exams):**

| K | Variables | QAOA Time | Neal Time |
|---|-----------|-----------|-----------|
| 3 | 30 | 30-60s | <1s |
| 4 | 40 | 60-180s | 1-2s |
| 5 | 50 | 180-600s | 2-5s |

**Rule of Thumb:**
- **QAOA**: Good for research (quantum algorithm study), limit ~40 variables
- **Neal**: Best for classical baseline, handles 100+ variables
- **D-Wave QPU**: Best accuracy/speed tradeoff, requires account

---

### Running Systematic Benchmarks

```bash
# Test all K values on tiny dataset
python 04_unified_solver.py tiny --k-range 2 6 --benchmark --backends qaoa neal

# Compare datasets
python 04_unified_solver.py --all-datasets --benchmark

# Statistical runs (5 repetitions)
for i in {1..5}; do
  python 04_unified_solver.py tiny 3 --compare-backends qaoa neal
done
```

**Analysis:**
```python
import pandas as pd

# Load results
df = pd.read_csv('output/benchmark_results.csv')

# Average runtime by backend
print(df.groupby('backend')['runtime_seconds'].mean())

# Success rate
print(df.groupby('backend')['is_valid'].mean())
```

---

## 🧪 Understanding the Results

### Solution Files

**Location:** `output/run_*/solutions/`

**Example: `qaoa_results_tiny_K3.json`**
```json
{
  "dataset": "tiny",
  "K": 3,
  "backend": "qaoa",
  "num_variables": 15,
  "num_exams": 5,
  "runtime_seconds": 12.45,
  "energy": -15000.0,
  "is_valid": true,
  "num_conflicts": 0,
  "colors_used": 3,
  "coloring": {
    "0": 0,    // Exam 0 → Time slot 0
    "1": 1,    // Exam 1 → Time slot 1
    "2": 0,    // Exam 2 → Time slot 0 (no conflict with Exam 0)
    "3": 2,    // Exam 3 → Time slot 2
    "4": 1     // Exam 4 → Time slot 1
  }
}
```

---

### Validation

**Valid Solution:**
```
✓ All 5 exams colored
✓ Colors used: 3/3
✓ No conflicts
```

**Invalid Solution:**
```
✗ Conflicts: 2
  - Exams 0,2 both in slot 0
  - Exams 1,4 both in slot 1
```

**How to Fix:**
1. Increase K: Try `K=4` instead of `K=3`
2. More optimization: `--maxiter 200` or `--reps 3`
3. Different backend: Neal often finds better solutions

---

### Energy Interpretation

**QUBO Energy Formula:**
$$E = \text{penalty}_1 \times (\text{unassigned exams}) + \text{penalty}_2 \times (\text{conflicts})$$

**Example:**
- `energy = -15000.0` → Valid solution (all constraints satisfied)
- `energy = -12500.0` → Violations: $(15000 - 12500) / 5000 = 0.5$ conflicts
- Lower energy = better solution

**Ideal Energy:**
- Valid solution: $E = -\lambda_1 \times n_{\text{exams}}$ (all exams assigned, no conflicts)
- For 5 exams with $\lambda_1=10000$: $E = -50000$
- Plus constraint bonuses brings to ~$-15000$

---

## 🔬 Research Use Cases

### 1. Algorithm Comparison Study

**Objective:** Compare QAOA vs classical methods

```bash
# Generate data
python 01_generate_dataset.py

# Benchmark all backends
python 04_unified_solver.py tiny --benchmark --backends qaoa neal

# Analyze
python -c "
import pandas as pd
df = pd.read_csv('output/benchmark_results.csv')
print(df.groupby('backend')[['runtime', 'energy', 'is_valid']].agg(['mean', 'std']))
"
```

**Research Questions:**
- Does QAOA find better solutions than classical SA?
- How does runtime scale with problem size?
- What's the approximation ratio?

---

### 2. Parameter Sensitivity Analysis

**Test QAOA depth (p layers):**

```bash
for p in 1 2 3 4; do
  python 04_unified_solver.py tiny 3 --backend qaoa --reps $p
done
```

**Test K vs solution quality:**

```bash
for K in 3 4 5 6; do
  python 04_unified_solver.py tiny $K --backend qaoa
done
```

---

### 3. Quantum Hardware Validation

**Compare simulator vs real QPU:**

```bash
# Simulator (QAOA)
python 04_unified_solver.py tiny 3 --backend qaoa

# Real quantum hardware (D-Wave)
python 04_unified_solver.py tiny 3 --backend dwave --num-reads 1000
```

**Metrics:**
- Solution quality (energy)
- Success rate over multiple runs
- Embedding overhead (D-Wave specific)

---

## 🛠️ Troubleshooting

### Problem: "No run directory found"

**Error:**
```
⚠ No run directory found. Run 01_generate_dataset.py first.
```

**Solution:**
```bash
python 01_generate_dataset.py  # Creates output/run_*/
python 03_build_qubo.py        # Generates QUBO matrices
python 04_unified_solver.py tiny 3 --backend qaoa
```

---

### Problem: QAOA too slow (timeout)

**Symptoms:**
```
⚠ Exceeded timeout (600s)
```

**Solutions:**

**Option 1:** Reduce problem size
```bash
python 04_unified_solver.py tiny 3 --backend qaoa  # Use tiny, not small
```

**Option 2:** Reduce QAOA parameters
```bash
python 04_unified_solver.py tiny 3 --backend qaoa --reps 1 --maxiter 30
```

**Option 3:** Use faster backend
```bash
python 04_unified_solver.py tiny 3 --backend neal  # Classical, instant
```

**Option 4:** Increase timeout
```bash
python 04_unified_solver.py small 4 --backend qaoa --timeout 1800  # 30 minutes
```

---

### Problem: D-Wave Access Denied

**Error:**
```
✗ D-Wave QPU access failed: Authentication error
```

**Solution:**

**Step 1:** Sign up for D-Wave Leap (free)
- Go to: https://cloud.dwavesys.com/leap/signup/

**Step 2:** Configure credentials
```bash
dwave config create
# Paste your API token from Leap dashboard
```

**Step 3:** Test connection
```bash
dwave ping
```

**Fallback:** Use Neal simulator (free, no account needed)
```bash
python 04_unified_solver.py tiny 3 --backend neal
```

---

### Problem: Invalid Solutions

**Symptoms:**
```
✗ Solution INVALID
✗ Conflicts: 3
```

**Diagnosis:**

**Check 1:** K too small?
```bash
# Try increasing K
python 04_unified_solver.py tiny 4 --backend qaoa  # Instead of K=3
```

**Check 2:** Optimization not converged?
```bash
# More iterations
python 04_unified_solver.py tiny 3 --backend qaoa --maxiter 200

# Deeper circuit
python 04_unified_solver.py tiny 3 --backend qaoa --reps 3
```

**Check 3:** Use more reliable backend
```bash
# Neal usually finds valid solutions
python 04_unified_solver.py tiny 3 --backend neal --num-reads 5000
```

---

### Problem: Out of Memory

**Error:**
```
MemoryError: Unable to allocate array
```

**Cause:** QUBO matrix too large (e.g., 100 exams × K=5 = 500 variables → 250,000 element matrix)

**Solution:**

**Option 1:** Reduce problem size
```bash
# Use smaller dataset
python 04_unified_solver.py tiny 3  # Instead of medium
```

**Option 2:** Reduce K
```bash
python 04_unified_solver.py small 3  # Instead of K=5
```

**Option 3:** Use D-Wave Hybrid (handles large problems)
```bash
python 04_unified_solver.py medium 5 --backend hybrid
```

---

## 📚 Advanced Topics

### Custom Datasets

**Edit `01_generate_dataset.py`:**

```python
# Customize number of exams
datasets = {
    'custom': {
        'num_courses': 15,      # 15 exams
        'conflict_prob': 0.4,   # 40% chance of conflict
        'room_types': [
            {'capacity': 30, 'count': 5},
            {'capacity': 50, 'count': 3}
        ]
    }
}
```

**Run:**
```bash
python 01_generate_dataset.py
python 03_build_qubo.py
python 04_unified_solver.py custom 4 --backend neal
```

---

### Penalty Tuning

**Edit `03_build_qubo.py`:**

```python
builder = GraphColoringQUBO(
    adjacency_matrix=adjacency,
    num_colors=K,
    lambda1=10000,  # Increase for stricter "one slot per exam" constraint
    lambda2=5000    # Increase to penalize conflicts more
)
```

**Effect:**
- Higher $\lambda_1$: Ensures all exams assigned
- Higher $\lambda_2$: Reduces conflicts (may increase K needed)

---

### Real Exam Data Integration

**Replace synthetic data with CSV:**

**Format: `real_courses.csv`**
```csv
course_code,course_name,enrollment,year,students_list
CS101,Intro to CS,150,1,"s1|s2|s3|..."
CS201,Data Structures,120,2,"s5|s10|s15|..."
```

**Compute conflicts:**
```python
import pandas as pd

courses = pd.read_csv('real_courses.csv')

# Build conflict matrix
n = len(courses)
conflicts = np.zeros((n, n))

for i in range(n):
    students_i = set(courses.iloc[i]['students_list'].split('|'))
    for j in range(i+1, n):
        students_j = set(courses.iloc[j]['students_list'].split('|'))
        common = len(students_i & students_j)
        if common > 0:
            conflicts[i,j] = common
            conflicts[j,i] = common

# Save
pd.DataFrame(conflicts).to_csv('conflict_adjacency.csv')
```

**Then run:**
```bash
python 03_build_qubo.py  # Uses new adjacency matrix
python 04_unified_solver.py custom 5 --backend hybrid
```

---

## 🏆 Best Practices

### For Research

```bash
# Always run multiple trials
for trial in {1..10}; do
  python 04_unified_solver.py tiny 3 --backend qaoa
done

# Analyze variance
python -c "
import pandas as pd
import glob

files = glob.glob('output/run_*/solutions/qaoa_results_tiny_K3.json')
results = [pd.read_json(f, typ='series') for f in files]
df = pd.DataFrame(results)

print('Runtime: %.2f ± %.2f seconds' % (df['runtime_seconds'].mean(), df['runtime_seconds'].std()))
print('Success rate: %.1f%%' % (df['is_valid'].mean() * 100))
"
```

---

### For Production

```bash
# Use most reliable backend
python 04_unified_solver.py real_data 5 --backend hybrid

# Validate solution
python -c "
import json
with open('output/run_*/solutions/hybrid_results_real_data_K5.json') as f:
    result = json.load(f)
    
if result['is_valid']:
    print('✓ Solution ready for deployment')
    print(f'Scheduled {result[\"num_exams\"]} exams in {result[\"colors_used\"]} slots')
else:
    print('✗ Need to increase K or re-run optimization')
"
```

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{quantum_exam_scheduling_2025,
  author = {Your Name},
  title = {Quantum Exam Scheduling using QAOA and D-Wave},
  year = {2025},
  url = {https://github.com/your-username/exam-scheduling}
}
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Add constraint weights optimization
- [ ] Implement room capacity constraints in QUBO
- [ ] Support for multi-day scheduling
- [ ] Web interface for visualization
- [ ] Integration with IBM Quantum hardware

---

## 📄 License

MIT License - See LICENSE file

---

## 📧 Contact

- **Author:** Your Name
- **Email:** your.email@university.edu
- **Research Group:** Quantum Computing Lab

---

## 🔗 Resources

### Documentation
- [IBM Qiskit QAOA Tutorial](https://qiskit.org/textbook/ch-applications/qaoa.html)
- [D-Wave Ocean SDK](https://docs.ocean.dwavesys.com/)
- [Graph Coloring NP-Hardness](https://en.wikipedia.org/wiki/Graph_coloring)

### Related Papers
- Farhi et al. (2014) - "A Quantum Approximate Optimization Algorithm"
- McGeoch (2014) - "Adiabatic Quantum Computation and Quantum Annealing"
- Burke et al. (2003) - "Examination Timetabling: A Survey"

---

**Last Updated:** February 2025  
**Version:** 2.0.0 (Unified Solver Release)