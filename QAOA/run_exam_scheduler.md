# Quantum Exam Scheduling - Simplified Interactive Version

A streamlined tool for solving exam timetabling using quantum and classical optimization.

---

## 🚀 Quick Start

### Installation

```bash
# Core dependencies
pip install qiskit qiskit-optimization dwave-ocean-sdk numpy pandas

# Optional (for visualization)
pip install matplotlib networkx
```

### Run Interactive Mode

```bash
python run_exam_scheduler.py
```

**You'll be prompted:**
```
Number of courses/exams (e.g., 8, 10, 15): 10
Number of students (e.g., 40, 60, 80): 50
Number of time slots/colors K (e.g., 3, 4, 5): 4
```

The script will:
1. ✅ Generate dataset with random uniform conflicts (40% density by default)
2. ✅ Build QUBO matrix
3. ✅ Solve with Neal (default backend)
4. ✅ Validate solutions
5. ✅ Generate timetables (if valid)

---

## 📋 Command-Line Usage

### Basic Examples

```bash
# Quick start (default: 50 students, 40% conflicts, Neal solver)
python run_exam_scheduler.py --courses 10 --k 4

# Specify number of students
python run_exam_scheduler.py --courses 10 --students 80 --k 4

# With visualization
python run_exam_scheduler.py --courses 10 --k 4 --visualize

# Using QAOA instead of Neal
python run_exam_scheduler.py --courses 8 --k 3 --backend qaoa

# Compare both backends
python run_exam_scheduler.py --courses 10 --k 4 --backend both
```

### Conflict Density Control

Conflicts are generated using **random uniform distribution**. Use `--conflict-pct` to control edge density (0-100%).

```bash
# Low conflict density (20%)
python run_exam_scheduler.py --courses 10 --k 3 --conflict-pct 20

# Medium conflict density (40%, default)
python run_exam_scheduler.py --courses 10 --k 4 --conflict-pct 40

# High conflict density (70%)
python run_exam_scheduler.py --courses 10 --k 6 --conflict-pct 70

# Very sparse graph (10%)
python run_exam_scheduler.py --courses 15 --k 3 --conflict-pct 10 --visualize

# Very dense graph (80%)
python run_exam_scheduler.py --courses 10 --k 8 --conflict-pct 80 --visualize
```

### Visualization Examples

```bash
# Basic visualization (adjacency heatmap + conflict graph + timetable)
python run_exam_scheduler.py --courses 10 --k 4 --visualize

# Sparse graph visualization
python run_exam_scheduler.py --courses 10 --k 3 --conflict-pct 20 --visualize

# Dense graph visualization
python run_exam_scheduler.py --courses 15 --k 6 --conflict-pct 60 --visualize
```

---

## 🎛️ All Parameters

```bash
python run_exam_scheduler.py \
  --courses 10 \              # Number of exams
  --students 50 \             # Number of students (default: 50, for metadata)
  --k 4 \                     # Number of time slots
  --avg-courses 4 \           # Avg courses per student (default: 4, for enrollment metadata)
  --conflict-pct 40.0 \       # Conflict percentage 0-100 (default: 40.0, controls edge density)
  --backend neal \            # qaoa | neal | both (default: neal)
  --reps 2 \                  # QAOA circuit depth (default: 2)
  --maxiter 100 \             # QAOA optimizer iterations (default: 100)
  --num-reads 1000 \          # Neal sampling count (default: 1000)
  --visualize                 # Generate visualization plots
```

**Parameters Explained:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--courses` | Interactive | Number of courses/exams to schedule |
| `--students` | 50 | Number of students (used for enrollment metadata, not for conflicts) |
| `--k` | Interactive | Number of time slots (colors) |
| `--avg-courses` | 4 | Average courses per student (for enrollment metadata only) |
| `--conflict-pct` | 40.0 | Conflict percentage 0-100 (controls edge density in random conflict graph) |
| `--backend` | neal | Solver: `qaoa`, `neal`, or `both` |
| `--reps` | 2 | QAOA circuit depth (higher = better quality, slower) |
| `--maxiter` | 100 | QAOA optimizer iterations |
| `--num-reads` | 1000 | Neal sampling count (higher = better quality) |
| `--visualize` | - | Generate PNG visualizations (requires matplotlib, networkx) |

---

## 🎨 Conflict Generation

### **Random Uniform Distribution**

Conflicts are randomly distributed uniformly across all course pairs. The `--conflict-pct` parameter controls the exact percentage of edges in the conflict graph.

```bash
python run_exam_scheduler.py --courses 10 --k 4 --conflict-pct 40
```

**Characteristics:**
- ✅ Controlled density (exact percentage of edges)
- ✅ Uniform distribution (all courses have similar degree)
- 📊 Density directly set by `--conflict-pct`
- 🎯 Reproducible for benchmarking

**Example:**
- 10 courses, 40% conflict density
- Edges: 18 out of 45 possible (exactly 40%)
- Max degree: ~4 (uniform)
- Students/enrollments generated for metadata only

**When to use:**
- 🔬 Research: Testing algorithm performance with controlled inputs
- 📈 Benchmarking: Comparing solvers at different densities
- 🎯 Edge cases: Creating specific graph structures
- 🛠️ Debugging: Reproducible test cases

**Density Guidelines:**
- **Low (10-30%)**: Easy to color, K ≈ 3-4 slots
- **Medium (30-60%)**: Moderate difficulty, K ≈ 4-6 slots
- **High (60-90%)**: Hard to color, K ≈ 6-10 slots

---

## 📂 Output

After running, you'll find:

```
output/run_YYYYMMDD_HHMMSS/
├── courses.csv                # Generated courses
├── students.csv               # Generated students
├── enrollments.csv            # Student enrollments (creates conflicts)
├── conflict_adjacency.csv     # Conflict graph (N×N matrix)
├── metadata.json              # Dataset statistics
├── qubo_matrix.npy            # QUBO matrix
├── qaoa_results.json          # QAOA solution + metrics (if backend=qaoa/both)
├── neal_results.json          # Neal solution + metrics (if backend=neal/both)
├── timetable_qaoa.csv         # Timetable (if valid, qaoa)
├── timetable_neal.csv         # Timetable (if valid, neal)
└── (if --visualize)
    ├── adjacency_heatmap.png           # Conflict matrix heatmap
    ├── conflict_graph.png              # Network graph visualization
    └── timetable_visualization.png     # Solution visualization
```

**Example `timetable_neal.csv`:**
```csv
time_slot,exam_id,course_code,year,enrollment
0,0,C01,2,45
0,3,C04,3,32
1,1,C02,2,50
1,5,C06,3,28
2,2,C03,2,41
```

**Visualization Files:**

1. **adjacency_heatmap.png**: Red/yellow/green heatmap showing which courses conflict
2. **conflict_graph.png**: Network diagram with nodes=courses, edges=conflicts
3. **timetable_visualization.png**: Two-panel visualization showing:
   - Left: Conflict graph with nodes colored by assigned time slot
   - Right: Bar chart of exams per time slot

**View visualizations:**
```bash
# macOS
open output/run_YYYYMMDD_HHMMSS/*.png

# Linux
xdg-open output/run_YYYYMMDD_HHMMSS/timetable_visualization.png

# Windows
start output/run_YYYYMMDD_HHMMSS/*.png
```

---

## 🎯 Example Workflows

### 1. Quick Test (Small Problem)

```bash
python run_exam_scheduler.py --courses 5 --k 3
```

**Expected:**
- Runtime: < 1 second
- Valid solution: likely ✓
- Conflicts: 0
- Output: dataset + qubo + results + timetable

---

### 2. Medium Problem with Visualization

```bash
python run_exam_scheduler.py --courses 10 --k 4 --visualize
```

**Expected:**
- Neal runtime: < 2s
- Output: CSV files + 3 PNG visualizations
- View conflict structure and solution quality

---

### 3. Controlled Conflict Experiments

```bash
# Test different conflict densities
for pct in 20 40 60 80; do
  python run_exam_scheduler.py --courses 10 --k 5 --conflict-pct $pct --visualize
done

# Compare sparse vs dense graphs
python run_exam_scheduler.py --courses 10 --k 3 --conflict-pct 15 --visualize
python run_exam_scheduler.py --courses 10 --k 7 --conflict-pct 85 --visualize
```

**Analysis:**
```bash
# Check conflict densities
grep "density" output/run_*/metadata.json

# View all visualizations
open output/run_*/conflict_graph.png
```

---

### 4. Find Minimum K (Binary Search)

```bash
# Try K=3
python run_exam_scheduler.py --courses 10 --k 3

# Check output: "Valid: ✗ NO, Conflicts: 5"
# If invalid (conflicts > 0), try K=4
python run_exam_scheduler.py --courses 10 --k 4

# Check output: "Valid: ✗ NO, Conflicts: 2"
# If still invalid, try K=5
python run_exam_scheduler.py --courses 10 --k 5

# Check output: "Valid: ✓ YES, Conflicts: 0"
# Minimum K found: K=5 (chromatic number)
```

**Automated:**
```bash
for k in {3..8}; do
  echo "Testing K=$k..."
  python run_exam_scheduler.py --courses 10 --k $k | grep "Valid:"
done
```

---

### 5. Test QAOA vs Neal Performance

```bash
# Small problem (QAOA feasible)
python run_exam_scheduler.py --courses 8 --k 4 --backend both

# Medium problem (QAOA slow)
python run_exam_scheduler.py --courses 12 --k 4 --backend both

# Large problem (QAOA may timeout, Neal fast)
python run_exam_scheduler.py --courses 15 --k 5 --backend both

# Output shows comparison:
# QAOA     |  45.23s | ✓ VALID     | Conflicts: 0
# NEAL     |   1.87s | ✓ VALID     | Conflicts: 0
```

---

### 6. Visualize Different Graph Structures

```bash
# Sparse graph (easy to color)
python run_exam_scheduler.py --courses 10 --k 3 --conflict-pct 15 --visualize

# Dense graph (hard to color)
python run_exam_scheduler.py --courses 10 --k 7 --conflict-pct 85 --visualize

# Compare visualizations side-by-side
open output/run_*/conflict_graph.png
```

---

### 7. Batch Testing for Research

```bash
# Create results directory
mkdir -p experiment_results

# Test 10 random instances
for i in {1..10}; do
  echo "Run $i..."
  python run_exam_scheduler.py \
    --courses 10 \
    --k 4 \
    --backend both \
    --visualize > experiment_results/run_$i.log
done

# Extract results
grep "Valid:" experiment_results/*.log
grep "Runtime:" experiment_results/*.log

# Statistical analysis
python -c "
import json
from pathlib import Path
results = []
for f in Path('output').glob('*/neal_results.json'):
    with open(f) as fp:
        results.append(json.load(fp))
print(f'Average runtime: {sum(r[\"runtime\"] for r in results)/len(results):.2f}s')
print(f'Valid solutions: {sum(1 for r in results if r[\"energy\"] == -100000)}/{len(results)}')
"
```

---

## 🛠️ Troubleshooting

### Problem: "Solution invalid, conflicts > 0"

**Cause:** K is too small (not enough time slots)

**Solution:**
```bash
# Increase K
python run_exam_scheduler.py --courses 10 --k 5  # Instead of k=4
```

---

### Problem: "No backends available"

**Cause:** Neither Qiskit nor D-Wave installed

**Solution:**
```bash
pip install qiskit qiskit-optimization dwave-ocean-sdk
```

---

### Problem: QAOA too slow

**Solution 1:** Use Neal instead (default)
```bash
python run_exam_scheduler.py --courses 10 --k 4
```

**Solution 2:** Reduce QAOA parameters
```bash
python run_exam_scheduler.py --courses 10 --k 4 --backend qaoa --reps 1 --maxiter 30
```

---

## 📚 How It Works

1. **Dataset Generation:**
   - Creates courses, students, enrollments
   - Builds conflict graph (courses that share students)

2. **QUBO Formulation:**
   - Variables: `x[exam, time_slot]` (binary)
   - Constraint 1: Each exam in exactly one slot
   - Constraint 2: Conflicting exams in different slots

3. **Quantum Solving:**
   - **QAOA**: Gate-based quantum algorithm (simulator)
   - **Neal**: Classical simulated annealing (fast baseline)

4. **Validation:**
   - Checks all exams assigned
   - Counts conflicts (should be 0 for valid solution)

5. **Timetable:**
   - Human-readable schedule
   - Groups exams by time slot

---

## 🔬 Research Questions

### Q1: Does QAOA find better solutions than Neal?

```bash
# Run 10 times with same parameters
for i in {1..10}; do
  python run_exam_scheduler.py --courses 10 --k 4 --backend both
done

# Analyze outputs in output/run_*/
python -c "
import json
from pathlib import Path

qaoa_valid = []
neal_valid = []

for run_dir in sorted(Path('output').glob('run_*'))[-10:]:
    qaoa_file = run_dir / 'qaoa_results.json'
    neal_file = run_dir / 'neal_results.json'
    
    if qaoa_file.exists():
        with open(qaoa_file) as f:
            qaoa = json.load(f)
            qaoa_valid.append(1 if qaoa.get('energy', 0) == -100000 else 0)
    
    if neal_file.exists():
        with open(neal_file) as f:
            neal = json.load(f)
            neal_valid.append(1 if neal.get('energy', 0) == -100000 else 0)

print(f'QAOA: {sum(qaoa_valid)}/10 valid solutions')
print(f'Neal: {sum(neal_valid)}/10 valid solutions')
"
```

---

### Q2: How does K affect solution quality?

```bash
# Test K from 3 to 7
for k in {3..7}; do
  echo "Testing K=$k..."
  python run_exam_scheduler.py --courses 10 --k $k --backend both | tee k${k}_results.txt
done

# Extract results
grep -A 2 "QAOA" k*_results.txt
grep -A 2 "NEAL" k*_results.txt
```

**Expected findings:**
- K=3: Invalid (conflicts > 0)
- K=4: Invalid or borderline
- K=5: Valid (chromatic number)
- K=6+: Valid (overconstrained, easier)

---

### Q3: How does conflict density affect solver performance?

```bash
# Test different densities
for pct in 10 20 30 40 50 60 70 80 90; do
  echo "Testing ${pct}% conflict density..."
  python run_exam_scheduler.py \
    --courses 10 \
    --k 6 \
    --conflict-pct $pct \
    --backend both \
    --visualize
done

# Analyze runtimes
grep "Runtime:" output/run_*/neal_results.json
```

**Hypothesis:**
- Low density (< 30%): Fast, always valid
- Medium density (30-60%): Moderate, usually valid
- High density (> 70%): Slow, may need more K

---

### Q4: How does problem size affect runtime?

```bash
# Test increasing problem sizes
for n in 5 10 15 20 25; do
  echo "Testing $n courses..."
  python run_exam_scheduler.py --courses $n --k 6 --backend both
done

# Compare:
# - Max degree (enrollment typically higher)
# - Solution validity
# - Runtime
```

---

### Q5: Scalability - how large can we go?

```bash
# Small
python run_exam_scheduler.py --courses 5 --k 3 --conflict-mode random --conflict-pct 40 --backend both

# Medium
python run_exam_scheduler.py --courses 10 --k 4 --conflict-mode random --conflict-pct 40 --backend both

# Large
python run_exam_scheduler.py --courses 15 --k 5 --conflict-mode random --conflict-pct 40 --backend both

# Very large (QAOA will timeout, Neal should work)
python run_exam_scheduler.py --courses 20 --k 6 --conflict-mode random --conflict-pct 40 --backend neal
```

**Variables:**
- 5 courses × 3 slots = 15 variables (QAOA: fast)
- 10 courses × 4 slots = 40 variables (QAOA: feasible)
- 15 courses × 5 slots = 75 variables (QAOA: slow/timeout)
- 20 courses × 6 slots = 120 variables (QAOA: timeout, Neal: OK)

---

## 📖 Citation

```bibtex
@software{quantum_exam_scheduling_2025,
  title = {Quantum Exam Scheduling using QAOA and D-Wave},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo}
}
```

---

## 📚 Complete Command Reference

### All Command-Line Options

```bash
python run_exam_scheduler.py [OPTIONS]

Options:
  --courses INT              Number of courses/exams (required unless interactive)
  --students INT             Number of students (default: 50, for enrollment metadata)
  --k INT                    Number of time slots (required unless interactive)
  --avg-courses INT          Average courses per student (default: 4, for enrollment metadata)
  --conflict-pct FLOAT       Conflict percentage 0-100 (default: 40.0, controls edge density)
  --backend STR              Solver backend: qaoa | neal | both (default: neal)
  --reps INT                 QAOA circuit depth (default: 2)
  --maxiter INT              QAOA optimizer max iterations (default: 100)
  --num-reads INT            Neal number of reads (default: 1000)
  --visualize                Generate visualization PNG files (requires matplotlib, networkx)
  -h, --help                 Show help message
```

### Most Common Commands

```bash
# 1. Quick test (defaults: 50 students, 40% conflicts, Neal)
python run_exam_scheduler.py --courses 10 --k 4

# 2. With visualization
python run_exam_scheduler.py --courses 10 --k 4 --visualize

# 3. Custom conflict density
python run_exam_scheduler.py --courses 10 --k 4 --conflict-pct 30 --visualize

# 4. Compare backends
python run_exam_scheduler.py --courses 10 --k 4 --backend both

# 5. High conflict scenario (70% density)
python run_exam_scheduler.py --courses 10 --k 6 --conflict-pct 70 --visualize

# 6. Low conflict scenario (15% density)
python run_exam_scheduler.py --courses 10 --k 3 --conflict-pct 15

# 7. Interactive mode (prompts for all inputs)
python run_exam_scheduler.py

# 8. Large problem (Neal only)
python run_exam_scheduler.py --courses 20 --k 6 --conflict-pct 30

# 9. QAOA performance tuning
python run_exam_scheduler.py --courses 8 --k 4 --backend qaoa --reps 3 --maxiter 200

# 10. Batch processing different densities
for i in {1..5}; do
  python run_exam_scheduler.py --courses 10 --k 4 --conflict-pct $((i*20)) --visualize
done
```

### Common Pitfall Solutions

```bash
# Problem: K too small, solution invalid
# Bad:  python run_exam_scheduler.py --courses 10 --k 3
# Good: python run_exam_scheduler.py --courses 10 --k 5

# Problem: QAOA too slow (>5 minutes)
# Bad:  python run_exam_scheduler.py --courses 15 --k 5 --backend qaoa
# Good: python run_exam_scheduler.py --courses 15 --k 5  # Uses Neal by default

# Problem: Want different conflict density
# Bad:  python run_exam_scheduler.py --courses 10 --k 4  # Uses default 40%
# Good: python run_exam_scheduler.py --courses 10 --k 4 --conflict-pct 25

# Problem: Forgot to visualize
# Bad:  python run_exam_scheduler.py --courses 10 --k 4
# Good: python run_exam_scheduler.py --courses 10 --k 4 --visualize
```

---

**Last Updated:** February 17, 2026  
**Version:** 5.0.0 (Simplified with Random Conflict Mode Only)