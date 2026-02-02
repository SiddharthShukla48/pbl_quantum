# Exam Scheduling QUBO Solver - D-Wave Implementation

## Overview

This implementation solves the **Exam Timetable Scheduling Problem** using **QUBO (Quadratic Unconstrained Binary Optimization)** on **D-Wave quantum computers**.

### What We've Implemented

**Stage 1: Timeslot Allocation** - Assigning each exam to a (day, timeslot) combination

**Constraints:**
- **H1 (Hard)**: No student conflicts - students cannot have 2 exams at the same time
- **H2 (Hard)**: Exam duration - multi-hour exams must get consecutive slots
- **H3 (Hard)**: One exam one slot - each exam scheduled exactly once
- **S1 (Soft)**: Spread exams - students should have gaps between exams for revision

---

## Files in This Project

```
exam-scheduling-qubo/
├── dataset-generator.py       # Generates realistic exam scheduling datasets
├── data-loader.py             # Loads and validates datasets
├── qubo-builder-example.py    # Educational QUBO building examples
├── qubo_solver_dwave.py       # ⭐ Main D-Wave solver (THIS IS THE KEY FILE)
├── requirements.txt           # Python dependencies
└── README_SOLVER.md          # This file
```

---

## Installation

### Step 1: Install Python Packages

```bash
pip install -r requirements.txt
```

This installs:
- `dwave-ocean-sdk` - Complete D-Wave toolkit
- `numpy`, `pandas` - Data processing
- `matplotlib`, `seaborn` - Visualization (optional)

### Step 2: Set Up D-Wave Account

1. **Create free account**: https://cloud.dwavesys.com/leap/
   - Free tier: 1 minute QPU time/month
   - Includes simulator and hybrid solver access

2. **Get API token**:
   - Login → Dashboard → API Token
   - Copy your token

3. **Configure D-Wave**:
   ```bash
   dwave config create
   ```
   - Paste your API token when prompted
   - Accept default endpoints

4. **Verify setup**:
   ```bash
   dwave ping
   ```
   Should show: "Using endpoint: https://cloud.dwavesys.com/sapi/..."

---

## Quick Start

### 1. Generate Dataset

```bash
python dataset-generator.py
```

This creates test datasets:
- `exam_scheduling_TINY/` - 5 courses, 2 days (for testing)
- `exam_scheduling_SMALL/` - 10 courses, 3 days
- `exam_scheduling_MEDIUM/` - 20 courses, 5 days

### 2. Run D-Wave Solver

```bash
python qubo_solver_dwave.py
```

**Output:**
```
D-Wave QUBO Solver - Stage 1: Timeslot Allocation
==================================================================
Courses:          5
Days:             2
Slots per day:    2
Total variables:  20
QUBO matrix size: 20 × 20
Solver type:      NEAL
==================================================================

Building QUBO Matrix...
  [1/4] Building H1: No Student Conflicts...
      ✓ Added 32 conflict penalties
  [2/4] Building H2: Exam Duration Constraints...
      ✓ Added 6 duration constraints
  [3/4] Building H3: One Exam One Slot...
      ✓ Added one-slot constraints for 5 courses
  [4/4] Building S1: Spread Exams...
      ✓ Added 128 spread penalties

QUBO Matrix Statistics:
  Dimensions:      20 × 20
  Non-zero:        186
  Density:         46.50%

Solving QUBO with NEAL solver...
  ✓ Completed 100 samples
  Best energy: -49850.00

EXAM SCHEDULE
Day 0:
  Slot 0: CS101 (80 students, 2hr)
  Slot 1: (empty)
Day 1:
  Slot 0: CS102 (50 students, 1hr), MATH101 (60 students, 1hr)
  Slot 1: CS201 (40 students, 1hr)

Validation Results:
  H1: No conflicts         ✓ PASS
  H2: Duration             ✓ PASS
  H3: All scheduled        ✓ PASS
  OVERALL                  ✓ VALID

✓ SUCCESS: Found valid exam schedule!
```

---

## Detailed Code Explanation

### Architecture

The solver follows this workflow:

```
Load Data → Build QUBO → Create BQM → Solve → Interpret Solution → Validate
```

### Core Components

#### 1. **SolverConfig** (Lines 42-54)
Configuration dataclass:
```python
config = SolverConfig(
    solver_type='neal',     # 'neal', 'dwave', or 'hybrid'
    num_reads=100,          # Number of solutions to sample
    lambda_h1=10000,        # Penalty for student conflicts
    lambda_h2=10000,        # Penalty for duration violations
    lambda_h3=10000,        # Penalty for scheduling violations
    mu_s1=100               # Penalty for poor spreading (soft)
)
```

**Why these values?**
- Hard constraints (λ = 10,000): Violations are unacceptable
- Soft constraints (μ = 100): Nice to satisfy but not critical
- Ratio λ/μ = 100:1 ensures hard constraints dominate

#### 2. **Variable Encoding** (Lines 94-113)

Each binary variable represents: `x[course, day, slot] = 1` if scheduled

**Mapping formula:**
```python
var_index = course_id × (num_days × num_slots) + day × num_slots + slot
```

**Example** (3 days, 2 slots per day):
- Course 0, Day 0, Slot 0 → index 0
- Course 0, Day 0, Slot 1 → index 1
- Course 0, Day 1, Slot 0 → index 2
- ...
- Course 1, Day 0, Slot 0 → index 6

**Total variables:** `num_courses × num_days × num_slots`

#### 3. **H1: No Conflicts Constraint** (Lines 115-181)

**Problem:** Students in both Course A and Course B cannot have exams at same time

**QUBO Encoding:**
For each course pair (A, B) with `n` common students:
For each timeslot (day, slot):
```python
Q[var_A, var_B] += λ × n
```

**Energy contribution:**
- If both scheduled at same time: `E += λ × n` (BAD - high energy)
- If at different times: `E += 0` (GOOD - low energy)

**Example:**
- Course 1 and Course 2 have 15 common students
- λ = 10,000
- If both on Day 0, Slot 0: Energy increases by 150,000!

**Code:**
```python
for ci in range(num_courses):
    for cj in range(ci + 1, num_courses):
        conflict_size = conflict_matrix[ci, cj]
        if conflict_size > 0:
            for day in range(num_days):
                for slot in range(num_slots):
                    var_ci = get_variable_index(ci, day, slot)
                    var_cj = get_variable_index(cj, day, slot)
                    Q[var_ci, var_cj] += lambda_h1 * conflict_size
```

#### 4. **H2: Duration Constraint** (Lines 183-244)

**Problem:** A 2-hour exam needs 2 consecutive slots

**Strategy:** Heavily penalize invalid slot assignments

For 2-hour exams:
- ✓ Can assign to Slot 0 (uses slots 0 and 1)
- ✗ Cannot assign to Slot 1 (no room for 2 hours)

**QUBO Encoding:**
```python
for multi_hour_exams:
    for invalid_slots:
        Q[var_index, var_index] += λ × 1000  # Linear penalty
```

**Effect:** Makes `x[course, day, invalid_slot] = 1` extremely expensive

**Example:**
- Exam requires 2 hours, only 2 slots per day
- Allowed: Day 0 Slot 0 ✓
- Forbidden: Day 0 Slot 1 ✗ (penalized with +10,000,000)

#### 5. **H3: One Exam One Slot** (Lines 246-339)

**Problem:** Each exam must be scheduled EXACTLY once (not 0, not 2, exactly 1)

**Mathematical Formulation:**
```
For each course: Σ(all timeslots) x = 1
```

Convert to QUBO by minimizing: `(Σx - 1)²`

**Expansion:**
```
(x₁ + x₂ + ... + xₙ - 1)²
= x₁² + x₂² + ... + 2x₁x₂ + 2x₁x₃ + ... - 2x₁ - 2x₂ - ... + 1
```

For binary variables (`x² = x`):
```
= x₁ + x₂ + ... + 2x₁x₂ + 2x₁x₃ + ... - 2x₁ - 2x₂ - ... + 1
= -x₁ - x₂ - ... + 2(x₁x₂ + x₁x₃ + ...) + 1
```

Drop constant, multiply by λ:
```
= λ(-x₁ - x₂ - ... + 2Σᵢ<ⱼ xᵢxⱼ)
```

**QUBO Form:**
- Diagonal: `Q[i,i] += -λ` (coefficient of xᵢ)
- Off-diagonal: `Q[i,j] += 2λ` (coefficient of xᵢxⱼ)

**Why this works:**
- If exactly 1 variable = 1: Energy = -λ (minimum!)
- If 0 variables = 1: Energy = 0
- If 2 variables = 1: Energy = -2λ + 2λ = 0
- If all n variables = 1: Energy = -nλ + 2λ×C(n,2) (very high!)

**Code:**
```python
for course in courses:
    course_vars = [get_index(course, d, s) for all d, s]
    
    # Diagonal terms
    for var in course_vars:
        Q[var, var] += -lambda_h3
    
    # Off-diagonal terms
    for i in range(len(course_vars)):
        for j in range(i+1, len(course_vars)):
            Q[course_vars[i], course_vars[j]] += 2 * lambda_h3
```

#### 6. **S1: Spread Exams** (Lines 341-415)

**Problem:** Students need revision time between exams

**Penalty Structure:**
| Gap | Penalty Weight | Rationale |
|-----|----------------|-----------|
| Same day | 5 | Very bad - no time to study |
| 1 day | 3 | Bad - minimal prep time |
| 2 days | 1 | Acceptable - some prep time |
| ≥3 days | 0 | Good - adequate prep time |

**QUBO Encoding:**
For courses A and B with common students:
```python
gap = |day_A - day_B|
proximity = penalty_for_gap(gap)
Q[var_A, var_B] += μ × conflict_size × proximity
```

**Example:**
- Course 1 and 2 have 20 common students
- μ = 100
- Same day: Energy += 100 × 20 × 5 = 10,000
- 1 day apart: Energy += 100 × 20 × 3 = 6,000
- 2 days apart: Energy += 100 × 20 × 1 = 2,000
- 3+ days apart: Energy += 0 (perfect!)

**This is a SOFT constraint** - solutions can violate it if necessary to satisfy hard constraints.

#### 7. **QUBO Matrix** (Lines 417-465)

Combines all constraints:
```
Q = λ_H1×H1 + λ_H2×H2 + λ_H3×H3 + μ_S1×S1
```

Must be **symmetric** for quantum solvers:
```python
Q = (Q + Q.T) / 2
```

**Matrix Statistics:**
- Size: `n × n` where `n = courses × days × slots`
- Density: % of non-zero elements
- Typically 20-50% dense (sparse matrix)

#### 8. **BQM Conversion** (Lines 467-498)

D-Wave uses **Binary Quadratic Model (BQM)** format:
```
E(x) = Σᵢ hᵢxᵢ + Σᵢ<ⱼ Jᵢⱼxᵢxⱼ + c
```

Where:
- `h`: Linear coefficients (from Q diagonal)
- `J`: Quadratic coefficients (from Q off-diagonal)
- `c`: Constant offset (we use 0)

**Conversion:**
```python
linear = {i: Q[i,i] for i in range(n)}
quadratic = {(i,j): Q[i,j] for i < j}
bqm = BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')
```

#### 9. **Solvers** (Lines 500-660)

**Neal Simulator (Lines 557-575):**
- Classical simulated annealing
- Unlimited problem size
- Free, runs locally
- Good for testing

```python
sampler = neal.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)
```

**D-Wave QPU (Lines 577-618):**
- Real quantum annealer
- ~5000 qubits (Advantage system)
- Requires embedding
- 1 min/month free tier

```python
sampler = EmbeddingComposite(DWaveSampler())
sampleset = sampler.sample(
    bqm,
    num_reads=100,
    annealing_time=20,  # microseconds
    chain_strength=auto_calculated
)
```

**Hybrid Solver (Lines 620-646):**
- Combines classical + quantum
- No qubit limit
- Handles large problems
- ~$0.30 per problem

```python
sampler = LeapHybridSampler()
sampleset = sampler.sample(bqm)
```

#### 10. **Solution Interpretation** (Lines 648-705)

Converts binary solution back to schedule:
```python
for var_idx, value in solution.items():
    if value == 1:
        course, day, slot = get_course_day_slot(var_idx)
        assignments[course] = (day, slot)
```

#### 11. **Validation** (Lines 707-761)

Checks all constraints:

**H1 Validation:**
```python
for conflicting_courses in conflict_matrix:
    if same_timeslot(course_i, course_j):
        violations['h1_conflicts'] += 1
```

**H2 Validation:**
```python
for multi_hour_exam:
    if assigned_to_invalid_slot:
        violations['h2_duration'] += 1
```

**H3 Validation:**
```python
if course not in assignments:
    violations['h3_unscheduled'] += 1
```

**Valid solution:** All hard constraint violations = 0

---

## Understanding the Output

### 1. QUBO Building Phase
```
Building QUBO Matrix
==================================================================
  [1/4] Building H1: No Student Conflicts...
      ✓ Added 32 conflict penalties
```
- Shows progress for each constraint
- "32 conflict penalties" = 32 course pairs with conflicts × timeslots

### 2. QUBO Statistics
```
QUBO Matrix Statistics:
  Dimensions:      20 × 20
  Non-zero:        186
  Density:         46.50%
```
- **Dimensions:** Total variables (5 courses × 2 days × 2 slots = 20)
- **Non-zero:** Number of non-zero Q elements (interactions)
- **Density:** % of matrix that's non-zero (46.5% is moderate)

### 3. Solving Phase
```
Running Neal simulated annealing...
  Num reads: 100
  ✓ Completed 100 samples
  Best energy: -49850.00
```
- Samples 100 solutions
- Returns lowest energy found
- Negative energy is normal (due to H3 constraint structure)

### 4. Schedule Output
```
EXAM SCHEDULE
Day 0:
  Slot 0: CS101 (80 students, 2hr)
  Slot 1: (empty)
Day 1:
  Slot 0: CS102 (50 students, 1hr), MATH101 (60 students, 1hr)
```
- Shows final assignments
- Multiple exams in same slot is OK if no student conflicts

### 5. Validation
```
Validation Results:
  H1: No conflicts         0 violations    ✓ PASS
  H2: Duration             0 violations    ✓ PASS
  H3: All scheduled        0 violations    ✓ PASS
  OVERALL                                  ✓ VALID
```
- All hard constraints satisfied = valid solution
- S1 violations are tolerated (soft constraint)

---

## Customization Guide

### Adjusting Penalty Weights

**Problem:** Getting invalid solutions?
**Solution:** Increase hard constraint penalties

```python
config = SolverConfig(
    lambda_h1=50000,  # Increase from 10000
    lambda_h2=50000,
    lambda_h3=50000,
    mu_s1=50          # Can decrease soft penalty
)
```

**Rule of thumb:** λ/μ ratio should be 100:1 or higher

### Changing Solver

**For testing (free, fast):**
```python
config.solver_type = 'neal'
config.num_reads = 100
```

**For D-Wave quantum (requires account):**
```python
config.solver_type = 'dwave'
config.num_reads = 50  # Fewer reads to save QPU time
config.annealing_time = 20  # Can increase for harder problems
```

**For large problems:**
```python
config.solver_type = 'hybrid'
# No num_reads needed
```

### Testing Different Datasets

```python
# In main() function, change:
data_dir = './exam_scheduling_TINY'   # 5 courses
data_dir = './exam_scheduling_SMALL'  # 10 courses
data_dir = './exam_scheduling_MEDIUM' # 20 courses
```

---

## Troubleshooting

### Problem: "D-Wave Ocean SDK not installed"
**Solution:**
```bash
pip install dwave-ocean-sdk
```

### Problem: "Dataset not found"
**Solution:**
```bash
python dataset-generator.py
```
This creates the required datasets.

### Problem: Invalid solutions (constraint violations)
**Solutions:**
1. Increase penalty weights (λ values)
2. Try more samples (`num_reads=500`)
3. Use hybrid solver for complex problems
4. Check conflict matrix - might be unsatisfiable

### Problem: "D-Wave connection failed"
**Solutions:**
1. Check API token: `dwave config create`
2. Test connection: `dwave ping`
3. Falls back to Neal automatically
4. Verify free tier not exhausted (check cloud.dwavesys.com)

### Problem: Embedding takes too long
**Solutions:**
1. Use hybrid solver (no embedding needed)
2. Reduce problem size
3. Increase chain_strength

### Problem: Energy is very negative
**Not actually a problem!** Negative energy is normal due to H3 constraint structure.
- What matters: Are hard constraints satisfied?
- Check validation output, not energy value

---

## Next Steps

### 1. Test Scalability
Run on progressively larger datasets:
```bash
# Modify main() to test each:
TINY:   5 courses → ~20 variables
SMALL:  10 courses → ~60 variables
MEDIUM: 20 courses → ~200 variables
```

Track:
- Solution time
- Energy values
- Constraint violations
- Whether D-Wave finds better solutions than Neal

### 2. Try D-Wave Hardware
```python
config.solver_type = 'dwave'
```

Compare with Neal:
- Solution quality (energy)
- Execution time
- Success rate (% valid solutions)

### 3. Implement Stage 2: Room Assignment
Next phase assigns rooms to scheduled exams.

### 4. Classical Baseline
Implement OR-Tools solver for comparison:
```bash
pip install ortools
```

### 5. Parameter Tuning
Experiment with:
- Annealing time (1-100 μs)
- Number of reads (10-1000)
- Chain strength (0.5-10.0×max|Q|)
- Penalty ratios (λ/μ)

---

## Key Takeaways

### What We Built
✅ Complete QUBO formulation for exam scheduling
✅ 4 constraints (H1, H2, H3, S1)
✅ D-Wave Ocean SDK integration
✅ 3 solver options (Neal, D-Wave, Hybrid)
✅ Comprehensive validation
✅ Human-readable output

### QUBO Concepts
- **Binary variables**: x ∈ {0, 1}
- **Energy minimization**: E(x) = Σ Qᵢⱼxᵢxⱼ
- **Penalty method**: High λ for hard, low μ for soft
- **Constraint encoding**: Mathematical → QUBO matrix

### D-Wave Workflow
1. Build Q matrix
2. Convert to BQM
3. Sample solutions
4. Interpret best result
5. Validate constraints

### When to Use Quantum
- ✅ Hard combinatorial optimization
- ✅ Many local minima
- ✅ Classical methods fail or too slow
- ❌ Simple problems (overkill)
- ❌ Problems with clean structure (use specialized algorithms)

---

## References

- **D-Wave Ocean Docs**: https://docs.ocean.dwavesys.com/
- **QUBO Tutorial**: https://docs.ocean.dwavesys.com/en/stable/concepts/quadratic_models.html
- **Leap Dashboard**: https://cloud.dwavesys.com/leap/
- **Example Code**: https://github.com/dwave-examples

---

## Contact & Support

For issues:
1. Check troubleshooting section above
2. Review D-Wave documentation
3. Verify dataset generated correctly
4. Check penalty weights are appropriate

Happy quantum scheduling! 🎓⚛️
