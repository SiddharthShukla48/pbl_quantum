# Exam Scheduling QUBO Sample Dataset System - Complete Guide

## What You Have

I've created **3 Python files** that work together to help you test your exam scheduling QUBO code:

1. **dataset-generator.py** - Generates realistic sample data
2. **data-loader.py** - Loads and validates data
3. **qubo-builder-example.py** - Shows how to build QUBO matrices

---

## Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
pip install pandas numpy
```

### Step 2: Generate Sample Data
```bash
python dataset-generator.py
```

This creates 4 directories with sample exam scheduling data:
- `exam_scheduling_data_tiny/` (5 courses, 30 students)
- `exam_scheduling_data_small/` (10 courses, 100 students) ← **Start here**
- `exam_scheduling_data_medium/` (20 courses, 250 students)
- `exam_scheduling_data_large/` (50 courses, 500 students)

### Step 3: Load and Inspect Data
```bash
python data-loader.py exam_scheduling_data_small
```

### Step 4: See QUBO Building Example
```bash
python qubo-builder-example.py
```

---

## Detailed Explanation

### File 1: dataset-generator.py

**What it does**: Creates realistic exam scheduling datasets

**Key Components**:

```python
class ExamSchedulingDataset:
    """Generates all the data needed for exam scheduling"""
    
    def __init__(self, num_courses=10, num_students=100, ...):
        # Configure dataset size
        
    def generate_courses(self):
        # Creates courses with enrollments
        # Example: CS101 with 120 students, 2-hour exam
        
    def generate_students(self):
        # Creates student population
        
    def generate_enrollments(self):
        # Maps students to courses
        # KEY: This determines conflicts!
        # If Student 5 takes BOTH CS101 and MATH101,
        # then CS101 and MATH101 conflict
        
    def generate_conflict_matrix(self):
        # Builds N×N matrix
        # conflict_matrix[i,j] = number of students in BOTH course i and j
        # This is CRITICAL for QUBO!
        
    def generate_rooms(self):
        # Room capacities and types
        
    def generate_faculty(self):
        # Faculty with ranks and duty quotas
```

**Example output** (courses.csv):
```csv
course_id,course_code,enrollment,duration_hours,room_type_requirement
0,CS101,120,2,regular
1,CS102,45,1,regular
2,MATH101,80,2,regular
```

**Example output** (conflict_matrix.csv):
```csv
      Course_0  Course_1  Course_2
Course_0      0        30        15
Course_1     30         0        10
Course_2     15        10         0
```

**Interpretation**: 
- 30 students take BOTH Course_0 and Course_1
- So these courses CANNOT be scheduled at the same time!

---

### File 2: data-loader.py

**What it does**: Loads CSV files into memory for your QUBO code

**Key Components**:

```python
class ExamSchedulingDataLoader:
    """Loads all CSV files from a directory"""
    
    def load(self):
        # Reads all CSV files
        # Returns ExamSchedulingData object
        
    def print_statistics(self, data):
        # Shows summary of loaded data
```

**How to use**:
```python
from data_loader import ExamSchedulingDataLoader

# Load data
loader = ExamSchedulingDataLoader('./exam_scheduling_data_small')
data = loader.load()

# Access components
courses = data.courses                # DataFrame
conflict_matrix = data.conflict_matrix # NumPy array
enrollments = data.enrollments        # DataFrame
rooms = data.rooms                    # DataFrame
faculty = data.faculty                # DataFrame

# Example: Get conflicts between two courses
num_conflicts = data.conflict_matrix[0, 1]
print(f"{num_conflicts} students take both courses 0 and 1")
```

---

### File 3: qubo-builder-example.py

**What it does**: Shows step-by-step how to build QUBO matrices

**This is the most important file for understanding!**

#### What is QUBO?

QUBO = Quadratic Unconstrained Binary Optimization

```
Format: minimize E(x) = Σ Q[i,j] * x[i] * x[j]

where:
- x = binary variables (0 or 1)
- Q = penalty matrix (you build this!)
- E(x) = energy/cost function

Goal: Quantum computer finds x that minimizes E(x)
```

#### Stage 1: Timeslot Allocation

**Problem**: Assign each exam to a (day, timeslot)

**Variables**: 
```
x[course_id, day, timeslot] = 1 if course scheduled at that time
                            = 0 otherwise

Example: 10 courses, 3 days, 2 slots = 60 variables
```

**Variable Indexing** (crucial!):
```python
def get_variable_index(course_id, day, slot):
    """Convert (course, day, slot) to single index"""
    return course_id * (days * slots) + day * slots + slot

Example:
Course 2, Day 1, Slot 0 with 3 days, 2 slots:
index = 2 * (3*2) + 1*2 + 0 = 12 + 2 = 14
```

#### Constraint H1: No Student Conflicts

**Rule**: If courses share students, they CAN'T be at same time

**Implementation**:
```python
def build_h1_no_conflict(self, lambda_h1=10000):
    Q = zeros matrix
    
    # For each pair of courses
    for course_i in courses:
        for course_j in courses:
            # Check conflict
            conflict_size = conflict_matrix[i, j]
            
            if conflict_size > 0:
                # For each timeslot
                for day in days:
                    for slot in slots:
                        var_i = get_index(i, day, slot)
                        var_j = get_index(j, day, slot)
                        
                        # Add penalty to Q matrix
                        # If both variables = 1, adds cost
                        Q[var_i, var_j] += lambda_h1 * conflict_size
    
    return Q
```

**What this does**:
- If Course 0 and Course 1 have 30 shared students
- And you try to schedule both at Day 0, Slot 0
- Then x[0,0,0] = 1 AND x[1,0,0] = 1
- This adds energy: E += 10000 * 30 = 300,000
- HUGE penalty! Quantum computer will avoid this

#### Constraint H3: One Exam One Slot

**Rule**: Each course must be scheduled EXACTLY ONCE

**Mathematical form**:
```
For each course c:
  Σ(all days, all slots) x[c, d, t] = 1

Convert to QUBO penalty:
  (Σx - 1)² = penalizes if sum ≠ 1
```

**Why (Σx - 1)²?**
```
If Σx = 0 (no slot):    (0-1)² = 1   (penalty)
If Σx = 1 (one slot):   (1-1)² = 0   (no penalty - good!)
If Σx = 2 (two slots):  (2-1)² = 1   (penalty)
If Σx = 3 (three slots):(3-1)² = 4   (bigger penalty)
```

**QUBO encoding of (Σx - 1)²**:
```
(x1 + x2 + ... + xn - 1)² 

Expands to:
= x1² + x2² + ... + 2·x1·x2 + 2·x1·x3 + ... - 2·x1 - 2·x2 - ... + 1

For binary variables (x² = x):
= x1 + x2 + ... + 2·Σ(i<j) xi·xj - 2·(x1 + x2 + ...) + 1

Simplifies to:
= -Σxi + 2·Σ(i<j) xi·xj

Q matrix entries:
- Diagonal Q[i,i] = λ·(-1) for each variable
- Off-diagonal Q[i,j] = λ·(2) for each pair
```

**Implementation**:
```python
def build_h3_one_exam_one_slot(self, lambda_h3=10000):
    Q = zeros matrix
    
    for course in courses:
        # Get all timeslot variables for this course
        vars = [get_index(course, d, t) for d in days for t in slots]
        
        # Diagonal terms (coefficient 1 from expansion)
        for var in vars:
            Q[var, var] += lambda_h3
        
        # Off-diagonal terms (coefficient 2 from expansion)
        for i in range(len(vars)):
            for j in range(i+1, len(vars)):
                Q[vars[i], vars[j]] += 2 * lambda_h3
    
    return Q
```

#### Constraint S1: Spread Exams (Soft)

**Rule**: Students prefer GAP between their exams for revision

**Penalty structure**:
```
Same day, different slot:  penalty = 5
Consecutive days:          penalty = 3
2 days apart:              penalty = 1
≥3 days apart:             penalty = 0 (no penalty)
```

**Implementation**:
```python
def build_s1_spread_exams(self, mu_s1=100):
    Q = zeros matrix
    
    for ci in courses:
        for cj in courses:
            conflict_size = conflict_matrix[ci, cj]
            
            if conflict_size > 0:
                for d1 in days:
                    for d2 in days:
                        gap = abs(d1 - d2)
                        
                        # Proximity penalty
                        if gap == 0:
                            proximity = 5
                        elif gap == 1:
                            proximity = 3
                        elif gap == 2:
                            proximity = 1
                        else:
                            proximity = 0
                        
                        if proximity > 0:
                            for t1 in slots:
                                for t2 in slots:
                                    var_i = get_index(ci, d1, t1)
                                    var_j = get_index(cj, d2, t2)
                                    Q[var_i, var_j] += mu_s1 * proximity * conflict_size
    
    return Q
```

**Note**: mu_s1 (100) << lambda_h1 (10000)
- Soft constraints have smaller penalties
- Hard constraints MUST be satisfied
- Soft constraints are NICE TO HAVE

#### Complete QUBO

```python
def build_full_qubo(self):
    Q = zeros matrix
    
    # Add all constraints
    Q += build_h1_no_conflict(lambda_h1=10000)   # HARD
    Q += build_h3_one_exam_one_slot(lambda_h3=10000)  # HARD
    Q += build_s1_spread_exams(mu_s1=100)        # SOFT
    
    # Make symmetric (required for quantum solvers)
    Q = (Q + Q.T) / 2
    
    return Q
```

---

## Complete Workflow Example

```python
# Step 1: Generate data (run once)
from dataset_generator import ExamSchedulingDataset

gen = ExamSchedulingDataset(num_courses=10, num_students=100)
gen.save_to_csv('./exam_scheduling_data_small')

# Step 2: Load data
from data_loader import ExamSchedulingDataLoader

loader = ExamSchedulingDataLoader('./exam_scheduling_data_small')
data = loader.load()

# Step 3: Build QUBO
from qubo_builder_example import Stage1QuboBuilder

builder = Stage1QuboBuilder(data)
Q = builder.build_full_qubo()

# Q is now a NumPy array (60 × 60 for 10 courses, 3 days, 2 slots)

# Step 4: Solve with QAOA (your code!)
from qiskit import ...
# ... your QAOA implementation ...
solution = qaoa_solver.solve(Q)

# Step 5: Interpret solution
for i, val in enumerate(solution):
    if val == 1:  # This variable is selected
        course, day, slot = builder.get_course_day_slot(i)
        print(f"Course {course} scheduled at Day {day}, Slot {slot}")
```

---

## Dataset Sizes

| Dataset | Courses | Students | Variables (Stage 1) | QUBO Size |
|---------|---------|----------|---------------------|-----------|
| TINY | 5 | 30 | 20 | 20×20 |
| SMALL | 10 | 100 | 60 | 60×60 |
| MEDIUM | 20 | 250 | 200 | 200×200 |
| LARGE | 50 | 500 | 1000 | 1000×1000 |

**Start with TINY for debugging, then scale to SMALL!**

---

## Key Concepts Summary

### 1. Conflict Matrix
```
conflict_matrix[i,j] = number of students in BOTH course i and j

This drives the entire scheduling problem!
```

### 2. Variable Encoding
```
x[course, day, slot] → single index

Example: 10 courses, 3 days, 2 slots
Course 0: indices 0-5
Course 1: indices 6-11
Course 2: indices 12-17
...
```

### 3. QUBO Penalty Structure
```
Hard constraints: λ = 10000 (must satisfy)
Soft constraints: μ = 100 (nice to have)

Energy = λ·H1 + λ·H3 + μ·S1

Minimize energy → find best schedule
```

### 4. Constraint Encoding
```
"Must be true" → (expression)² penalty
"Prefer true" → linear penalty

Example:
- "Exactly one": (Σx - 1)²
- "At most one": Σ(i<j) xi·xj
- "At least one": -(Σxi)
```

---

## Testing Your Code

### Test 1: Can you load data?
```bash
python data-loader.py exam_scheduling_data_tiny
```

### Test 2: Can you build QUBO?
```python
from data_loader import ExamSchedulingDataLoader
from qubo_builder_example import Stage1QuboBuilder

loader = ExamSchedulingDataLoader('./exam_scheduling_data_tiny')
data = loader.load()

builder = Stage1QuboBuilder(data)
Q = builder.build_full_qubo()

print(f"QUBO shape: {Q.shape}")  # Should be (20, 20) for TINY
```

### Test 3: Does solution satisfy H3?
```python
# If your QAOA returns solution x
for course in range(5):  # 5 courses in TINY
    # Check how many timeslots this course is assigned to
    count = 0
    for day in range(2):
        for slot in range(2):
            var_idx = builder.get_variable_index(course, day, slot)
            if solution[var_idx] == 1:
                count += 1
    
    if count != 1:
        print(f"ERROR: Course {course} assigned {count} times (should be 1)")
```

### Test 4: Does solution avoid conflicts?
```python
# Check H1: no conflicts
for day in range(2):
    for slot in range(2):
        courses_in_slot = []
        for course in range(5):
            var_idx = builder.get_variable_index(course, day, slot)
            if solution[var_idx] == 1:
                courses_in_slot.append(course)
        
        # Check all pairs in this slot
        for i in range(len(courses_in_slot)):
            for j in range(i+1, len(courses_in_slot)):
                ci, cj = courses_in_slot[i], courses_in_slot[j]
                conflicts = data.conflict_matrix[ci, cj]
                if conflicts > 0:
                    print(f"ERROR: Courses {ci} and {cj} conflict ({conflicts} students)")
```

---

## Next Steps

1. **Run the example**: `python qubo-builder-example.py`
2. **Generate your own data**: Modify parameters in dataset-generator.py
3. **Implement your QAOA solver**: Use the Q matrix from builder
4. **Test on TINY dataset** (20 variables, easy to debug)
5. **Scale to SMALL** (60 variables)
6. **Add Stage 2 and Stage 3** following the same pattern

---

## Questions You Might Have

**Q: Why is conflict_matrix so important?**
A: It determines which courses CAN'T be at the same time. This is the core of the scheduling problem!

**Q: What's the difference between lambda and mu?**
A: lambda (λ) for HARD constraints (must satisfy), mu (μ) for SOFT constraints (nice to have). λ >> μ.

**Q: How do I know if my solution is good?**
A: Check:
1. Energy value (lower = better)
2. All H3 satisfied? (each course scheduled once)
3. All H1 satisfied? (no conflicts)
4. S1 score (how well spread)

**Q: Can I add more constraints?**
A: Yes! Follow the same pattern:
1. Define rule
2. Convert to mathematical expression
3. Square it if it's "must be true"
4. Add to Q matrix with appropriate penalty

---

## File Locations After Running

```
exam_scheduling_data_tiny/
  ├── courses.csv
  ├── students.csv
  ├── enrollments.csv
  ├── rooms.csv
  ├── faculty.csv
  ├── conflict_matrix.csv
  ├── faculty_preferences.csv
  ├── room_availability.csv
  └── metadata.json

exam_scheduling_data_small/
  └── (same structure)

dataset-generator.py
data-loader.py
qubo-builder-example.py
README.md (this file)
```

---

## Summary

You now have:
✅ Sample dataset generator (4 sizes)
✅ Data loader with validation
✅ Complete QUBO building example
✅ Detailed explanations of every step

**Start here**:
```bash
python dataset-generator.py
python qubo-builder-example.py
```

Then implement your QAOA solver using the Q matrix!

Good luck with your research! 🚀
