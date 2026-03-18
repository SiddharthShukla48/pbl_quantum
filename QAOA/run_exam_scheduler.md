# Exam Scheduling Pipeline (Neal Only) - Full Technical Formulation

This document is a report-ready reference for the implementation in `run_exam_scheduler.py`.
It includes all nomenclature, mathematical formulations, and stage-wise equations used in the pipeline.

---

## 1) Problem Statement

Given a set of exams, assign each exam to one time slot such that:

1. Each exam is assigned to exactly one slot.
2. Exams with student overlap should not be in the same slot.
3. Exams in the same semester should avoid consecutive slots (soft preference).
4. Slot load should stay close to room capacity using a slack-based soft penalty.

This is solved as a Binary Quadratic Optimization (QUBO) problem and optimized with D-Wave Neal (simulated annealing).

---

## 2) End-to-End Pipeline Stages

1. Data ingestion (synthetic or CSV).
2. CSV filtering and graph construction.
3. Decision-variable construction.
4. QUBO objective assembly (C1-C4).
5. Conversion to BQM and optimization with Neal.
6. Decode solution to timetable.
7. Validate conflicts and export artifacts.

---

## 3) Nomenclature (Symbols and Sets)

### 3.1 Core Sets

- $\mathcal{E} = \{1,2,\dots,n\}$: set of exams/courses.
- $\mathcal{K} = \{1,2,\dots,K\}$: set of time slots.
- $\mathcal{B} = \{0,1,\dots,B-1\}$: set of slack-bit indices per slot.

### 3.2 Inputs

- $n$: number of exams.
- $K$: number of slots.
- $A_{ij} \in \{0,1\}$: conflict adjacency matrix.
  - $A_{ij}=1$ means exam $i$ conflicts with exam $j$.
  - $A$ is symmetric and diagonal is 0.
- $e_i$: enrollment of exam $i$.
- $y_i$: semester index used in C3 (stored as `year` in code).
- $C$: slot capacity (`--capacity`).

### 3.3 Decision Variables

- $x_{ik} \in \{0,1\}$: 1 if exam $i$ is assigned to slot $k$, else 0.
- $s_{kb} \in \{0,1\}$: slack bit $b$ for slot $k$ with weight $2^b$.

### 3.4 Hyperparameters

- $\lambda_1$: C1 one-hot penalty.
- $\lambda_2$: C2 conflict penalty.
- $\lambda_3$: C3 consecutive-semester penalty.
- $\lambda_4$: C4 capacity-slack penalty.

---

## 4) Data and Graph Construction

## 4.1 CSV Filtering Rules

When `--input-csv` is used, rows are filtered by:

1. Registration Status = Approved
2. Course Classification = Theory
3. Academic Session in {JUL-NOV 2025, WINTER 2025}
4. Semester not in {I, II}

Optional pre-filter truncation:

- `--max-rows M`: only first $M$ rows are considered before filtering.

### 4.2 Graph Modes

- `major`: use only rows where Course Type = MAJOR.
- `all`: use all filtered theory rows.
- `both`: run `major` and `all` sequentially.

### 4.3 Conflict Edge Definition

For each student, let enrolled course set be $S_u$.
For each unordered pair $(i,j) \subset S_u$, set:

$$
A_{ij} = A_{ji} = 1
$$

Hence one undirected conflict pair is counted once in metrics:

$$
|\mathcal{C}| = \sum_{i<j} A_{ij}
$$

---

## 5) Decision Variable Indexing in QUBO Vector

The flattened binary vector contains exam variables first, then slack variables.

### 5.1 Exam Variable Index

$$
p(i,k) = iK + k
$$

for zero-based exam index $i \in [0,n-1]$ and slot index $k \in [0,K-1]$.

### 5.2 Slack Variable Index

$$
p_s(k,b) = nK + kB + b
$$

where $B$ is number of slack bits per slot.

### 5.3 Number of Slack Bits

Current implementation uses capacity-based bound:

$$
B = \left\lceil \log_2(C+1) \right\rceil
$$

This matches formulation $\left(L_k + S_k - C\right)^2$ where slack represents residual to capacity.

### 5.4 Total Variables

$$
N_{vars} = nK + KB
$$

---

## 6) Full Objective Function

The total QUBO objective is:

$$
E = E_1 + E_2 + E_3 + E_4
$$

## 6.1 C1: One Exam in Exactly One Slot

$$
E_1 = \lambda_1 \sum_{i \in \mathcal{E}} \left(1 - \sum_{k \in \mathcal{K}} x_{ik}\right)^2
$$

Expanded contribution pattern:

- Diagonal $x_{ik}$ term: $-\lambda_1$
- Same-exam, different-slot pair $(k \neq k')$: $+2\lambda_1$

This enforces one-hot assignment per exam.

## 6.2 C2: Conflict-Free Slot Assignment

$$
E_2 = \lambda_2 \sum_{i<j} A_{ij} \sum_{k \in \mathcal{K}} x_{ik}x_{jk}
$$

Interpretation: if conflicting exams share slot $k$, energy increases.

## 6.3 C3: Same-Semester Exams Avoid Consecutive Slots (Soft)

Using semester proxy $y_i$:

$$
E_3 = \lambda_3 \sum_{i<j} \mathbf{1}[y_i=y_j]
\sum_{k=1}^{K-1} \left(x_{ik}x_{j,k+1} + x_{i,k+1}x_{jk}\right)
$$

Interpretation: same-semester pairs in adjacent slots are penalized.

## 6.4 C4: Capacity with Binary-Weighted Slack Bits

Define per-slot load and slack:

$$
L_k = \sum_i e_i x_{ik}, \quad
S_k = \sum_{b=0}^{B-1} 2^b s_{kb}
$$

Penalty:

$$
E_4 = \lambda_4 \sum_{k \in \mathcal{K}} (L_k + S_k - C)^2
$$

Note: in this formulation, $S_k$ models residual to capacity with binary encoding.

---

## 7) C4 Expansion to QUBO Coefficients

For each slot $k$:

$$
(L_k + S_k - C)^2 = L_k^2 + S_k^2 + 2L_kS_k - 2CL_k - 2CS_k + C^2
$$

Constant $C^2$ does not affect argmin and can be ignored.

### 7.1 Exam Diagonal Coefficient

For each $x_{ik}$:

$$
Q[x_{ik},x_{ik}] \;{+}=\; \lambda_4(e_i^2 - 2Ce_i)
$$

### 7.2 Exam Off-Diagonal (same slot)

For $i<j$:

$$
Q[x_{ik},x_{jk}] \;{+}=\; 2\lambda_4 e_i e_j
$$

### 7.3 Slack Diagonal

For slack bit $b$ with weight $w_b=2^b$:

$$
Q[s_{kb},s_{kb}] \;{+}=\; \lambda_4\, w_b(w_b-2C)
$$

### 7.4 Slack Off-Diagonal (same slot)

For $b<b'$:

$$
Q[s_{kb},s_{kb'}] \;{+}=\; 2\lambda_4\,2^b2^{b'}
= \lambda_4\,2^{b+b'+1}
$$

### 7.5 Exam-Slack Cross Terms

$$
Q[x_{ik},s_{kb}] \;{+}=\; 2\lambda_4 e_i 2^b
$$

---

## 8) Synthetic Mode Formulation Notes

When no CSV is given:

1. Random synthetic courses and enrollments are generated.
2. Conflict graph is generated by target edge density (`--conflict-pct`), not from co-enrollment.
3. Same C1-C4 optimization machinery is used afterward.

---

## 9) Solver Stage (Neal)

The QUBO matrix $Q$ is converted into a Binary Quadratic Model (BQM):

$$
\min_{z \in \{0,1\}^{N_{vars}}} z^T Q z
$$

Neal runs simulated annealing for `--num-reads` independent samples and returns best-energy sample.

---

## 10) Decoding and Validation

## 10.1 Decoding

For each exam $i$, decoded slot is the slot index where $x_{ik}=1$ (first one found in scan order).
Slack variables are ignored in timetable output.

## 10.2 Conflict Validation Metric

After decoding, conflicts are counted as:

$$
\text{violations} = \sum_{i<j} A_{ij}\,\mathbf{1}[slot(i)=slot(j)]
$$

A solution is valid if violations = 0 and all exams are assigned.

---

## 11) Reported Metrics in JSON

Each `neal_results.json` contains:

- backend
- adjacency_mode
- num_courses
- num_students
- num_enrollments
- k
- avg_courses_per_student
- runtime_seconds
- energy
- is_valid
- num_conflicts
- colors_used
- coloring map

In CSV mode, student and enrollment counts come from the selected filtered graph metadata.

---

## 12) CLI Parameters (Complete)

```text
--courses INT              Synthetic mode: number of courses
--students INT             Synthetic mode metadata only (default: 50)
--k INT                    Number of time slots
--avg-courses INT          Synthetic mode metadata only (default: 4)
--conflict-pct FLOAT       Synthetic mode edge density (default: 40.0)

--input-csv PATH           Enable CSV mode
--max-rows INT             CSV mode: use first N rows before filters
--adjacency-mode STR       major | all | both (default: all)

--backend STR              neal (Neal only)
--num-reads INT            Neal reads (default: 1000)

--lambda1 FLOAT            C1 penalty (default: 10000)
--lambda2 FLOAT            C2 penalty (default: 5000)
--lambda3 FLOAT            C3 penalty (default: 500, 0 disables)
--lambda4 FLOAT            C4 penalty (default: 200, 0 disables)
--capacity INT             Room capacity C for C4

--visualize                Save heatmap/graph/timetable PNGs
```

---

## 13) Output Artifacts

For each run:

```text
output/run_YYYYMMDD_HHMMSS/
  courses.csv
  students.csv
  enrollments.csv
  conflict_adjacency.csv
  metadata.json
  qubo_matrix.npy
  neal_results.json
  timetable_neal.csv (if valid)
  *.png (if --visualize)
```

CSV mode also saves both graph matrices:

- conflict_adjacency_major.csv
- conflict_adjacency_all.csv

For `--adjacency-mode both`, outputs are separated in:

```text
output/run_YYYYMMDD_HHMMSS/major/
output/run_YYYYMMDD_HHMMSS/all/
```

---

## 14) Recommended Command Templates

### 14.1 Full CSV Run (all rows)

```bash
python run_exam_scheduler.py \
  --input-csv "../Student Course (Jul-Nov 2025 and Winter 2025).csv" \
  --adjacency-mode both \
  --k 18 --num-reads 1000 \
  --lambda1 10000 --lambda2 5000 --lambda3 500 --lambda4 200 --capacity 120
```

### 14.2 Partial CSV Run (for experiments)

```bash
python run_exam_scheduler.py \
  --input-csv "../Student Course (Jul-Nov 2025 and Winter 2025).csv" \
  --max-rows 500 \
  --adjacency-mode major \
  --k 18 --num-reads 1000 \
  --lambda1 10000 --lambda2 5000 --lambda3 500 --lambda4 200 --capacity 120
```

### 14.3 Synthetic Baseline Run

```bash
python run_exam_scheduler.py \
  --courses 50 --students 500 --k 12 --conflict-pct 35 \
  --num-reads 1000 \
  --lambda1 10000 --lambda2 5000 --lambda3 500 --lambda4 200 --capacity 120
```

---

## 15) Important Implementation Notes for Report Writing

1. Conflict matrix is undirected and symmetric; one pair is counted once using $i<j$.
2. C3 is soft and semester-based (via internal `year` field from semester mode).
3. C4 uses capacity-based binary slack dimension: $\lceil\log_2(C+1)\rceil$.
4. Decoding ignores slack variables and outputs only exam-slot assignments.
5. In CSV mode, reported student counts in results are taken from filtered selected graph metadata.

This document is aligned with the current implementation state and is suitable as the mathematical specification section of your project report.
