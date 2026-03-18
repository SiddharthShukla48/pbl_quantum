# Exam Scheduling Pipeline (Neal Only)

This document describes the current workflow in [QAOA/run_exam_scheduler.py](QAOA/run_exam_scheduler.py).

The pipeline is now Neal-only (D-Wave simulated annealing). QAOA/Qiskit backend options were removed.

## What the script does

1. Builds exam-conflict data either from synthetic random data or from your university CSV.
2. Builds QUBO with four constraints:
1. C1: one exam in exactly one slot.
2. C2: conflicting exams cannot share a slot.
3. C3: same-semester exams should avoid consecutive slots.
4. C4: soft slot-capacity penalty.
3. Solves with Neal.
4. Validates conflicts and exports timetable/results.

## Installation

```bash
pip install dwave-ocean-sdk numpy pandas
pip install matplotlib networkx  # optional, for --visualize
```

## Core usage

### 1. Synthetic random mode

```bash
python run_exam_scheduler.py --courses 10 --k 5 --conflict-pct 40
```

### 2. University CSV mode (current real-data workflow)

```bash
python run_exam_scheduler.py \
  --input-csv "../Student Course (Jul-Nov 2025 and Winter 2025).csv" \
  --adjacency-mode all \
  --k 16 \
  --num-reads 1000 \
  --lambda1 10000 --lambda2 5000 --lambda3 500 --lambda4 200 --capacity 120
```

## Real-data filters currently applied

When `--input-csv` is used, the script applies these filters before graph creation:

1. Registration Status == Approved
2. Course Classification == Theory
3. Academic Session in {JUL-NOV 2025, WINTER 2025}
4. Excludes Semester I and Semester II

## Major vs All adjacency

CSV mode builds and saves two conflict graphs:

1. major graph: rows where Course Type == MAJOR
2. all graph: all filtered theory rows (major + elective + open elective, etc.)

Select the solver graph with:

- --adjacency-mode major
- --adjacency-mode all
- --adjacency-mode both

Both adjacency CSV files are always exported for comparison. With `both`, solver runs sequentially for each graph in separate output subfolders.

## C3 interpretation in CSV mode

For CSV data, semester number is mapped into the internal year field used by C3. That means C3 acts as:

- same-semester exams should not be consecutive.

## Parameters

```text
--courses INT              Synthetic mode: number of courses
--students INT             Synthetic mode metadata only (default: 50)
--k INT                    Number of time slots
--avg-courses INT          Synthetic mode metadata only (default: 4)
--conflict-pct FLOAT       Synthetic mode edge density (default: 40.0)

--input-csv PATH           Enable university CSV mode
--adjacency-mode STR       major | all | both (default: all)

--backend STR              neal (fixed; Neal only)
--num-reads INT            Neal reads (default: 1000)

--lambda1 FLOAT            C1 penalty (default: 10000)
--lambda2 FLOAT            C2 penalty (default: 5000)
--lambda3 FLOAT            C3 penalty (default: 500; 0 disables)
--lambda4 FLOAT            C4 penalty (default: 200; 0 disables)
--capacity INT             Slot capacity for C4 (default: auto)

--visualize                Save heatmap/graph/timetable PNGs
```

## Output structure

Each run creates:

```text
output/run_YYYYMMDD_HHMMSS/
├── courses.csv
├── students.csv
├── enrollments.csv
├── conflict_adjacency.csv              # selected graph used in solver
├── conflict_adjacency_major.csv        # CSV mode only
├── conflict_adjacency_all.csv          # CSV mode only
├── metadata.json
├── qubo_matrix.npy
├── neal_results.json
├── timetable_neal.csv                  # if valid
└── *.png                               # if --visualize
```

## Recommended commands for your current dataset

### Major-only graph

```bash
python run_exam_scheduler.py \
  --input-csv "../Student Course (Jul-Nov 2025 and Winter 2025).csv" \
  --adjacency-mode major \
  --k 16 --num-reads 1000 \
  --lambda1 10000 --lambda2 5000 --lambda3 500 --lambda4 200 --capacity 120
```

### All-courses graph

```bash
python run_exam_scheduler.py \
  --input-csv "../Student Course (Jul-Nov 2025 and Winter 2025).csv" \
  --adjacency-mode all \
  --k 18 --num-reads 1000 \
  --lambda1 10000 --lambda2 5000 --lambda3 500 --lambda4 200 --capacity 120
```

### Run both graphs in one command

```bash
python run_exam_scheduler.py \
  --input-csv "../Student Course (Jul-Nov 2025 and Winter 2025).csv" \
  --adjacency-mode both \
  --k 18 --num-reads 1000 \
  --lambda1 10000 --lambda2 5000 --lambda3 500 --lambda4 200 --capacity 120
```

For `--adjacency-mode both`, outputs are separated into:

```text
output/run_YYYYMMDD_HHMMSS/
├── major/
└── all/
```

## Notes

1. If Valid: NO, increase --k and/or --num-reads.
2. Start with C3/C4 enabled, but if feasibility is hard, first find a conflict-free baseline with larger k, then tune penalties.
3. --backend is retained only for compatibility but accepts neal only.
