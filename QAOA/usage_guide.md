# QAOA Solver Usage Guide

## Quick Start

### 1. Generate Datasets (Once)
```bash
python QAOA/01_generate_dataset.py
python QAOA/02_visualize_graph.py
python QAOA/03_build_qubo.py
```

### 2. Find Your System's Limits
```bash
# Automatic benchmarking
python QAOA/benchmark_k_values.py tiny

# Expected output:
#   K=2: ✓ 10 variables, 0.5s
#   K=3: ✓ 15 variables, 1.2s
#   K=4: ✓ 20 variables, 3.5s
#   K=5: ⚠ timeout after 600s
```

### 3. Solve Specific Problems
```bash
# Syntax: python 04_solve_qaoa.py <dataset> <K> [options]

# Easy problems
python QAOA/04_solve_qaoa.py tiny 3          # ~15 vars, 1-2 min
python QAOA/04_solve_qaoa.py tiny 4          # ~20 vars, 2-5 min

# Challenging problems
python QAOA/04_solve_qaoa.py small 3 --reps 1 --maxiter 50  # ~30 vars
python QAOA/04_solve_qaoa.py small 4 --reps 1 --maxiter 30  # ~40 vars (limit!)
```

---

## Problem Size Reference

| Dataset | K | Variables | Expected Time | Difficulty |
|---------|---|-----------|---------------|------------|
| tiny    | 2 | 10        | < 1 min       | ✓ Easy     |
| tiny    | 3 | 15        | 1-2 min       | ✓ Easy     |
| tiny    | 4 | 20        | 2-5 min       | ⚠ Medium   |
| tiny    | 5 | 25        | 5-10 min      | ⚠ Medium   |
| small   | 3 | 30        | 5-15 min      | ⚠ Medium   |
| small   | 4 | 40        | 10-30 min     | ✗ Hard     |
| small   | 5 | 50        | > 30 min      | ✗ Very Hard|

**Rule of thumb:** ~40 variables is the practical limit for QAOA simulator on most laptops.

---

## Performance Tuning

### If Runtime Too Long
1. Reduce QAOA depth: `--reps 1`
2. Reduce optimizer iterations: `--maxiter 30`
3. Try smaller K value
4. Use smaller dataset

### If Memory Issues
1. Close other applications
2. Use smaller dataset
3. Reduce K value

### For Research/Publication
- Use `--reps 2` for quality results
- Use `--maxiter 100` for thorough optimization
- Test multiple K values: K-1, K, K+1 around chromatic number

---

## Command Examples

### Experimentation
```bash
# Compare different K values on same dataset
python QAOA/04_solve_qaoa.py tiny 3
python QAOA/04_solve_qaoa.py tiny 4
python QAOA/04_solve_qaoa.py tiny 5

# Compare tiny vs small
python QAOA/04_solve_qaoa.py tiny 3
python QAOA/04_solve_qaoa.py small 3 --reps 1
```

### For Presentations
```bash
# Generate nice visualizations
python QAOA/04_solve_qaoa.py tiny 3
# Opens PNG in output/run_*/visualizations/
```

### For Paper Results
```bash
# Run full benchmark suite
python QAOA/benchmark_k_values.py --all --max-k 5
# Generates: output/benchmark_results.csv
#            output/benchmark_plot.png
```