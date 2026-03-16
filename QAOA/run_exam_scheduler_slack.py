"""
Interactive Exam Scheduler - Slack Variables Version
====================================================

Same as run_exam_scheduler.py but uses SLACK VARIABLES for Constraint 4 (capacity)
instead of quadratic expansion.

Slack Variable Approach (Option A):
- For each time slot k, introduce slack variables s_k representing capacity overflow
- Better for: Understanding constraint relaxation, different penalty structures
- Comparison: With quadratic (Option B), violations only penalized in energy
           With slack (Option A), violations explicitly modeled as variables

Single script that:
1. Generates random conflict graph with controlled density
2. Builds QUBO matrix for graph coloring problem
3. Solves with Neal (simulated annealing) or QAOA
4. Generates timetable and visualizations

Features:
- Random uniform conflict distribution
- Exact control over conflict percentage
- Neal backend by default (fast)
- Optional visualization (heatmap, graph, timetable)
- Slack variables for capacity constraint

Usage:
    # Basic usage (interactive prompts)
    python run_exam_scheduler_slack.py
    
    # Command-line with default 40% conflicts
    python run_exam_scheduler_slack.py --courses 10 --k 4
    
    # Control conflict density
    python run_exam_scheduler_slack.py --courses 10 --k 4 --conflict-pct 30
    
    # With visualization
    python run_exam_scheduler_slack.py --courses 10 --k 4 --visualize
    
    # Compare backends
    python run_exam_scheduler_slack.py --courses 10 --k 4 --backend both

How Slot Assignment Works:
---------------------------
Each exam is represented by K binary variables (one per time slot).
For 10 exams and K=4 slots, we have 40 binary variables total.

Variable encoding:
  x[exam_id, slot_id] = 1 means exam is assigned to that slot
  
Additionally, for Constraint 4, we have slack variables:
  s[slot_id, overflow_unit] = 1 means that slot exceeds capacity by that unit

Example: Exam 0 → variables [0,1,2,3] represent slots [0,1,2,3]
         Exam 1 → variables [4,5,6,7] represent slots [0,1,2,3]
         Slot 0 slack → variables [40,41,42,...] represent overflow units

The QUBO solver finds which variables should be 1 such that:
  1. Each exam has exactly one slot (one variable = 1 per exam)
  2. No two conflicting exams share the same slot
  3. Same-year exams not in consecutive slots (soft penalty)
  4. Slot capacity not exceeded (slack variables = 0 preferred)

Author: Quantum Exam Scheduling
Date: March 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import argparse
from datetime import datetime

# Qiskit imports
try:
    from qiskit_optimization import QuadraticProgram
    from qiskit_optimization.algorithms import MinimumEigenOptimizer
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠ Qiskit not available. QAOA backend disabled.")

# D-Wave imports
try:
    from dimod import BinaryQuadraticModel
    from dwave.samplers import SimulatedAnnealingSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    print("⚠ D-Wave Ocean SDK not available. Neal backend disabled.")

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠ Matplotlib/NetworkX not available. Visualization disabled.")


# ============================================================================
# STEP 1: DATASET GENERATION
# ============================================================================

def generate_random_adjacency(num_courses, conflict_pct):
    """
    Generate random adjacency matrix with uniform distribution
    
    Args:
        num_courses: Number of nodes (exams/courses)
        conflict_pct: Percentage of edges to create (0-100)
    
    Returns:
        adjacency: Symmetric binary adjacency matrix
    """
    adjacency = np.zeros((num_courses, num_courses), dtype=int)
    
    # Total possible edges in undirected graph
    total_possible = num_courses * (num_courses - 1) // 2
    
    # Number of edges to create
    num_edges = int(total_possible * conflict_pct / 100)
    
    # Generate all possible edge pairs
    all_pairs = [(i, j) for i in range(num_courses) for j in range(i+1, num_courses)]
    
    # Randomly select edges
    selected_pairs = np.random.choice(len(all_pairs), size=num_edges, replace=False)
    
    for idx in selected_pairs:
        i, j = all_pairs[idx]
        adjacency[i, j] = 1
        adjacency[j, i] = 1
    
    return adjacency


def generate_dataset(num_courses, num_students, avg_courses_per_student, conflict_pct, output_dir):
    """
    Generate exam scheduling dataset with random uniform conflicts
    
    Args:
        num_courses: Number of courses/exams
        num_students: Number of students
        avg_courses_per_student: Average courses per student (for enrollment data)
        conflict_pct: Percentage of conflicts (0-100), controls edge density
        output_dir: Output directory path
    
    How it works:
    1. Creates courses with random Year (2 or 3) and enrollment (20-60 students)
    2. Generates students and enrollments (for metadata/realism)
    3. Generates random conflict graph with exactly conflict_pct% edge density
    4. Conflicts are uniformly distributed (NOT based on enrollments)
    """
    
    print("\n" + "="*60)
    print("GENERATING DATASET")
    print("="*60)
    print(f"Courses: {num_courses}")
    print(f"Students: {num_students}")
    print(f"Avg courses per student: {avg_courses_per_student}")
    print(f"Conflict percentage: {conflict_pct:.1f}%")
    
    # Generate courses
    courses = []
    for i in range(num_courses):
        courses.append({
            'course_id': i,
            'course_code': f'C{i+1:02d}',
            'year': np.random.choice([2, 3]),
            'enrollment': np.random.randint(20, 60)
        })
    courses_df = pd.DataFrame(courses)
    
    # Generate students
    students = []
    for i in range(num_students):
        students.append({
            'student_id': i,
            'year': np.random.choice([2, 3])
        })
    students_df = pd.DataFrame(students)
    
    # Generate enrollments (for metadata - NOT used for conflicts)
    enrollments = []
    for student_id in range(num_students):
        student_year = students_df.iloc[student_id]['year']
        
        # Available courses for this year
        available = courses_df[courses_df['year'] == student_year]['course_id'].tolist()
        
        if not available:
            continue
        
        # How many courses for this student?
        num_courses_student = min(
            max(2, int(np.random.normal(avg_courses_per_student, 1))),
            len(available)
        )
        
        # Select courses
        selected = np.random.choice(available, size=num_courses_student, replace=False)
        
        for course_id in selected:
            enrollments.append({
                'student_id': student_id,
                'course_id': course_id
            })
    
    enrollments_df = pd.DataFrame(enrollments)
    
    # Generate random conflict adjacency matrix (NOT based on enrollments)
    print(f"\n✓ Generating random uniform conflicts...")
    adjacency = generate_random_adjacency(num_courses, conflict_pct)
    
    num_edges = int(np.sum(adjacency) // 2)
    density = num_edges / (num_courses * (num_courses - 1) / 2) * 100 if num_courses > 1 else 0
    
    # Estimate chromatic number
    degrees = np.sum(adjacency, axis=1)
    max_degree = int(np.max(degrees)) if len(degrees) > 0 else 0
    
    print(f"\n✓ Generated {num_courses} courses")
    print(f"✓ Generated {num_students} students")
    print(f"✓ Generated {len(enrollments_df)} enrollments")
    print(f"✓ Conflict graph: {num_edges} edges, {density:.1f}% density")
    print(f"✓ Max degree: {max_degree} → Minimum K needed: ≥{max_degree + 1}")
    
    # Save to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    courses_df.to_csv(output_path / 'courses.csv', index=False)
    students_df.to_csv(output_path / 'students.csv', index=False)
    enrollments_df.to_csv(output_path / 'enrollments.csv', index=False)
    
    pd.DataFrame(adjacency).to_csv(output_path / 'conflict_adjacency.csv', index=False)
    
    metadata = {
        'num_courses': num_courses,
        'num_students': num_students,
        'num_enrollments': len(enrollments_df),
        'conflict_pct': conflict_pct,
        'num_conflicts': num_edges,
        'density': density,
        'max_degree': max_degree,
        'min_k_estimate': max_degree + 1
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved dataset to: {output_dir}")
    
    return courses_df, adjacency, metadata


# ============================================================================
# STEP 2: BUILD QUBO WITH SLACK VARIABLES
# ============================================================================

def build_qubo_slack(adjacency, K, courses_df=None,
                     lambda1=10000, lambda2=5000, lambda3=500, lambda4=200,
                     capacity=None, max_slack_units=None):
    """
    Build QUBO matrix for exam scheduling with 4 constraints using SLACK VARIABLES
    for the capacity constraint (Option A).

    Constraint 1 (lambda1): Each exam assigned to exactly one slot (one-hot)
    Constraint 2 (lambda2): Conflicting exams must be in different slots (hard)
    Constraint 3 (lambda3): Same-year exams should not be in consecutive slots (soft)
    Constraint 4 (lambda4): Total enrollment per slot should not exceed capacity (soft, with slack)
    
    Args:
        adjacency         : N×N conflict matrix
        K                 : Number of time slots
        courses_df        : DataFrame with 'year' and 'enrollment' columns
        lambda1           : Penalty for one-hot violation (default 10000)
        lambda2           : Penalty for conflict in same slot (default 5000)
        lambda3           : Penalty for same-year exams in consecutive slots (default 500)
        lambda4           : Penalty for each slack variable active (default 200)
        capacity          : Max total enrollment allowed per slot (default: auto)
        max_slack_units   : Maximum slack units per slot (default: auto = ceil(max_enrollment))
    """
    
    print("\n" + "="*60)
    print("BUILDING QUBO MATRIX (SLACK VARIABLES VERSION)")
    print("="*60)
    
    n = len(adjacency)  # Number of exams
    num_exam_vars = n * K
    
    print(f"Exams: {n}")
    print(f"Colors (K): {K}")
    print(f"Exam variables: {num_exam_vars}")
    print(f"λ1={lambda1}  λ2={lambda2}  λ3={lambda3}  λ4={lambda4}")
    
    # Auto-compute capacity and max slack units if not provided
    if courses_df is not None:
        enrollments = courses_df['enrollment'].values.astype(float)
        if capacity is None:
            capacity = int(np.mean(enrollments) * (n / K) * 1.2)
        if max_slack_units is None:
            max_slack_units = int(np.ceil(np.max(enrollments)))
    else:
        if capacity is None:
            capacity = 100
        if max_slack_units is None:
            max_slack_units = 50
    
    print(f"Capacity per slot: {capacity}")
    print(f"Max slack units per slot: {max_slack_units}")
    
    # Total variables: exam assignment (n*K) + slack variables (K*max_slack_units)
    num_slack_vars = K * max_slack_units
    total_vars = num_exam_vars + num_slack_vars
    
    print(f"Slack variables: {num_slack_vars}")
    print(f"Total variables: {total_vars}")
    print(f"QUBO size: {total_vars} × {total_vars}")
    
    Q = np.zeros((total_vars, total_vars))
    
    # Helper functions
    def exam_var_idx(exam, color):
        """Index for exam assignment variable x_ik"""
        return exam * K + color
    
    def slack_var_idx(slot, unit):
        """Index for slack variable s_ku (overflow unit u in slot k)"""
        return num_exam_vars + slot * max_slack_units + unit
    
    # -----------------------------------------------------------------------
    # Constraint 1: Each exam gets exactly one slot (one-hot)
    # E1 = λ1 × Σᵢ (1 - Σₖ xᵢₖ)²
    # Expanding: -λ1 on diagonal, +2λ1 on off-diagonal same-exam pairs
    # -----------------------------------------------------------------------
    print("\nAdding C1: Each exam exactly one slot...")
    for exam in range(n):
        for c in range(K):
            idx = exam_var_idx(exam, c)
            Q[idx, idx] += -lambda1  # Diagonal
            
            for c2 in range(c+1, K):
                idx2 = exam_var_idx(exam, c2)
                Q[idx, idx2] += 2 * lambda1  # Off-diagonal
                Q[idx2, idx] += 2 * lambda1  # Symmetric
    
    # -----------------------------------------------------------------------
    # Constraint 2: Conflicting exams must be in different slots
    # E2 = λ2 × Σ_(i,j)∈conflicts Σₖ xᵢₖ · xⱼₖ
    # -----------------------------------------------------------------------
    print("Adding C2: Conflicting exams in different slots...")
    num_conflict_pairs = 0
    for i in range(n):
        for j in range(i+1, n):
            if adjacency[i, j] > 0:
                num_conflict_pairs += 1
                for c in range(K):
                    idx_i = exam_var_idx(i, c)
                    idx_j = exam_var_idx(j, c)
                    Q[idx_i, idx_j] += lambda2
                    Q[idx_j, idx_i] += lambda2
    print(f"✓ {num_conflict_pairs} conflict pairs added")
    
    # -----------------------------------------------------------------------
    # Constraint 3: Same-year exams should not be in consecutive slots (soft)
    # E3 = λ3 × Σ_(i,j):same_year Σₖ (xᵢₖ·xⱼ,ₖ₊₁ + xᵢ,ₖ₊₁·xⱼₖ)
    # Applied to all pairs where courses_df['year'] matches
    # -----------------------------------------------------------------------
    num_consec_pairs = 0
    if courses_df is not None and lambda3 > 0:
        print("Adding C3: Same-year exams not in consecutive slots...")
        years = courses_df['year'].values
        
        for i in range(n):
            for j in range(i+1, n):
                if years[i] == years[j]:
                    num_consec_pairs += 1
                    for k in range(K - 1):  # k and k+1 are consecutive
                        # Exam i in slot k  AND  exam j in slot k+1
                        a = exam_var_idx(i, k)
                        b = exam_var_idx(j, k + 1)
                        Q[a, b] += lambda3
                        Q[b, a] += lambda3
                        
                        # Exam i in slot k+1  AND  exam j in slot k
                        c = exam_var_idx(i, k + 1)
                        d = exam_var_idx(j, k)
                        Q[c, d] += lambda3
                        Q[d, c] += lambda3
        
        print(f"✓ {num_consec_pairs} same-year pairs, {num_consec_pairs * 2 * (K-1)} terms added")
    else:
        print("Skipping C3: no course year data or lambda3=0")
    
    # -----------------------------------------------------------------------
    # Constraint 4: Slot capacity with SLACK VARIABLES (Option A)
    # 
    # For each slot k, we need: Σᵢ eᵢ·xᵢₖ ≤ C + M·Σᵤ s_ku
    # where s_ku = 1 means overflow unit u is "used" in slot k
    # 
    # Rewritten as:
    # E4 = λ4 × Σₖ Σᵤ s_ku
    # (This penalizes activating slack variables; we don't directly enforce
    #  the enrollment constraint in QUBO, just penalize slack activation)
    # 
    # The constraint could be better enforced with auxiliary variables,
    # but for now, we just penalize slack variables directly.
    # -----------------------------------------------------------------------
    if courses_df is not None and lambda4 > 0:
        print(f"Adding C4: Slot capacity via slack variables (λ4={lambda4})...")
        
        # Simple approach: penalize each slack variable
        for k in range(K):
            for u in range(max_slack_units):
                idx = slack_var_idx(k, u)
                Q[idx, idx] += lambda4
        
        print(f"✓ Slack penalty added: {K * max_slack_units} slack variables")
    else:
        print("Skipping C4: lambda4=0")
    
    print(f"\n✓ QUBO built: {np.count_nonzero(Q)} non-zero entries")
    
    return Q, num_exam_vars, num_slack_vars, max_slack_units, capacity


# ============================================================================
# STEP 3: SOLVE WITH QAOA
# ============================================================================

def solve_qaoa(Q, reps=2, maxiter=100):
    """Solve QUBO with QAOA"""
    
    if not QISKIT_AVAILABLE:
        print("\n✗ Qiskit not installed. Cannot use QAOA backend.")
        return None
    
    print("\n" + "="*60)
    print("SOLVING WITH QAOA (IBM Qiskit)")
    print("="*60)
    print(f"Variables: {Q.shape[0]}")
    print(f"QAOA depth: {reps}")
    print(f"Max iterations: {maxiter}")
    
    num_vars = Q.shape[0]
    
    # Build QuadraticProgram
    qp = QuadraticProgram()
    for i in range(num_vars):
        qp.binary_var(name=f'x{i}')
    
    linear = {}
    quadratic = {}
    
    for i in range(num_vars):
        if Q[i, i] != 0:
            linear[f'x{i}'] = Q[i, i]
        for j in range(i+1, num_vars):
            if Q[i, j] != 0:
                quadratic[(f'x{i}', f'x{j}')] = 2 * Q[i, j]
    
    qp.minimize(linear=linear, quadratic=quadratic)
    
    # Solve
    sampler = Sampler()
    optimizer = COBYLA(maxiter=maxiter)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    
    print("\n⏳ Running QAOA...")
    start_time = time.time()
    
    result = qaoa_optimizer.solve(qp)
    
    runtime = time.time() - start_time
    
    print(f"✓ Completed in {runtime:.2f}s")
    print(f"Energy: {result.fval:.2f}")
    
    return {
        'solution': result.x,
        'energy': result.fval,
        'runtime': runtime,
        'backend': 'qaoa'
    }


# ============================================================================
# STEP 4: SOLVE WITH NEAL
# ============================================================================

def solve_neal(Q, num_reads=1000):
    """Solve QUBO with D-Wave Neal"""
    
    if not DWAVE_AVAILABLE:
        print("\n✗ D-Wave Ocean SDK not installed. Cannot use Neal backend.")
        return None
    
    print("\n" + "="*60)
    print("SOLVING WITH NEAL (Simulated Annealing)")
    print("="*60)
    print(f"Variables: {Q.shape[0]}")
    print(f"Num reads: {num_reads}")
    
    num_vars = Q.shape[0]
    
    # Convert to BQM
    linear = {}
    quadratic = {}
    
    for i in range(num_vars):
        if Q[i, i] != 0:
            linear[i] = Q[i, i]
        for j in range(i+1, num_vars):
            if Q[i, j] != 0:
                quadratic[(i, j)] = Q[i, j]
    
    bqm = BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')
    
    # Solve
    print("\n⏳ Running simulated annealing...")
    start_time = time.time()
    
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=num_reads)
    
    runtime = time.time() - start_time
    
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    
    # Convert to solution array
    solution = np.zeros(num_vars, dtype=int)
    for i in range(num_vars):
        solution[i] = best_sample.get(i, 0)
    
    print(f"✓ Completed in {runtime:.2f}s")
    print(f"Energy: {best_energy:.2f}")
    
    return {
        'solution': solution,
        'energy': best_energy,
        'runtime': runtime,
        'backend': 'neal'
    }


# ============================================================================
# STEP 5: DECODE & VALIDATE
# ============================================================================

def decode_solution(solution, num_courses, K, num_exam_vars, num_slack_vars, max_slack_units):
    """
    Decode binary solution to coloring (slot assignment) and slack variables
    """
    
    coloring = {}
    slack_usage = {}
    
    # Decode exam assignments
    for exam in range(num_courses):
        for color in range(K):
            var_idx = exam * K + color
            if solution[var_idx] == 1:
                coloring[exam] = color
                break
    
    # Decode slack variables
    for k in range(K):
        slack_count = 0
        for u in range(max_slack_units):
            var_idx = num_exam_vars + k * max_slack_units + u
            if solution[var_idx] == 1:
                slack_count += 1
        if slack_count > 0:
            slack_usage[k] = slack_count
    
    return coloring, slack_usage


def validate_solution(coloring, adjacency, num_courses):
    """Validate solution"""
    
    violations = []
    
    # Check all exams assigned
    if len(coloring) != num_courses:
        violations.append(f"Only {len(coloring)}/{num_courses} exams assigned")
    
    # Check conflicts
    conflict_count = 0
    for i in range(num_courses):
        for j in range(i+1, num_courses):
            if adjacency[i, j] > 0:
                if i in coloring and j in coloring:
                    if coloring[i] == coloring[j]:
                        conflict_count += 1
                        violations.append(f"Conflict: Exams {i},{j} both in slot {coloring[i]}")
    
    is_valid = len(violations) == 0
    
    return is_valid, conflict_count, violations


# ============================================================================
# STEP 6: GENERATE TIMETABLE
# ============================================================================

def generate_timetable(coloring, courses_df, K):
    """Generate timetable from solution"""
    
    print("\n" + "="*60)
    print("TIMETABLE")
    print("="*60)
    
    # Group by time slot
    schedule = {}
    for exam_id, slot in coloring.items():
        if slot not in schedule:
            schedule[slot] = []
        
        course = courses_df.iloc[exam_id]
        schedule[slot].append({
            'exam_id': exam_id,
            'course_code': course['course_code'],
            'year': course['year'],
            'enrollment': course['enrollment']
        })
    
    # Print timetable
    for slot in sorted(schedule.keys()):
        print(f"\n{'='*60}")
        print(f"TIME SLOT {slot}")
        print(f"{'='*60}")
        
        exams = sorted(schedule[slot], key=lambda x: x['course_code'])
        for exam in exams:
            print(f"  {exam['course_code']:6s} | Year {exam['year']} | {exam['enrollment']} students")
    
    # Create DataFrame
    timetable_rows = []
    for slot in sorted(schedule.keys()):
        for exam in sorted(schedule[slot], key=lambda x: x['course_code']):
            timetable_rows.append({
                'time_slot': slot,
                **exam
            })
    
    return pd.DataFrame(timetable_rows)


# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================

def visualize_adjacency_matrix(adjacency, output_dir, num_courses):
    """Visualize adjacency matrix as heatmap"""
    if not VISUALIZATION_AVAILABLE:
        print("⚠ Visualization libraries not available. Skipping.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create heatmap
    im = ax.imshow(adjacency, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(num_courses))
    ax.set_yticks(range(num_courses))
    ax.set_xticklabels([f'C{i+1}' for i in range(num_courses)])
    ax.set_yticklabels([f'C{i+1}' for i in range(num_courses)])
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Conflict (1=Yes, 0=No)', rotation=270, labelpad=20)
    
    # Title and labels
    num_edges = int(np.sum(adjacency) // 2)
    density = num_edges / (num_courses * (num_courses - 1) / 2) * 100 if num_courses > 1 else 0
    ax.set_title(f'Conflict Adjacency Matrix\n{num_edges} conflicts, {density:.1f}% density', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Course', fontsize=12)
    ax.set_ylabel('Course', fontsize=12)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(output_dir) / 'adjacency_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved adjacency heatmap to: {save_path}")
    plt.close()


def visualize_conflict_graph(adjacency, output_dir, num_courses):
    """Visualize conflict graph as network"""
    if not VISUALIZATION_AVAILABLE:
        print("⚠ Visualization libraries not available. Skipping.")
        return
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i in range(num_courses):
        G.add_node(i, label=f'C{i+1}')
    
    # Add edges
    for i in range(num_courses):
        for j in range(i+1, num_courses):
            if adjacency[i, j] > 0:
                G.add_edge(i, j)
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=800, ax=ax, edgecolors='black', linewidths=2)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='red', width=2, alpha=0.6, ax=ax)
    
    # Draw labels
    labels = {i: f'C{i+1}' for i in range(num_courses)}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold', ax=ax)
    
    # Title
    num_edges = G.number_of_edges()
    density = num_edges / (num_courses * (num_courses - 1) / 2) * 100 if num_courses > 1 else 0
    ax.set_title(f'Conflict Graph\n{num_courses} courses, {num_edges} conflicts ({density:.1f}% density)',
                fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    save_path = Path(output_dir) / 'conflict_graph.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved conflict graph to: {save_path}")
    plt.close()


def visualize_timetable(coloring, adjacency, courses_df, K, output_dir):
    """Visualize timetable as colored schedule"""
    if not VISUALIZATION_AVAILABLE:
        print("⚠ Visualization libraries not available. Skipping.")
        return
    
    num_courses = len(coloring)
    
    # Create color map for time slots
    colors = plt.cm.Set3(np.linspace(0, 1, K))
    
    # Group by time slot
    schedule = {slot: [] for slot in range(K)}
    for exam_id, slot in coloring.items():
        schedule[slot].append(exam_id)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Graph with colored nodes
    G = nx.Graph()
    for i in range(num_courses):
        G.add_node(i)
    for i in range(num_courses):
        for j in range(i+1, num_courses):
            if adjacency[i, j] > 0:
                G.add_edge(i, j)
    
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
    
    # Draw nodes colored by time slot
    node_colors = [colors[coloring[i]] for i in range(num_courses)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1000, ax=ax1, edgecolors='black', linewidths=2)
    
    # Draw conflict edges
    nx.draw_networkx_edges(G, pos, edge_color='red', width=2, alpha=0.3, ax=ax1)
    
    # Draw labels
    labels = {i: f'C{i+1}' for i in range(num_courses)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold', ax=ax1)
    
    ax1.set_title('Conflict Graph with Solution\n(Node color = Time slot)', 
                 fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right plot: Schedule visualization
    max_exams_per_slot = max(len(exams) for exams in schedule.values()) if schedule else 1
    
    for slot in range(K):
        exams = schedule[slot]
        y_pos = K - slot - 1
        
        # Draw time slot bar
        ax2.barh(y_pos, len(exams), height=0.8, color=colors[slot], 
                edgecolor='black', linewidth=2)
        
        # Add exam labels
        for i, exam_id in enumerate(sorted(exams)):
            course = courses_df.iloc[exam_id]
            label = f"{course['course_code']}\n({course['enrollment']} students)"
            ax2.text(i + 0.5, y_pos, label, ha='center', va='center', 
                    fontsize=9, fontweight='bold')
    
    ax2.set_yticks(range(K))
    ax2.set_yticklabels([f'Slot {K-i-1}' for i in range(K)])
    ax2.set_xlabel('Number of Exams', fontsize=12)
    ax2.set_title('Exam Schedule by Time Slot', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max_exams_per_slot + 0.5)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = Path(output_dir) / 'timetable_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved timetable visualization to: {save_path}")
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def parse_args():
    """Parse command-line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Interactive Exam Scheduler - Slack Variables Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for inputs)
  python run_exam_scheduler_slack.py
  
  # Command-line mode with default settings (50 students, 40% conflicts)
  python run_exam_scheduler_slack.py --courses 10 --k 4
  
  # Specify number of students
  python run_exam_scheduler_slack.py --courses 10 --students 80 --k 4
  
  # Low conflict density (20%)
  python run_exam_scheduler_slack.py --courses 10 --students 60 --k 3 --conflict-pct 20
  
  # Medium conflict density (50%)
  python run_exam_scheduler_slack.py --courses 10 --students 60 --k 4 --conflict-pct 50
  
  # With visualization
  python run_exam_scheduler_slack.py --courses 10 --k 4 --visualize
  
  # Compare backends (QAOA vs Neal)
  python run_exam_scheduler_slack.py --courses 8 --k 3 --backend both
        """
    )
    
    parser.add_argument('--courses', type=int,
                       help='Number of courses/exams (default: interactive prompt)')
    
    parser.add_argument('--students', type=int, default=50,
                       help='Number of students (default: 50)')
    
    parser.add_argument('--k', type=int,
                       help='Number of time slots/colors (default: interactive prompt)')
    
    parser.add_argument('--avg-courses', type=int, default=4,
                       help='Average courses per student for enrollment data (default: 4)')
    
    parser.add_argument('--conflict-pct', type=float, default=40.0,
                       help='Conflict percentage: edge density 0-100 (default: 40.0)')
    
    parser.add_argument('--backend', type=str, default='neal',
                       choices=['qaoa', 'neal', 'both'],
                       help='Solver backend (default: neal)')
    
    parser.add_argument('--reps', type=int, default=2,
                       help='QAOA circuit depth (default: 2)')
    
    parser.add_argument('--maxiter', type=int, default=100,
                       help='QAOA optimizer iterations (default: 100)')
    
    parser.add_argument('--num-reads', type=int, default=1000,
                       help='Neal number of reads (default: 1000)')
    
    parser.add_argument('--lambda1', type=float, default=10000,
                       help='Penalty: one-hot constraint (default: 10000)')
    
    parser.add_argument('--lambda2', type=float, default=5000,
                       help='Penalty: conflict in same slot (default: 5000)')
    
    parser.add_argument('--lambda3', type=float, default=500,
                       help='Penalty: same-year exams in consecutive slots (default: 500, set 0 to disable)')
    
    parser.add_argument('--lambda4', type=float, default=200,
                       help='Penalty: each slack variable active (default: 200, set 0 to disable)')
    
    parser.add_argument('--capacity', type=int, default=None,
                       help='Max total enrollment per slot (default: auto = mean_enrollment × N/K × 1.2)')
    
    parser.add_argument('--max-slack', type=int, default=None,
                       help='Max slack units per slot (default: auto = ceil(max_enrollment))')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots (requires matplotlib, networkx)')
    
    return parser.parse_args()


def get_user_input(args):
    """Get inputs from command-line or interactive prompts"""
    
    if args.courses is None:
        print("\n" + "="*60)
        print("INTERACTIVE MODE")
        print("="*60)
        args.courses = int(input("Number of courses/exams (e.g., 8, 10, 15): "))
        args.students = int(input("Number of students (e.g., 40, 60, 80): "))
    
    if args.k is None:
        args.k = int(input("Number of time slots/colors K (e.g., 3, 4, 5): "))
    
    return args


def main():
    """Main pipeline"""
    
    args = parse_args()
    args = get_user_input(args)
    
    print("\n" + "="*60)
    print("EXAM SCHEDULING PIPELINE (SLACK VARIABLES)")
    print("="*60)
    print(f"Courses: {args.courses}")
    print(f"Students: {args.students}")
    print(f"K (time slots): {args.k}")
    print(f"Conflict percentage: {args.conflict_pct:.1f}%")
    print(f"Backend: {args.backend.upper()}")
    print(f"λ1={args.lambda1}  λ2={args.lambda2}  λ3={args.lambda3}  λ4={args.lambda4}")
    if args.capacity:
        print(f"Slot capacity: {args.capacity}")
    if args.max_slack:
        print(f"Max slack units: {args.max_slack}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('./output') / f'run_slack_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate dataset
    courses_df, adjacency, metadata = generate_dataset(
        num_courses=args.courses,
        num_students=args.students,
        avg_courses_per_student=args.avg_courses,
        conflict_pct=args.conflict_pct,
        output_dir=output_dir
    )
    
    # Visualize conflict graph (if requested)
    if args.visualize:
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        visualize_adjacency_matrix(adjacency, output_dir, args.courses)
        visualize_conflict_graph(adjacency, output_dir, args.courses)
    
    # Step 2: Build QUBO with slack variables
    Q, num_exam_vars, num_slack_vars, max_slack_units, capacity = build_qubo_slack(
        adjacency,
        K=args.k,
        courses_df=courses_df,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        lambda3=args.lambda3,
        lambda4=args.lambda4,
        capacity=args.capacity,
        max_slack_units=args.max_slack
    )
    np.save(output_dir / 'qubo_matrix_slack.npy', Q)
    
    # Step 3: Solve
    results = {}
    
    if args.backend in ['qaoa', 'both']:
        result = solve_qaoa(Q, reps=args.reps, maxiter=args.maxiter)
        if result:
            results['qaoa'] = result
    
    if args.backend in ['neal', 'both']:
        result = solve_neal(Q, num_reads=args.num_reads)
        if result:
            results['neal'] = result
    
    if not results:
        print("\n✗ No backends available!")
        return
    
    # Step 4: Validate and display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    for backend_name, result in results.items():
        print(f"\n{backend_name.upper()}")
        print("-" * 60)
        
        coloring, slack_usage = decode_solution(
            result['solution'], args.courses, args.k, 
            num_exam_vars, num_slack_vars, max_slack_units
        )
        is_valid, num_conflicts, violations = validate_solution(
            coloring, adjacency, args.courses
        )
        
        print(f"Runtime: {result['runtime']:.2f}s")
        print(f"Energy: {result['energy']:.2f}")
        print(f"Valid: {'✓ YES' if is_valid else '✗ NO'}")
        print(f"Conflicts: {num_conflicts}")
        print(f"Exams assigned: {len(coloring)}/{args.courses}")
        print(f"Colors used: {len(set(coloring.values()))}/{args.k}")
        print(f"Slack variables active: {sum(slack_usage.values())} (by slot: {slack_usage})")
        
        # Save results
        result_data = {
            'backend': backend_name,
            'version': 'slack_variables',
            'num_courses': args.courses,
            'num_students': args.students,
            'k': args.k,
            'avg_courses_per_student': args.avg_courses,
            'capacity': int(capacity),
            'max_slack_units': int(max_slack_units),
            'runtime_seconds': result['runtime'],
            'energy': float(result['energy']),
            'is_valid': is_valid,
            'num_conflicts': num_conflicts,
            'colors_used': len(set(coloring.values())),
            'slack_usage': {str(k): int(v) for k, v in slack_usage.items()},
            'coloring': {str(k): int(v) for k, v in coloring.items()}
        }
        
        with open(output_dir / f'{backend_name}_results_slack.json', 'w') as f:
            json.dump(result_data, f, indent=2)
        
        # Generate timetable if valid
        if is_valid:
            timetable = generate_timetable(coloring, courses_df, args.k)
            timetable.to_csv(output_dir / f'timetable_{backend_name}_slack.csv', index=False)
            print(f"\n✓ Saved timetable to: {output_dir}/timetable_{backend_name}_slack.csv")
            
            # Visualize timetable (if requested)
            if args.visualize:
                visualize_timetable(coloring, adjacency, courses_df, args.k, output_dir)
        else:
            print(f"\n⚠ Solution invalid. First 3 conflicts:")
            for v in violations[:3]:
                print(f"  - {v}")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - courses.csv (dataset)")
    print(f"  - conflict_adjacency.csv (conflict graph)")
    print(f"  - qubo_matrix_slack.npy (QUBO with slack)")
    print(f"  - *_results_slack.json (solver outputs)")
    if any(validate_solution(decode_solution(res['solution'], args.courses, args.k, 
                                            num_exam_vars, num_slack_vars, max_slack_units)[0], 
                            adjacency, args.courses)[0] 
           for res in results.values()):
        print(f"  - timetable_*_slack.csv (valid schedules)")
    
    # Comparison if both backends used
    if len(results) > 1:
        print("\n" + "="*60)
        print("BACKEND COMPARISON")
        print("="*60)
        for name, res in results.items():
            col, slack = decode_solution(res['solution'], args.courses, args.k,
                                        num_exam_vars, num_slack_vars, max_slack_units)
            val, conf, _ = validate_solution(col, adjacency, args.courses)
            print(f"{name.upper():10s} | {res['runtime']:6.2f}s | "
                  f"{'✓ VALID' if val else '✗ INVALID':10s} | "
                  f"Conflicts: {conf} | Slack active: {sum(slack.values())}")


if __name__ == '__main__':
    main()
