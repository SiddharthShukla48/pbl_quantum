"""
Interactive Exam Scheduler - Simplified Random Conflict Version
================================================================

Single script that:
1. Generates random conflict graph with controlled density
2. Builds QUBO matrix for graph coloring problem
3. Solves with Neal (simulated annealing)
4. Generates timetable and visualizations

Features:
- Random uniform conflict distribution
- Exact control over conflict percentage
- Neal backend by default (fast)
- Optional visualization (heatmap, graph, timetable)

Usage:
    # Basic usage (interactive prompts)
    python run_exam_scheduler.py
    
    # Command-line with default 40% conflicts
    python run_exam_scheduler.py --courses 10 --k 4
    
    # Control conflict density
    python run_exam_scheduler.py --courses 10 --k 4 --conflict-pct 30
    
    # With visualization
    python run_exam_scheduler.py --courses 10 --k 4 --visualize
    
    # Neal backend (only backend)
    python run_exam_scheduler.py --courses 10 --k 4

How Slot Assignment Works:
---------------------------
Each exam is represented by K binary variables (one per time slot).
For 10 exams and K=4 slots, we have 40 binary variables total.

Variable encoding:
  x[exam_id, slot_id] = 1 means exam is assigned to that slot
  
Example: Exam 0 → variables [0,1,2,3] represent slots [0,1,2,3]
         Exam 1 → variables [4,5,6,7] represent slots [0,1,2,3]
         Exam 2 → variables [8,9,10,11] represent slots [0,1,2,3]

The QUBO solver finds which variables should be 1 such that:
  1. Each exam has exactly one slot (one variable = 1 per exam)
  2. No two conflicting exams share the same slot

Author: Quantum Exam Scheduling
Date: February 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import argparse
import itertools
from datetime import datetime

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
    if conflict_pct < 0 or conflict_pct > 100:
        raise ValueError(f"conflict_pct must be in [0, 100], got {conflict_pct}")

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


def _semester_to_numeric(series):
    """Convert semester labels (I..X) to numeric values where possible."""
    mapping = {
        'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
        'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10
    }
    mapped = series.astype(str).str.strip().map(mapping)
    return pd.to_numeric(mapped, errors='coerce')


def _build_courses_and_adjacency_from_rows(filtered_df, label):
    """
    Build courses table and conflict adjacency from enrollment rows.

    Conflict rule: two courses conflict if at least one student is enrolled in both.
    """
    if filtered_df.empty:
        raise ValueError(f"No rows available after filtering for mode '{label}'.")

    course_col = 'Course Code'
    student_col = 'Registration No.'

    # Build canonical course list
    course_codes = sorted(filtered_df[course_col].astype(str).str.strip().unique().tolist())
    course_to_id = {code: idx for idx, code in enumerate(course_codes)}

    # Enrollment and semester summaries per course
    sem_numeric = _semester_to_numeric(filtered_df['Semester'])
    tmp = filtered_df.copy()
    tmp['semester_num'] = sem_numeric

    enroll_count = tmp.groupby(course_col)[student_col].nunique().to_dict()
    desc_mode = tmp.groupby(course_col)['Description'].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]
    ).to_dict()

    semester_mode = tmp.groupby(course_col)['semester_num'].agg(
        lambda s: int(s.mode().iloc[0]) if s.notna().any() and not s.mode().empty else 1
    ).to_dict()

    courses = []
    for code in course_codes:
        courses.append({
            'course_id': course_to_id[code],
            'course_code': code,
            'description': desc_mode.get(code, code),
            # Reuse existing downstream field name expected by C3.
            'year': int(semester_mode.get(code, 1)),
            # Reuse existing downstream field name expected by C4.
            'enrollment': int(enroll_count.get(code, 0))
        })
    courses_df = pd.DataFrame(courses).sort_values('course_id').reset_index(drop=True)

    # Build adjacency from co-enrollment
    n = len(course_codes)
    adjacency = np.zeros((n, n), dtype=int)
    enrollments = []

    grouped = tmp.groupby(student_col)[course_col].apply(lambda s: sorted(set(s.tolist())))
    for student_id, courses_for_student in grouped.items():
        for course_code in courses_for_student:
            enrollments.append({
                'student_id': student_id,
                'course_id': course_to_id[course_code]
            })

        for c1, c2 in itertools.combinations(courses_for_student, 2):
            i = course_to_id[c1]
            j = course_to_id[c2]
            adjacency[i, j] = 1
            adjacency[j, i] = 1

    students_df = pd.DataFrame({'student_id': sorted(tmp[student_col].unique().tolist())})
    enrollments_df = pd.DataFrame(enrollments)

    num_edges = int(np.sum(adjacency) // 2)
    density = num_edges / (n * (n - 1) / 2) * 100 if n > 1 else 0
    degrees = np.sum(adjacency, axis=1)
    max_degree = int(np.max(degrees)) if len(degrees) > 0 else 0

    meta = {
        'label': label,
        'num_courses': int(n),
        'num_students': int(students_df['student_id'].nunique()),
        'num_enrollments': int(len(enrollments_df)),
        'num_conflicts': int(num_edges),
        'density': float(density),
        'max_degree': int(max_degree),
        'min_k_estimate': int(max_degree + 1)
    }

    return courses_df, students_df, enrollments_df, adjacency, meta


def generate_dataset_from_csv(input_csv, output_dir, adjacency_mode='all', max_rows=None):
    """
    Generate dataset from university CSV and build two adjacency matrices:
    1) major-only (Course Type == MAJOR)
    2) all-theory (MAJOR + ELECTIVE + OPEN ELECTIVE and any other theory rows)

    Args:
        input_csv: Source enrollment CSV path
        output_dir: Output directory
        adjacency_mode: Which graph to use in solver ('major' or 'all')
        max_rows: Optional row limit to load from CSV before applying filters
    """
    print("\n" + "="*60)
    print("GENERATING DATASET FROM CSV")
    print("="*60)
    print(f"Input CSV: {input_csv}")
    print(f"Adjacency mode for solver: {adjacency_mode}")
    if max_rows is not None:
        print(f"Row limit: first {max_rows} rows")

    src = pd.read_csv(input_csv)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("--max-rows must be a positive integer")
        src = src.head(max_rows).copy()

    required = [
        'Registration No.', 'Course Code', 'Academic Session', 'Registration Status',
        'Course Classification', 'Course Type', 'Semester', 'Description'
    ]
    missing = [c for c in required if c not in src.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    for col in required:
        src[col] = src[col].astype(str).str.strip()

    # Apply user-requested filters: Approved + Theory + both sessions.
    base = src[
        (src['Registration Status'].str.upper() == 'APPROVED')
        & (src['Course Classification'].str.upper() == 'THEORY')
        & (src['Academic Session'].isin(['JUL-NOV 2025', 'WINTER 2025']))
    ].copy()

    # Exclude Semester I and II students as requested.
    base = base[~base['Semester'].isin(['I', 'II'])].copy()

    if base.empty:
        raise ValueError("No rows left after filters (Approved + Theory + JUL-NOV/WINTER).")

    major_rows = base[base['Course Type'].str.upper() == 'MAJOR'].copy()
    all_rows = base.copy()

    if major_rows.empty:
        raise ValueError("No MAJOR rows left after filtering.")

    major_courses, major_students, major_enrollments, major_adj, major_meta = _build_courses_and_adjacency_from_rows(
        major_rows, 'major'
    )
    all_courses, all_students, all_enrollments, all_adj, all_meta = _build_courses_and_adjacency_from_rows(
        all_rows, 'all'
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save both matrices for analysis/comparison.
    pd.DataFrame(
        major_adj,
        index=major_courses['course_code'],
        columns=major_courses['course_code']
    ).to_csv(output_path / 'conflict_adjacency_major.csv')

    pd.DataFrame(
        all_adj,
        index=all_courses['course_code'],
        columns=all_courses['course_code']
    ).to_csv(output_path / 'conflict_adjacency_all.csv')

    # Save selected working set used by solver.
    if adjacency_mode == 'major':
        selected_courses = major_courses
        selected_students = major_students
        selected_enrollments = major_enrollments
        selected_adjacency = major_adj
        selected_meta = major_meta
    else:
        selected_courses = all_courses
        selected_students = all_students
        selected_enrollments = all_enrollments
        selected_adjacency = all_adj
        selected_meta = all_meta

    selected_courses.to_csv(output_path / 'courses.csv', index=False)
    selected_students.to_csv(output_path / 'students.csv', index=False)
    selected_enrollments.to_csv(output_path / 'enrollments.csv', index=False)
    pd.DataFrame(selected_adjacency).to_csv(output_path / 'conflict_adjacency.csv', index=False)

    metadata = {
        'source': 'csv',
        'input_csv': str(input_csv),
        'max_rows': int(max_rows) if max_rows is not None else None,
        'filters': {
            'registration_status': 'Approved',
            'course_classification': 'Theory',
            'academic_sessions': ['JUL-NOV 2025', 'WINTER 2025'],
            'excluded_semesters': ['I', 'II']
        },
        'adjacency_mode_selected': adjacency_mode,
        'major_graph': major_meta,
        'all_graph': all_meta,
        'selected_graph': selected_meta
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Rows after filters (all): {len(all_rows)}")
    print(f"✓ Rows after filters (major): {len(major_rows)}")
    print(
        f"✓ Major graph: {major_meta['num_courses']} courses, "
        f"{major_meta['num_conflicts']} edges ({major_meta['density']:.2f}% density)"
    )
    print(
        f"✓ All graph: {all_meta['num_courses']} courses, "
        f"{all_meta['num_conflicts']} edges ({all_meta['density']:.2f}% density)"
    )
    print(f"✓ Selected '{adjacency_mode}' graph for solver")
    print(f"✓ Saved dataset to: {output_dir}")

    return selected_courses, selected_adjacency, metadata


# ============================================================================
# STEP 2: BUILD QUBO
# ============================================================================

def build_qubo(adjacency, K, courses_df=None,
               lambda1=10000, lambda2=5000, lambda3=500, lambda4=200,
               capacity=None):
    """
    Build QUBO matrix for exam scheduling with 4 constraints:

    Constraint 1 (lambda1): Each exam assigned to exactly one slot (one-hot)
    Constraint 2 (lambda2): Conflicting exams must be in different slots (hard)
    Constraint 3 (lambda3): Same-year exams should not be in consecutive slots (soft)
    Constraint 4 (lambda4): Total enrollment per slot should not exceed capacity (soft)
                            Uses binary-weighted slack bits for efficient overflow encoding
    
    Args:
        adjacency   : N×N conflict matrix
        K           : Number of time slots
        courses_df  : DataFrame with 'year' and 'enrollment' columns
        lambda1     : Penalty for one-hot violation (default 10000)
        lambda2     : Penalty for conflict in same slot (default 5000)
        lambda3     : Penalty for same-year exams in consecutive slots (default 500)
        lambda4     : Penalty for exceeding slot capacity (default 200)
        capacity    : Room capacity (max total enrollment per slot). REQUIRED for C4.
    """
    
    print("\n" + "="*60)
    print("BUILDING QUBO MATRIX")
    print("="*60)
    
    n = len(adjacency)  # Number of exams
    
    # Determine number of slack bits needed for C4
    enrollments = courses_df['enrollment'].values.astype(float) if courses_df is not None else np.ones(n)
    
    if capacity is None:
        # Default: auto-compute based on mean per-slot load
        capacity = int(np.mean(enrollments) * (n / K) * 1.2)
    
    # Number of bits to represent slack in (L_k + S_k - C)^2.
    # Here S_k models residual to reach capacity, so bound is capacity itself.
    max_slack_value = max(0, int(capacity))
    num_slack_bits = max(1, int(np.ceil(np.log2(max_slack_value + 1)))) if max_slack_value > 0 else 1
    
    # Total variables: n*K exam variables + K*num_slack_bits slack variables
    num_exam_vars = n * K
    num_slack_vars = K * num_slack_bits
    num_vars = num_exam_vars + num_slack_vars
    
    print(f"Exams: {n}")
    print(f"Colors (K): {K}")
    print(f"Exam variables: {num_exam_vars}")
    print(f"Slack bits per slot: {num_slack_bits} (max slack value: {max_slack_value})")
    print(f"Slack variables: {num_slack_vars}")
    print(f"Total variables: {num_vars}")
    print(f"QUBO size: {num_vars} × {num_vars}")
    print(f"Room capacity: {capacity}")
    print(f"λ1={lambda1}  λ2={lambda2}  λ3={lambda3}  λ4={lambda4}")
    
    Q = np.zeros((num_vars, num_vars))
    
    # Helper function: variable index for exam i in slot k
    def var_idx(exam, color):
        return exam * K + color
    
    # Helper function: variable index for slack bit b in slot k
    def slack_idx(slot, bit):
        return num_exam_vars + slot * num_slack_bits + bit
    
    # -----------------------------------------------------------------------
    # Constraint 1: Each exam gets exactly one slot (one-hot)
    # E1 = λ1 × Σᵢ (1 - Σₖ xᵢₖ)²
    # Expanding: -λ1 on diagonal, +2λ1 on off-diagonal same-exam pairs
    # -----------------------------------------------------------------------
    print("\nAdding C1: Each exam exactly one slot...")
    for exam in range(n):
        for c in range(K):
            idx = var_idx(exam, c)
            Q[idx, idx] += -lambda1  # Diagonal
            
            for c2 in range(c+1, K):
                idx2 = var_idx(exam, c2)
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
                    idx_i = var_idx(i, c)
                    idx_j = var_idx(j, c)
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
                        a = var_idx(i, k)
                        b = var_idx(j, k + 1)
                        Q[a, b] += lambda3
                        Q[b, a] += lambda3
                        
                        # Exam i in slot k+1  AND  exam j in slot k
                        c = var_idx(i, k + 1)
                        d = var_idx(j, k)
                        Q[c, d] += lambda3
                        Q[d, c] += lambda3
        
        print(f"✓ {num_consec_pairs} same-year pairs, {num_consec_pairs * 2 * (K-1)} terms added")
    else:
        print("Skipping C3: no course year data or lambda3=0")
    
    
    # -----------------------------------------------------------------------
    # Constraint 4: Slot capacity with binary-weighted slack bits (soft)
    # E4 = λ4 × Σₖ (Σᵢ eᵢ·xᵢₖ + Σᵦ 2ᵇ·sₖᵦ - C)²
    #
    # Where:
    #   eᵢ = enrollment of course i
    #   xᵢₖ = binary var for course i in slot k
    #   sₖᵦ = binary slack var for slot k, bit b (weight = 2^b)
    #   C = room capacity
    #
    # Slack bits use binary (exponential) weights: 1, 2, 4, 8, ..., 2^(num_slack_bits-1)
    # This efficiently encodes residual-capacity slack values using minimal variables
    #
    # Expansion of (L_k + S_k - C)²:
    #   L_k = Σᵢ eᵢ·xᵢₖ  (load from course assignments)
    #   S_k = Σᵦ 2ᵇ·sₖᵦ  (slack bits with exponential weights)
    #   E4 = λ4 × Σₖ [L_k² + S_k² + C² + 2·L_k·S_k - 2·C·L_k - 2·C·S_k]
    #
    # QUBO coefficients:
    #   1. Exam diagonal:     λ4·(eᵢ² - 2·C·eᵢ)
    #   2. Exam off-diagonal (same slot): 2·λ4·eᵢ·eⱼ
    #   3. Slack diagonal:    λ4·(2^(2b) - 2·C·2ᵇ)
    #   4. Slack off-diagonal (same slot): 2·λ4·2ᵇ·2ᵇ' = λ4·2^(b+b'+1)
    #   5. Cross terms (exam-slack, same slot): 2·λ4·eᵢ·2ᵇ
    # -----------------------------------------------------------------------
    if courses_df is not None and lambda4 > 0:
        enrollments = courses_df['enrollment'].values.astype(float)
        
        print(f"Adding C4: Slot capacity ≤ {capacity} with binary-weighted slack bits...")
        print(f"           {num_slack_bits} slack bits per slot × {K} slots = {num_slack_vars} slack variables")
        
        for k in range(K):
            # 1. Exam variable diagonal contributions: λ4·(eᵢ² - 2·C·eᵢ)
            for i in range(n):
                idx_i = var_idx(i, k)
                ei = enrollments[i]
                Q[idx_i, idx_i] += lambda4 * (ei**2 - 2 * capacity * ei)
            
            # 2. Exam off-diagonal contributions (same slot): 2·λ4·eᵢ·eⱼ
            for i in range(n):
                for j in range(i+1, n):
                    idx_i = var_idx(i, k)
                    idx_j = var_idx(j, k)
                    ei = enrollments[i]
                    ej = enrollments[j]
                    Q[idx_i, idx_j] += 2 * lambda4 * ei * ej
                    Q[idx_j, idx_i] += 2 * lambda4 * ei * ej
            
            # 3. Slack variable diagonal contributions: λ4·(2^(2b) - 2·C·2ᵇ) = λ4·2ᵇ·(2ᵇ - 2·C)
            for b in range(num_slack_bits):
                idx_slack = slack_idx(k, b)
                weight = 2**b
                Q[idx_slack, idx_slack] += lambda4 * weight * (weight - 2 * capacity)
            
            # 4. Slack off-diagonal contributions (same slot): λ4·2^(b+b'+1)
            for b in range(num_slack_bits):
                for b_prime in range(b+1, num_slack_bits):
                    idx_b = slack_idx(k, b)
                    idx_bp = slack_idx(k, b_prime)
                    weight = 2**(b + b_prime + 1)
                    Q[idx_b, idx_bp] += lambda4 * weight
                    Q[idx_bp, idx_b] += lambda4 * weight
            
            # 5. Cross terms (exam-slack, same slot): 2·λ4·eᵢ·2ᵇ
            for i in range(n):
                idx_i = var_idx(i, k)
                ei = enrollments[i]
                for b in range(num_slack_bits):
                    idx_slack = slack_idx(k, b)
                    weight_b = 2**b
                    Q[idx_i, idx_slack] += 2 * lambda4 * ei * weight_b
                    Q[idx_slack, idx_i] += 2 * lambda4 * ei * weight_b
        
        print(f"✓ Capacity constraint with binary slack bits added (C={capacity})")
    else:
        print("Skipping C4: no enrollment data or lambda4=0")
    
    
    print(f"\n✓ QUBO built: {np.count_nonzero(Q)} non-zero entries")
    
    return Q


# ============================================================================
# STEP 3: SOLVE WITH NEAL
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

def decode_solution(solution, num_courses, K, num_slack_bits=None):
    """
    Decode binary solution to slot assignment, extracting only exam variables.
    
    The solution vector includes:
    - Indices 0 to num_courses*K-1: Exam assignment variables
    - Indices num_courses*K onward: Slack variables (ignored in decoding)
    
    How slot assignment works:
    1. Each exam has K binary variables (one per time slot)
    2. Variable x[exam_i, slot_c] = 1 means exam i is assigned to slot c
    3. Variable index: var_idx = exam * K + slot
    4. For exam 0: variables 0,1,2,...,K-1 represent slots 0,1,2,...,K-1
    5. For exam 1: variables K,K+1,...,2K-1 represent slots 0,1,2,...,K-1
    
    Slack variables (if present) are used only for capacity constraint penalty
    and are not used in the actual timetable.
    
    Example with 10 exams, K=4, num_slack_bits=4:
    - Exam variables: 0-39 (10 exams × 4 slots)
    - Slack variables: 40-55 (4 slots × 4 bits)
    - Decoding uses only indices 0-39
    """
    
    coloring = {}
    num_exam_vars = num_courses * K
    
    for exam in range(num_courses):
        for slot in range(K):
            var_idx = exam * K + slot
            if solution[var_idx] == 1:
                coloring[exam] = slot
                break
    
    return coloring


def validate_solution(coloring, adjacency, num_courses, solution=None, K=None,
                      courses_df=None, capacity=None):
    """Validate solution across C1-C4 checks and return detailed metrics."""
    
    violations = []
    violation_details = {
        'c1': [],
        'c2': [],
        'c3': [],
        'c4': [],
        'other': []
    }
    
    # Check all exams assigned in decoded coloring
    if len(coloring) != num_courses:
        msg = f"Only {len(coloring)}/{num_courses} exams assigned"
        violations.append(msg)
        violation_details['other'].append({
            'type': 'incomplete_assignment',
            'assigned_exams': int(len(coloring)),
            'num_courses': int(num_courses),
            'message': msg
        })

    # Strict one-hot check from raw solution vector when available.
    if solution is not None and K is not None:
        onehot_violations = 0
        for i in range(num_courses):
            start = i * K
            end = start + K
            ones = int(np.sum(solution[start:end]))
            if ones != 1:
                onehot_violations += 1
                violation_details['c1'].append({
                    'exam': int(i),
                    'active_slots': int(ones),
                    'slot_values': [int(v) for v in solution[start:end]]
                })
                if onehot_violations <= 5:
                    violations.append(f"One-hot violation: Exam {i} has {ones} active slots")
        if onehot_violations > 5:
            violations.append(
                f"... and {onehot_violations - 5} more one-hot violations"
            )
    
    # Check conflicts (C2)
    conflict_count = 0
    for i in range(num_courses):
        for j in range(i+1, num_courses):
            if adjacency[i, j] > 0:
                if i in coloring and j in coloring:
                    if coloring[i] == coloring[j]:
                        conflict_count += 1
                        violation_details['c2'].append({
                            'exam_i': int(i),
                            'exam_j': int(j),
                            'slot': int(coloring[i])
                        })
                        violations.append(f"Conflict: Exams {i},{j} both in slot {coloring[i]}")

    # Check C3: same-year exams in consecutive slots
    c3_consecutive_violations = 0
    if courses_df is not None:
        years = courses_df['year'].values
        for i in range(num_courses):
            for j in range(i + 1, num_courses):
                if years[i] == years[j] and i in coloring and j in coloring:
                    if abs(int(coloring[i]) - int(coloring[j])) == 1:
                        c3_consecutive_violations += 1
                        violation_details['c3'].append({
                            'exam_i': int(i),
                            'exam_j': int(j),
                            'year': int(years[i]),
                            'slot_i': int(coloring[i]),
                            'slot_j': int(coloring[j])
                        })
                        if c3_consecutive_violations <= 5:
                            violations.append(
                                f"C3 violation: Exams {i},{j} (year={years[i]}) in consecutive slots "
                                f"{coloring[i]} and {coloring[j]}"
                            )
        if c3_consecutive_violations > 5:
            violations.append(
                f"... and {c3_consecutive_violations - 5} more C3 violations"
            )

    # Check C4: per-slot capacity
    c4_slots_over_capacity = 0
    c4_total_overflow = 0.0
    c4_max_overflow = 0.0
    if courses_df is not None and capacity is not None and K is not None:
        enrollments = courses_df['enrollment'].values.astype(float)
        slot_loads = np.zeros(int(K), dtype=float)

        for exam, slot in coloring.items():
            slot_loads[int(slot)] += enrollments[int(exam)]

        over_mask = slot_loads > float(capacity)
        c4_slots_over_capacity = int(np.sum(over_mask))
        if c4_slots_over_capacity > 0:
            overflow_vals = slot_loads[over_mask] - float(capacity)
            c4_total_overflow = float(np.sum(overflow_vals))
            c4_max_overflow = float(np.max(overflow_vals))
            over_slots = np.where(over_mask)[0]
            for slot_idx in over_slots:
                overflow = float(slot_loads[slot_idx] - float(capacity))
                exams_in_slot = [int(exam) for exam, slot in coloring.items() if int(slot) == int(slot_idx)]
                violation_details['c4'].append({
                    'slot': int(slot_idx),
                    'slot_load': float(slot_loads[slot_idx]),
                    'capacity': float(capacity),
                    'overflow': overflow,
                    'exams_in_slot': exams_in_slot
                })
            violations.append(
                f"C4 violation: {c4_slots_over_capacity} slots exceed capacity {capacity}; "
                f"max overflow={c4_max_overflow:.2f}, total overflow={c4_total_overflow:.2f}"
            )
    
    is_valid = len(violations) == 0

    metrics = {
        'c1_onehot_violations': int(len(violation_details['c1'])),
        'c2_conflict_violations': int(conflict_count),
        'c3_consecutive_violations': int(c3_consecutive_violations),
        'c4_slots_over_capacity': int(c4_slots_over_capacity),
        'c4_total_overflow': float(c4_total_overflow),
        'c4_max_overflow': float(c4_max_overflow),
        'other_violations': int(len(violation_details['other']))
    }

    return is_valid, conflict_count, violations, metrics, violation_details


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
        description='Interactive Exam Scheduler - Complete Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for inputs)
  python run_exam_scheduler.py
  
  # Command-line mode with default settings (50 students, 40% conflicts)
  python run_exam_scheduler.py --courses 10 --k 4
  
  # Specify number of students
  python run_exam_scheduler.py --courses 10 --students 80 --k 4
  
  # Low conflict density (20%)
  python run_exam_scheduler.py --courses 10 --students 60 --k 3 --conflict-pct 20
  
  # Medium conflict density (50%)
  python run_exam_scheduler.py --courses 10 --students 60 --k 4 --conflict-pct 50
  
  # High conflict density (70%)
  python run_exam_scheduler.py --courses 10 --students 60 --k 6 --conflict-pct 70
  
  # With visualization
  python run_exam_scheduler.py --courses 10 --k 4 --visualize
  
    # Real CSV with dual adjacency graphs (major/all), solve with Neal
    python run_exam_scheduler.py --input-csv "../Student Course (Jul-Nov 2025 and Winter 2025).csv" --adjacency-mode all --k 16
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

    parser.add_argument('--input-csv', type=str, default=None,
                       help='Path to university enrollment CSV (uses Approved+Theory filters and builds major/all adjacency)')

    parser.add_argument('--max-rows', type=int, default=None,
                       help='CSV mode only: use only first N rows from the input dataset before filters')

    parser.add_argument('--adjacency-mode', type=str, default='all',
                       choices=['major', 'all', 'both'],
                       help='When --input-csv is used, choose graph for solver: major, all, or both (default: all)')
    
    parser.add_argument('--backend', type=str, default='neal',
                       choices=['neal'],
                       help='Solver backend (Neal only)')
    
    parser.add_argument('--num-reads', type=int, default=1000,
                       help='Neal number of reads (default: 1000)')
    
    parser.add_argument('--lambda1', type=float, default=10000,
                       help='Penalty: one-hot constraint (default: 10000)')
    
    parser.add_argument('--lambda2', type=float, default=5000,
                       help='Penalty: conflict in same slot (default: 5000)')
    
    parser.add_argument('--lambda3', type=float, default=500,
                       help='Penalty: same-year exams in consecutive slots (default: 500, set 0 to disable)')
    
    parser.add_argument('--lambda4', type=float, default=200,
                       help='Penalty: slot capacity exceeded (default: 200, set 0 to disable)')
    
    parser.add_argument('--capacity', type=int, default=None,
                       help='Max total enrollment per slot (default: auto = mean_enrollment × N/K × 1.2)')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualization plots (requires matplotlib, networkx)')
    
    return parser.parse_args()


def get_user_input(args):
    """Get inputs from command-line or interactive prompts"""
    
    if args.input_csv is None and args.courses is None:
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
    print("EXAM SCHEDULING PIPELINE")
    print("="*60)
    if args.input_csv:
        print(f"Input CSV: {args.input_csv}")
        print(f"Adjacency mode: {args.adjacency_mode}")
        if args.max_rows is not None:
            print(f"Row limit: first {args.max_rows} rows")
    else:
        print(f"Courses: {args.courses}")
        print(f"Students: {args.students}")
    print(f"K (time slots): {args.k}")
    if not args.input_csv:
        print(f"Conflict percentage: {args.conflict_pct:.1f}%")
    print("Backend: NEAL")
    print(f"λ1={args.lambda1}  λ2={args.lambda2}  λ3={args.lambda3}  λ4={args.lambda4}")
    if args.capacity:
        print(f"Slot capacity: {args.capacity}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output_dir = Path('./output') / f'run_{timestamp}'
    root_output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.input_csv:
        modes = ['major', 'all'] if args.adjacency_mode == 'both' else [args.adjacency_mode]
        for mode in modes:
            mode_output_dir = root_output_dir / mode
            mode_output_dir.mkdir(parents=True, exist_ok=True)
            courses_df, adjacency, metadata = generate_dataset_from_csv(
                input_csv=args.input_csv,
                output_dir=mode_output_dir,
                adjacency_mode=mode,
                max_rows=args.max_rows
            )
            jobs.append((mode, mode_output_dir, courses_df, adjacency, metadata))
    else:
        courses_df, adjacency, metadata = generate_dataset(
            num_courses=args.courses,
            num_students=args.students,
            avg_courses_per_student=args.avg_courses,
            conflict_pct=args.conflict_pct,
            output_dir=root_output_dir
        )
        jobs.append(('random', root_output_dir, courses_df, adjacency, metadata))

    for mode, output_dir, courses_df, adjacency, metadata in jobs:
        num_courses = len(courses_df)

        print("\n" + "="*60)
        print(f"RUNNING SOLVER FOR MODE: {mode.upper()}")
        print("="*60)

        # Visualize conflict graph (if requested)
        if args.visualize:
            print("\n" + "="*60)
            print("GENERATING VISUALIZATIONS")
            print("="*60)
            visualize_adjacency_matrix(adjacency, output_dir, num_courses)
            visualize_conflict_graph(adjacency, output_dir, num_courses)

        # Step 2: Build QUBO
        Q = build_qubo(
            adjacency,
            K=args.k,
            courses_df=courses_df,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3,
            lambda4=args.lambda4,
            capacity=args.capacity
        )
        np.save(output_dir / 'qubo_matrix.npy', Q)

        # Step 3: Solve (Neal only)
        result = solve_neal(Q, num_reads=args.num_reads)
        if not result:
            print("\n✗ Neal backend unavailable!")
            continue
        results = {'neal': result}

        # Step 4: Validate and display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)

        for backend_name, result in results.items():
            print(f"\n{backend_name.upper()}")
            print("-" * 60)

            coloring = decode_solution(result['solution'], num_courses, args.k)
            is_valid, num_conflicts, violations, soft_metrics, violation_details = validate_solution(
                coloring, adjacency, num_courses,
                solution=result['solution'], K=args.k,
                courses_df=courses_df, capacity=args.capacity
            )

            print(f"Runtime: {result['runtime']:.2f}s")
            print(f"Energy: {result['energy']:.2f}")
            print(f"Valid: {'✓ YES' if is_valid else '✗ NO'}")
            total_conflict_edges = int(np.sum(adjacency) // 2)
            violated_conflict_pct = (
                (num_conflicts / total_conflict_edges) * 100.0
                if total_conflict_edges > 0 else 0.0
            )
            if args.input_csv:
                graph_conflict_density_pct = float(
                    metadata.get('selected_graph', {}).get('density', 0.0)
                )
            else:
                graph_conflict_density_pct = float(metadata.get('density', 0.0))
            print(f"Graph conflicts (dataset edges): {total_conflict_edges}")
            print(f"Graph conflict density: {graph_conflict_density_pct:.2f}%")
            print(f"C1 one-hot violations: {soft_metrics['c1_onehot_violations']}")
            print(f"C2 same-slot conflict violations: {soft_metrics['c2_conflict_violations']}")
            print(f"Solution conflict violations: {num_conflicts}")
            print(f"Solution conflict violations (% of graph edges): {violated_conflict_pct:.2f}%")
            print(f"C3 consecutive-slot violations: {soft_metrics['c3_consecutive_violations']}")
            print(f"C4 slots over capacity: {soft_metrics['c4_slots_over_capacity']}")
            print(f"C4 max overflow: {soft_metrics['c4_max_overflow']:.2f}")
            print(f"C4 total overflow: {soft_metrics['c4_total_overflow']:.2f}")
            print(f"Exams assigned: {len(coloring)}/{num_courses}")
            print(f"Colors used: {len(set(coloring.values()))}/{args.k}")

            if args.input_csv:
                selected_meta = metadata.get('selected_graph', {})
                reported_num_students = int(selected_meta.get('num_students', 0))
                reported_num_enrollments = int(selected_meta.get('num_enrollments', 0))
                reported_avg_courses = (
                    reported_num_enrollments / reported_num_students
                    if reported_num_students > 0 else None
                )
            else:
                reported_num_students = int(args.students)
                reported_num_enrollments = int(metadata.get('num_enrollments', 0))
                reported_avg_courses = float(args.avg_courses)

            # Save results
            result_data = {
                'backend': backend_name,
                'adjacency_mode': mode,
                'num_courses': num_courses,
                'num_students': reported_num_students,
                'num_enrollments': reported_num_enrollments,
                'k': args.k,
                'avg_courses_per_student': reported_avg_courses,
                'runtime_seconds': result['runtime'],
                'energy': float(result['energy']),
                'is_valid': is_valid,
                'graph_conflict_edges': total_conflict_edges,
                'solution_conflict_violations': num_conflicts,
                'num_conflicts': num_conflicts,
                'graph_conflict_density_pct': graph_conflict_density_pct,
                'conflict_violations_pct': violated_conflict_pct,
                'c1_onehot_violations': soft_metrics['c1_onehot_violations'],
                'c2_conflict_violations': soft_metrics['c2_conflict_violations'],
                'c3_consecutive_violations': soft_metrics['c3_consecutive_violations'],
                'c4_slots_over_capacity': soft_metrics['c4_slots_over_capacity'],
                'c4_max_overflow': soft_metrics['c4_max_overflow'],
                'c4_total_overflow': soft_metrics['c4_total_overflow'],
                'other_violations': soft_metrics['other_violations'],
                'colors_used': len(set(coloring.values())),
                'coloring': {str(k): int(v) for k, v in coloring.items()}
            }

            with open(output_dir / f'{backend_name}_results.json', 'w') as f:
                json.dump(result_data, f, indent=2)

            all_conflicts_data = {
                'backend': backend_name,
                'adjacency_mode': mode,
                'is_valid': bool(is_valid),
                'summary': {
                    'c1_onehot_violations': int(soft_metrics['c1_onehot_violations']),
                    'c2_conflict_violations': int(soft_metrics['c2_conflict_violations']),
                    'c3_consecutive_violations': int(soft_metrics['c3_consecutive_violations']),
                    'c4_slots_over_capacity': int(soft_metrics['c4_slots_over_capacity']),
                    'other_violations': int(soft_metrics['other_violations'])
                },
                'violations': violation_details
            }
            all_conflicts_path = output_dir / f'{backend_name}_all_conflicts.json'
            with open(all_conflicts_path, 'w') as f:
                json.dump(all_conflicts_data, f, indent=2)
            print(f"All conflicts file: {all_conflicts_path}")

            # Generate timetable if valid
            if is_valid:
                timetable = generate_timetable(coloring, courses_df, args.k)
                timetable.to_csv(output_dir / f'timetable_{backend_name}.csv', index=False)
                print(f"\n✓ Saved timetable to: {output_dir}/timetable_{backend_name}.csv")

                # Visualize timetable (if requested)
                if args.visualize:
                    visualize_timetable(coloring, adjacency, courses_df, args.k, output_dir)
            else:
                print(f"\n⚠ Solution invalid. First 3 conflicts:")
                for v in violations[:3]:
                    print(f"  - {v}")

        # Summary per mode
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print(f"Output directory: {output_dir}")
        print(f"Files generated:")
        print(f"  - courses.csv (dataset)")
        print(f"  - conflict_adjacency.csv (conflict graph)")
        print(f"  - qubo_matrix.npy (QUBO)")
        print(f"  - *_results.json (solver outputs)")
        print(f"  - *_all_conflicts.json (all C1/C2/C3/C4 violations)")
        if any(validate_solution(decode_solution(res['solution'], num_courses, args.k),
                     adjacency, num_courses,
                     solution=res['solution'], K=args.k,
                 courses_df=courses_df, capacity=args.capacity)[0]
               for res in results.values()):
            print(f"  - timetable_*.csv (valid schedules)")

    if args.input_csv and args.adjacency_mode == 'both':
        print("\n" + "="*60)
        print("BOTH-MODE RUN COMPLETE")
        print("="*60)
        print(f"Root output directory: {root_output_dir}")
        print(f"Subfolders:")
        print(f"  - {root_output_dir / 'major'}")
        print(f"  - {root_output_dir / 'all'}")
    
if __name__ == '__main__':
    main()