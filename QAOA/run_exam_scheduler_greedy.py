#!/usr/bin/env python3
"""
Greedy Graph Coloring Solver for Exam Scheduling
Companion to run_exam_scheduler.py for comparing greedy vs quantum/annealing approaches

Uses Welsh-Powell greedy algorithm for graph coloring.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def generate_random_adjacency(num_courses, conflict_pct):
    """
    Generate random uniform conflict adjacency matrix
    
    Args:
        num_courses: Number of courses
        conflict_pct: Percentage of conflicts (0-100)
    
    Returns:
        adjacency: N×N binary matrix where 1 = conflict
    """
    adjacency = np.zeros((num_courses, num_courses), dtype=int)
    
    # Total possible edges (undirected graph, no self-loops)
    max_edges = num_courses * (num_courses - 1) // 2
    
    # How many edges to create
    num_edges = int(max_edges * conflict_pct / 100.0)
    
    # Generate all possible edge indices
    edge_candidates = []
    for i in range(num_courses):
        for j in range(i + 1, num_courses):
            edge_candidates.append((i, j))
    
    # Randomly select edges
    selected_edges = np.random.choice(len(edge_candidates), size=num_edges, replace=False)
    
    for idx in selected_edges:
        i, j = edge_candidates[idx]
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
    
    Returns:
        courses_df: DataFrame with course data
        adjacency: Conflict adjacency matrix
        metadata: Dictionary with dataset statistics
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
    
    # Statistics
    num_edges = np.sum(adjacency) // 2
    max_possible = num_courses * (num_courses - 1) // 2
    density = (num_edges / max_possible * 100) if max_possible > 0 else 0
    
    degrees = np.sum(adjacency, axis=1)
    max_degree = int(np.max(degrees))
    
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
    pd.DataFrame(adjacency).to_csv(output_path / 'conflict_adjacency.csv', index=False, header=False)
    
    metadata = {
        'num_courses': int(num_courses),
        'num_students': int(num_students),
        'num_enrollments': int(len(enrollments_df)),
        'conflict_pct': float(conflict_pct),
        'num_conflicts': int(num_edges),
        'density': float(density),
        'max_degree': int(max_degree),
        'min_k_estimate': int(max_degree + 1)
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return courses_df, adjacency, metadata


def greedy_coloring_welsh_powell(adjacency, k):
    """
    Welsh-Powell greedy graph coloring algorithm
    
    Algorithm:
    1. Sort vertices by degree (descending)
    2. For each vertex, assign the smallest available color
    
    Args:
        adjacency: N×N binary conflict matrix
        k: Number of colors (time slots) available
    
    Returns:
        coloring: Array where coloring[i] = color assigned to vertex i
    """
    n = len(adjacency)
    
    # Calculate degrees
    degrees = np.sum(adjacency, axis=1)
    
    # Sort vertices by degree (descending)
    vertices_sorted = np.argsort(-degrees)
    
    # Initialize coloring (-1 means not colored yet)
    coloring = np.full(n, -1, dtype=int)
    
    # Color each vertex
    for vertex in vertices_sorted:
        # Find colors used by neighbors
        used_colors = set()
        for neighbor in range(n):
            if adjacency[vertex, neighbor] == 1 and coloring[neighbor] != -1:
                used_colors.add(coloring[neighbor])
        
        # Find first available color
        color_assigned = None
        for color in range(k):
            if color not in used_colors:
                color_assigned = color
                break
        
        if color_assigned is not None:
            coloring[vertex] = color_assigned
        else:
            # No color available within k colors - assign anyway for analysis
            # This will result in conflicts
            coloring[vertex] = 0  # Default to slot 0
    
    return coloring


def validate_solution(adjacency, coloring):
    """
    Validate graph coloring solution
    
    Args:
        adjacency: N×N conflict matrix
        coloring: Assignment of colors to vertices
    
    Returns:
        is_valid: True if no conflicts
        num_conflicts: Number of conflicting edges
    """
    n = len(adjacency)
    conflicts = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency[i, j] == 1:  # Conflict edge exists
                if coloring[i] == coloring[j]:  # Same color assigned
                    conflicts += 1
    
    return conflicts == 0, conflicts


def solve_greedy(adjacency, k):
    """
    Solve exam scheduling using greedy graph coloring
    
    Args:
        adjacency: N×N conflict matrix
        k: Number of time slots
    
    Returns:
        result: Dictionary with solution and metrics
    """
    print("\n" + "="*60)
    print("GREEDY SOLVER (Welsh-Powell)")
    print("="*60)
    
    start_time = time.time()
    
    # Run greedy coloring
    coloring = greedy_coloring_welsh_powell(adjacency, k)
    
    runtime = time.time() - start_time
    
    # Validate
    is_valid, num_conflicts = validate_solution(adjacency, coloring)
    
    # Count exams per slot
    slot_counts = {}
    for slot in range(k):
        slot_counts[slot] = int(np.sum(coloring == slot))
    
    result = {
        'algorithm': 'greedy_welsh_powell',
        'k': k,
        'runtime': runtime,
        'is_valid': is_valid,
        'num_conflicts': num_conflicts,
        'coloring': coloring.tolist(),
        'slot_counts': slot_counts
    }
    
    # Print results
    print(f"\nRuntime: {runtime:.4f}s")
    print(f"Valid: {'✓ YES' if is_valid else '✗ NO'}")
    print(f"Conflicts: {num_conflicts}")
    print(f"\nSlot distribution:")
    for slot in range(k):
        count = slot_counts[slot]
        bar = "█" * count
        print(f"  Slot {slot}: {count:2d} exams  {bar}")
    
    return result


def generate_timetable(courses_df, coloring, output_path):
    """
    Generate human-readable timetable from coloring
    
    Args:
        courses_df: DataFrame with course information
        coloring: Array of color assignments
        output_path: Path to save timetable CSV
    """
    timetable = []
    
    for course_id, slot in enumerate(coloring):
        course_info = courses_df.iloc[course_id]
        timetable.append({
            'time_slot': slot,
            'exam_id': course_id,
            'course_code': course_info['course_code'],
            'year': course_info['year'],
            'enrollment': course_info['enrollment']
        })
    
    timetable_df = pd.DataFrame(timetable)
    timetable_df = timetable_df.sort_values(['time_slot', 'exam_id'])
    timetable_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Timetable saved: {output_path}")
    
    return timetable_df


def parse_args():
    """Parse command-line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Greedy Exam Scheduler - Welsh-Powell Algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python run_exam_scheduler_greedy.py
  
  # Command-line mode with defaults
  python run_exam_scheduler_greedy.py --courses 10 --k 4
  
  # Specify all parameters
  python run_exam_scheduler_greedy.py --courses 10 --students 80 --k 4 --conflict-pct 30
  
  # Compare with quantum solvers
  python run_exam_scheduler.py --courses 10 --k 4 --backend both
  python run_exam_scheduler_greedy.py --courses 10 --k 4
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
    
    return parser.parse_args()


def get_user_input(args):
    """Get inputs from command-line or interactive prompts"""
    
    if args.courses is None:
        print("\n" + "="*60)
        print("INTERACTIVE MODE - GREEDY SOLVER")
        print("="*60)
        args.courses = int(input("Number of courses/exams (e.g., 8, 10, 15): "))
        args.students = int(input("Number of students (e.g., 40, 60, 80): "))
    
    if args.k is None:
        args.k = int(input("Number of time slots/colors K (e.g., 3, 4, 5): "))
    
    return args


def main():
    """Main execution pipeline"""
    
    args = parse_args()
    args = get_user_input(args)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('output') / f'run_{timestamp}_greedy'
    
    print("\n" + "="*60)
    print("GREEDY EXAM SCHEDULING PIPELINE")
    print("="*60)
    print(f"Courses: {args.courses}")
    print(f"Students: {args.students}")
    print(f"K (time slots): {args.k}")
    print(f"Conflict percentage: {args.conflict_pct:.1f}%")
    print(f"Algorithm: Welsh-Powell Greedy")
    
    # Step 1: Generate dataset
    courses_df, adjacency, metadata = generate_dataset(
        num_courses=args.courses,
        num_students=args.students,
        avg_courses_per_student=args.avg_courses,
        conflict_pct=args.conflict_pct,
        output_dir=output_dir
    )
    
    # Step 2: Solve with greedy
    result = solve_greedy(adjacency, args.k)
    
    # Step 3: Save results
    result_path = output_dir / 'greedy_results.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✓ Results saved: {result_path}")
    
    # Step 4: Generate timetable if valid
    if result['is_valid']:
        timetable_path = output_dir / 'timetable_greedy.csv'
        generate_timetable(courses_df, np.array(result['coloring']), timetable_path)
    else:
        print(f"\n⚠ Solution invalid ({result['num_conflicts']} conflicts)")
        print(f"  Try increasing K (current: {args.k})")
        print(f"  Recommended K ≥ {metadata['min_k_estimate']}")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Output directory: {output_dir}/")
    print(f"\nGreedy (Welsh-Powell):")
    print(f"  Runtime:   {result['runtime']:.4f}s")
    print(f"  Valid:     {'✓ YES' if result['is_valid'] else '✗ NO'}")
    print(f"  Conflicts: {result['num_conflicts']}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    main()
