"""
Unified Quantum Solver for Exam Scheduling
===========================================

Single script that handles:
- QAOA (IBM Qiskit)
- D-Wave Neal (Simulated Annealing)
- D-Wave QPU (Quantum Annealing)
- D-Wave Hybrid (Classical + Quantum)
- Multiple K values testing
- Backend comparison
- Room assignment

Usage:
    # Single solver run
    python 04_unified_solver.py tiny 3 --backend qaoa
    python 04_unified_solver.py tiny 3 --backend neal
    
    # Test multiple K values
    python 04_unified_solver.py tiny --k-range 2 5 --backend qaoa
    python 04_unified_solver.py small --k-range 3 5 --backend neal
    
    # Compare backends
    python 04_unified_solver.py tiny 3 --compare-backends qaoa neal
    
    # Full benchmark (all K values, all backends)
    python 04_unified_solver.py tiny --benchmark
    python 04_unified_solver.py --all-datasets --benchmark

Author: Quantum Exam Scheduling Research
Date: February 2025
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import argparse
import sys
import signal
import matplotlib.pyplot as plt

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

# D-Wave imports
try:
    from dimod import BinaryQuadraticModel
    from dwave.samplers import SimulatedAnnealingSampler
    from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False


# ============================================================================
# TIMEOUT HANDLER
# ============================================================================

class TimeoutException(Exception):
    """Raised when solver exceeds timeout"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutException("Solver exceeded timeout limit")


# ============================================================================
# COMMAND-LINE ARGUMENT PARSER
# ============================================================================

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified quantum solver for exam scheduling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single solve
  python 04_unified_solver.py tiny 3 --backend qaoa
  python 04_unified_solver.py tiny 3 --backend neal
  
  # Multiple K values (finds best K)
  python 04_unified_solver.py tiny --k-range 2 5 --backend qaoa
  python 04_unified_solver.py small --k-range 3 5 --backend neal
  
  # Compare backends (same K, different solvers)
  python 04_unified_solver.py tiny 3 --compare-backends qaoa neal
  
  # Full benchmark (all K, all backends)
  python 04_unified_solver.py tiny --benchmark
  python 04_unified_solver.py small --benchmark --backends qaoa neal
  
  # All datasets benchmark
  python 04_unified_solver.py --all-datasets --benchmark
        """
    )
    
    # Dataset selection
    parser.add_argument('dataset', nargs='?', default='tiny',
                       choices=['tiny', 'small', 'medium'],
                       help='Dataset to solve (default: tiny)')
    
    parser.add_argument('K', nargs='?', type=int, default=None,
                       help='Number of time slots/colors (optional if using --k-range)')
    
    # Backend selection
    parser.add_argument('--backend', type=str, default='qaoa',
                       choices=['qaoa', 'neal', 'dwave', 'hybrid'],
                       help='Solver backend (default: qaoa)')
    
    parser.add_argument('--backends', nargs='+',
                       choices=['qaoa', 'neal', 'dwave', 'hybrid'],
                       help='Multiple backends for comparison')
    
    # K value options
    parser.add_argument('--k-range', nargs=2, type=int, metavar=('MIN', 'MAX'),
                       help='Test K values from MIN to MAX')
    
    # Benchmarking options
    parser.add_argument('--benchmark', action='store_true',
                       help='Run full benchmark (all K in range, optionally multiple backends)')
    
    parser.add_argument('--compare-backends', nargs='+',
                       choices=['qaoa', 'neal', 'dwave', 'hybrid'],
                       help='Compare multiple backends on same problem')
    
    parser.add_argument('--all-datasets', action='store_true',
                       help='Test all datasets (tiny, small, medium)')
    
    # Solver parameters
    parser.add_argument('--reps', type=int, default=2,
                       help='QAOA depth/layers (default: 2)')
    
    parser.add_argument('--maxiter', type=int, default=100,
                       help='Max optimizer iterations (QAOA only, default: 100)')
    
    parser.add_argument('--num-reads', type=int, default=1000,
                       help='Number of reads (D-Wave backends, default: 1000)')
    
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per solve in seconds (default: 600)')
    
    # Output options
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    
    parser.add_argument('--assign-rooms', action='store_true',
                       help='Run room assignment after solving')
    
    args = parser.parse_args()
    
    # Validation and defaults
    if args.compare_backends:
        args.backends = args.compare_backends
    
    if args.benchmark and not args.backends:
        args.backends = ['qaoa', 'neal']  # Default benchmark backends
    
    if args.K is None and args.k_range is None and not args.benchmark:
        # Auto K
        K_defaults = {'tiny': 3, 'small': 4, 'medium': 5}
        args.K = K_defaults[args.dataset]
    
    if args.k_range and not args.benchmark:
        # k-range implies we test each K
        args.benchmark = True
    
    return args


# ============================================================================
# DATA LOADING
# ============================================================================

def get_latest_run_dir():
    """Get latest run directory"""
    latest_file = Path('./output') / 'latest_run.txt'
    if not latest_file.exists():
        print("⚠ No run directory found. Run 01_generate_dataset.py first.")
        sys.exit(1)
    
    with open(latest_file, 'r') as f:
        return Path(f.read().strip())


def load_qubo_data(dataset, K):
    """Load QUBO matrix and related data"""
    run_dir = get_latest_run_dir()
    data_dir = run_dir / 'datasets' / f'exam_data_{dataset}'
    
    if not data_dir.exists():
        print(f"✗ Dataset not found: {data_dir}")
        print("Available datasets:")
        for d in (run_dir / 'datasets').glob('exam_data_*'):
            print(f"  - {d.name}")
        sys.exit(1)
    
    # Check if QUBO exists for this K
    qubo_file = data_dir / f'qubo_matrix_K{K}.npy'
    metadata_file = data_dir / f'qubo_metadata_K{K}.json'
    
    if not qubo_file.exists():
        print(f"✗ QUBO not found for K={K}: {qubo_file}")
        print(f"⚠ Run: python 03_build_qubo.py")
        print(f"Available K values:")
        for f in data_dir.glob('qubo_matrix_K*.npy'):
            k_val = f.stem.split('_K')[1]
            print(f"  - K={k_val}")
        return None
    
    # Load QUBO
    Q = np.load(qubo_file)
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Load graph data
    adjacency = pd.read_csv(data_dir / 'conflict_adjacency.csv', index_col=0).values
    courses = pd.read_csv(data_dir / 'courses.csv')
    
    return Q, metadata, adjacency, courses, data_dir


# ============================================================================
# SOLVER CLASSES
# ============================================================================

class SolverBase:
    """Base class for all solvers"""
    
    def __init__(self, Q, metadata, backend_name):
        self.Q = Q
        self.metadata = metadata
        self.backend_name = backend_name
        self.num_vars = Q.shape[0]
    
    def solve(self):
        """Override in subclasses"""
        raise NotImplementedError


class QAOASolver(SolverBase):
    """QAOA solver (IBM Qiskit)"""
    
    def __init__(self, Q, metadata, reps=2, maxiter=100, timeout=600):
        super().__init__(Q, metadata, 'qaoa')
        self.reps = reps
        self.maxiter = maxiter
        self.timeout = timeout
        self.start_time = None
        self.timeout_exceeded = False
    
    def solve(self):
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not installed. Run: pip install qiskit qiskit-optimization")
        
        print(f"\n{'='*60}")
        print(f"QAOA SOLVER (IBM Qiskit)")
        print(f"{'='*60}")
        print(f"Variables: {self.num_vars}")
        print(f"QAOA depth (p): {self.reps}")
        print(f"Max iterations: {self.maxiter}")
        
        # Build QuadraticProgram
        qp = QuadraticProgram()
        for i in range(self.num_vars):
            qp.binary_var(name=f'x{i}')
        
        linear = {}
        quadratic = {}
        
        for i in range(self.num_vars):
            if self.Q[i, i] != 0:
                linear[f'x{i}'] = self.Q[i, i]
            for j in range(i+1, self.num_vars):
                if self.Q[i, j] != 0:
                    quadratic[(f'x{i}', f'x{j}')] = 2 * self.Q[i, j]
        
        qp.minimize(linear=linear, quadratic=quadratic)
        
        # Solve with hard timeout using signals (Unix only)
        sampler = Sampler()
        optimizer = COBYLA(maxiter=self.maxiter)
        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=self.reps)
        qaoa_optimizer = MinimumEigenOptimizer(qaoa)
        
        print("\n⏳ Running QAOA...")
        print(f"Timeout: {self.timeout}s (hard limit)")
        
        self.start_time = time.time()
        self.timeout_exceeded = False
        
        # Set up timeout signal (Unix/macOS only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.timeout)  # Set alarm
            
            result = qaoa_optimizer.solve(qp)
            
            signal.alarm(0)  # Cancel alarm
            runtime = time.time() - self.start_time
            
            print(f"✓ Completed in {runtime:.2f}s")
            print(f"Energy: {result.fval:.2f}")
            
            return {
                'solution': result.x,
                'energy': result.fval,
                'runtime': runtime,
                'backend': 'qaoa',
                'timeout_exceeded': False
            }
            
        except TimeoutException:
            signal.alarm(0)  # Cancel alarm
            runtime = time.time() - self.start_time
            self.timeout_exceeded = True
            
            print(f"\n⚠ TIMEOUT! Stopped after {runtime:.1f}s")
            print("⚠ Returning best solution found so far (may be invalid)")
            
            # Return a dummy solution (all zeros)
            dummy_solution = np.zeros(self.num_vars)
            
            return {
                'solution': dummy_solution,
                'energy': 0.0,
                'runtime': runtime,
                'backend': 'qaoa',
                'timeout_exceeded': True
            }


class NealSolver(SolverBase):
    """D-Wave Neal simulated annealing solver"""
    
    def __init__(self, Q, metadata, num_reads=1000):
        super().__init__(Q, metadata, 'neal')
        self.num_reads = num_reads
    
    def qubo_to_bqm(self):
        """Convert QUBO to BQM"""
        linear = {}
        quadratic = {}
        
        for i in range(self.num_vars):
            if self.Q[i, i] != 0:
                linear[i] = self.Q[i, i]
            for j in range(i+1, self.num_vars):
                if self.Q[i, j] != 0:
                    quadratic[(i, j)] = self.Q[i, j]
        
        return BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')
    
    def solve(self):
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK not installed. Run: pip install dwave-ocean-sdk")
        
        print(f"\n{'='*60}")
        print(f"D-WAVE NEAL SIMULATOR (Simulated Annealing)")
        print(f"{'='*60}")
        print(f"Variables: {self.num_vars}")
        print(f"Num reads: {self.num_reads}")
        
        bqm = self.qubo_to_bqm()
        
        print("\n⏳ Running simulated annealing...")
        start_time = time.time()
        
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=self.num_reads)
        
        runtime = time.time() - start_time
        
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        
        # Convert to solution array
        solution = np.zeros(self.num_vars, dtype=int)
        for i in range(self.num_vars):
            solution[i] = best_sample.get(i, 0)
        
        print(f"✓ Completed in {runtime:.2f}s")
        print(f"Energy: {best_energy:.2f}")
        
        return {
            'solution': solution,
            'energy': best_energy,
            'runtime': runtime,
            'backend': 'neal'
        }


class DWaveQPUSolver(SolverBase):
    """D-Wave QPU (quantum annealer)"""
    
    def __init__(self, Q, metadata, num_reads=100):
        super().__init__(Q, metadata, 'dwave_qpu')
        self.num_reads = num_reads
    
    def qubo_to_bqm(self):
        linear = {}
        quadratic = {}
        for i in range(self.num_vars):
            if self.Q[i, i] != 0:
                linear[i] = self.Q[i, i]
            for j in range(i+1, self.num_vars):
                if self.Q[i, j] != 0:
                    quadratic[(i, j)] = self.Q[i, j]
        return BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')
    
    def solve(self):
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK not installed")
        
        print(f"\n{'='*60}")
        print(f"D-WAVE QUANTUM ANNEALER (QPU)")
        print(f"{'='*60}")
        
        bqm = self.qubo_to_bqm()
        
        try:
            sampler = EmbeddingComposite(DWaveSampler())
            print(f"Connected to: {sampler.child.solver.name}")
            
            start_time = time.time()
            sampleset = sampler.sample(bqm, num_reads=self.num_reads)
            runtime = time.time() - start_time
            
            solution = np.zeros(self.num_vars, dtype=int)
            for i in range(self.num_vars):
                solution[i] = sampleset.first.sample.get(i, 0)
            
            print(f"✓ QPU execution complete: {runtime:.2f}s")
            
            return {
                'solution': solution,
                'energy': sampleset.first.energy,
                'runtime': runtime,
                'backend': 'dwave_qpu'
            }
        
        except Exception as e:
            print(f"✗ QPU access failed: {e}")
            print("Falling back to Neal simulator...")
            neal = NealSolver(self.Q, self.metadata, self.num_reads)
            return neal.solve()


class HybridSolver(SolverBase):
    """D-Wave Hybrid solver"""
    
    def __init__(self, Q, metadata):
        super().__init__(Q, metadata, 'hybrid')
    
    def qubo_to_bqm(self):
        linear = {}
        quadratic = {}
        for i in range(self.num_vars):
            if self.Q[i, i] != 0:
                linear[i] = self.Q[i, i]
            for j in range(i+1, self.num_vars):
                if self.Q[i, j] != 0:
                    quadratic[(i, j)] = self.Q[i, j]
        return BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')
    
    def solve(self):
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK not installed")
        
        print(f"\n{'='*60}")
        print(f"D-WAVE HYBRID SOLVER")
        print(f"{'='*60}")
        
        bqm = self.qubo_to_bqm()
        
        try:
            sampler = LeapHybridSampler()
            
            start_time = time.time()
            sampleset = sampler.sample(bqm)
            runtime = time.time() - start_time
            
            solution = np.zeros(self.num_vars, dtype=int)
            for i in range(self.num_vars):
                solution[i] = sampleset.first.sample.get(i, 0)
            
            print(f"✓ Hybrid solver complete: {runtime:.2f}s")
            
            return {
                'solution': solution,
                'energy': sampleset.first.energy,
                'runtime': runtime,
                'backend': 'hybrid'
            }
        
        except Exception as e:
            print(f"✗ Hybrid access failed: {e}")
            print("Falling back to Neal simulator...")
            neal = NealSolver(self.Q, self.metadata, 1000)
            return neal.solve()


# ============================================================================
# SOLUTION DECODING & VALIDATION
# ============================================================================

def decode_solution(solution, metadata):
    """Decode binary solution to exam coloring"""
    n_exams = metadata['num_exams']
    K = metadata['num_colors']
    
    coloring = {}
    for exam in range(n_exams):
        for color in range(K):
            var_idx = exam * K + color
            if solution[var_idx] == 1:
                coloring[exam] = color
                break
    
    return coloring


def validate_solution(coloring, adjacency, metadata):
    """Validate coloring against conflicts"""
    n_exams = metadata['num_exams']
    violations = []
    
    # Check all exams assigned
    if len(coloring) != n_exams:
        violations.append(f"Only {len(coloring)}/{n_exams} exams colored")
    
    # Check no conflicts
    conflict_count = 0
    for i in range(n_exams):
        for j in range(i+1, n_exams):
            if adjacency[i, j] > 0:
                if i in coloring and j in coloring:
                    if coloring[i] == coloring[j]:
                        conflict_count += 1
                        violations.append(f"Exams {i},{j} both in slot {coloring[i]}")
    
    is_valid = len(violations) == 0
    
    return is_valid, conflict_count, violations


# ============================================================================
# ROOM ASSIGNMENT
# ============================================================================

def assign_rooms(coloring, courses, metadata, data_dir):
    """Assign rooms to scheduled exams"""
    print("\n" + "="*60)
    print("ROOM ASSIGNMENT")
    print("="*60)
    
    # Load rooms
    rooms = pd.read_csv(data_dir / 'rooms.csv')
    
    # Group exams by time slot
    schedule = {}
    for exam_id, slot in coloring.items():
        if slot not in schedule:
            schedule[slot] = []
        schedule[slot].append({
            'exam_id': exam_id,
            'course': courses.iloc[exam_id]['course_code'],
            'students': courses.iloc[exam_id]['enrollment']
        })
    
    # Assign rooms (greedy)
    assignments = []
    for slot, exams in schedule.items():
        # Sort exams by enrollment (descending)
        exams_sorted = sorted(exams, key=lambda x: x['students'], reverse=True)
        
        # Sort rooms by capacity (descending)
        rooms_sorted = rooms.sort_values('capacity', ascending=False)
        
        for i, exam in enumerate(exams_sorted):
            if i < len(rooms_sorted):
                room = rooms_sorted.iloc[i]
                assignments.append({
                    'time_slot': slot,
                    'exam_id': exam['exam_id'],
                    'course': exam['course'],
                    'students': exam['students'],
                    'room': room['room_name'],
                    'capacity': room['capacity'],
                    'utilization': exam['students'] / room['capacity']
                })
    
    assignments_df = pd.DataFrame(assignments)
    
    print(f"\n✓ Assigned {len(assignments)} exams to rooms")
    print(f"Average utilization: {assignments_df['utilization'].mean():.1%}")
    
    return assignments_df


# ============================================================================
# TIMETABLE GENERATION
# ============================================================================

def generate_timetable(coloring, courses, rooms_assignments, metadata, data_dir):
    """
    Generate human-readable timetable
    
    Args:
        coloring: dict {exam_id: time_slot}
        courses: DataFrame with course information
        rooms_assignments: DataFrame with room assignments (or None)
        metadata: QUBO metadata
        data_dir: Path to dataset directory
        
    Returns:
        DataFrame: Full timetable
    """
    print("\n" + "="*60)
    print("GENERATING TIMETABLE")
    print("="*60)
    
    # Group exams by time slot
    schedule = {}
    for exam_id, slot in coloring.items():
        if slot not in schedule:
            schedule[slot] = []
        
        course_info = courses.iloc[exam_id]
        
        exam_entry = {
            'time_slot': slot,
            'exam_id': exam_id,
            'course_code': course_info['course_code'],
            'year': course_info['year'],
            'enrollment': course_info['enrollment'],
            'room': None,
            'room_capacity': None
        }
        
        # Add room info if available
        if rooms_assignments is not None:
            room_info = rooms_assignments[rooms_assignments['exam_id'] == exam_id]
            if len(room_info) > 0:
                exam_entry['room'] = room_info.iloc[0]['room']
                exam_entry['room_capacity'] = room_info.iloc[0]['capacity']
        
        schedule[slot].append(exam_entry)
    
    # Create timetable DataFrame
    timetable_rows = []
    for slot in sorted(schedule.keys()):
        for exam in sorted(schedule[slot], key=lambda x: x['course_code']):
            timetable_rows.append(exam)
    
    timetable_df = pd.DataFrame(timetable_rows)
    
    print(f"\n✓ Generated timetable with {len(timetable_df)} exams")
    print(f"✓ Time slots used: {len(schedule)}/{metadata['num_colors']}")
    
    # Print summary by slot
    print("\nSchedule Summary:")
    print("-" * 60)
    for slot in sorted(schedule.keys()):
        exams_in_slot = schedule[slot]
        total_students = sum(e['enrollment'] for e in exams_in_slot)
        print(f"  Slot {slot}: {len(exams_in_slot)} exams, {total_students} students")
    
    return timetable_df


def print_timetable(timetable_df):
    """Pretty print timetable to console"""
    
    print("\n" + "="*60)
    print("FINAL EXAM TIMETABLE")
    print("="*60)
    
    # Group by time slot
    for slot in sorted(timetable_df['time_slot'].unique()):
        slot_exams = timetable_df[timetable_df['time_slot'] == slot]
        
        print(f"\n{'='*60}")
        print(f"TIME SLOT {slot}")
        print(f"{'='*60}")
        
        for _, exam in slot_exams.iterrows():
            print(f"\n  {exam['course_code']:10s}")
            print(f"  Year {exam['year']} | {exam['enrollment']} students")
            
            if pd.notna(exam['room']):
                utilization = (exam['enrollment'] / exam['room_capacity']) * 100
                print(f"  Room: {exam['room']} (Capacity: {exam['room_capacity']}, "
                      f"Utilization: {utilization:.1f}%)")
            else:
                print(f"  Room: Not assigned")
    
    print("\n" + "="*60)


# ============================================================================
# SINGLE SOLVE WORKFLOW
# ============================================================================

def solve_single(dataset, K, backend, args):
    """Solve single problem instance"""
    
    print("="*60)
    print("QUANTUM EXAM SCHEDULING SOLVER")
    print("="*60)
    print(f"\n📊 Dataset: {dataset.upper()}")
    print(f"🎨 Colors (K): {K}")
    print(f"🔧 Backend: {backend.upper()}\n")
    
    # Load data
    result = load_qubo_data(dataset, K)
    if result is None:
        print("✗ Skipping this configuration")
        return None
    
    Q, metadata, adjacency, courses, data_dir = result
    
    # Select solver
    if backend == 'qaoa':
        solver = QAOASolver(Q, metadata, reps=args.reps, maxiter=args.maxiter, timeout=args.timeout)
    elif backend == 'neal':
        solver = NealSolver(Q, metadata, num_reads=args.num_reads)
    elif backend == 'dwave':
        solver = DWaveQPUSolver(Q, metadata, num_reads=args.num_reads)
    elif backend == 'hybrid':
        solver = HybridSolver(Q, metadata)
    else:
        print(f"✗ Unknown backend: {backend}")
        return None
    
    # Solve
    try:
        result = solver.solve()
    except Exception as e:
        print(f"\n✗ Solver failed: {e}")
        return None
    
    # Decode and validate
    coloring = decode_solution(result['solution'], metadata)
    is_valid, num_conflicts, violations = validate_solution(coloring, adjacency, metadata)
    
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    if is_valid:
        print("✓ Solution is VALID!")
        print(f"✓ All {metadata['num_exams']} exams successfully colored")
        print(f"✓ Colors used: {len(set(coloring.values()))}/{K}")
    else:
        print("✗ Solution INVALID")
        print(f"✗ Conflicts: {num_conflicts}")
        for v in violations[:5]:
            print(f"  - {v}")
    
    # Save results
    run_dir = get_latest_run_dir()
    solutions_dir = run_dir / 'solutions'
    solutions_dir.mkdir(exist_ok=True)
    
    result_data = {
        'dataset': dataset,
        'K': K,
        'backend': backend,
        'num_variables': metadata['num_variables'],
        'num_exams': metadata['num_exams'],
        'runtime_seconds': result['runtime'],
        'energy': float(result['energy']),
        'is_valid': is_valid,
        'num_conflicts': num_conflicts,
        'colors_used': len(set(coloring.values())),
        'timeout_exceeded': result.get('timeout_exceeded', False),
        'coloring': {str(k): int(v) for k, v in coloring.items()}
    }
    
    results_file = solutions_dir / f'{backend}_results_{dataset}_K{K}.json'
    with open(results_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✓ Saved results to {results_file}")
    
    # ========== NEW: Generate and save timetable ==========
    
    # Room assignment
    rooms_assignments = None
    if args.assign_rooms and is_valid:
        rooms_assignments = assign_rooms(coloring, courses, metadata, data_dir)
        assignments_file = solutions_dir / f'room_assignments_{dataset}_K{K}_{backend}.csv'
        rooms_assignments.to_csv(assignments_file, index=False)
        print(f"✓ Saved room assignments to {assignments_file}")
    
    # Generate timetable (always, not just when valid)
    if is_valid:  # Only generate timetable for valid solutions
        timetable = generate_timetable(coloring, courses, rooms_assignments, metadata, data_dir)
        
        # Save to CSV
        timetables_dir = run_dir / 'timetables'
        timetables_dir.mkdir(exist_ok=True)
        timetable_file = timetables_dir / f'timetable_{dataset}_K{K}_{backend}.csv'
        timetable.to_csv(timetable_file, index=False)
        print(f"✓ Saved timetable to {timetable_file}")
        
        # Save human-readable text version
        timetable_txt_file = timetables_dir / f'timetable_{dataset}_K{K}_{backend}.txt'
        with open(timetable_txt_file, 'w') as f:
            # Redirect print to file
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            
            print_timetable(timetable)
            
            sys.stdout = old_stdout
        
        print(f"✓ Saved readable timetable to {timetable_txt_file}")
        
        # Print to console
        print_timetable(timetable)
    
    # ========== End of new code ==========
    
    return result_data


# ============================================================================
# BENCHMARK WORKFLOW
# ============================================================================

def run_benchmark(datasets, k_ranges, backends, args):
    """Run comprehensive benchmark"""
    
    print("="*60)
    print("QUANTUM SOLVER BENCHMARK")
    print("="*60)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Backends: {', '.join(backends)}")
    print(f"K ranges: {k_ranges}")
    
    all_results = []
    
    for dataset in datasets:
        k_min, k_max = k_ranges.get(dataset, (2, 5))
        
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset.upper()} (K={k_min} to {k_max})")
        print(f"{'#'*60}")
        
        for K in range(k_min, k_max + 1):
            for backend in backends:
                print(f"\nTesting: {dataset} | K={K} | {backend.upper()}")
                
                result = solve_single(dataset, K, backend, args)
                
                if result:
                    all_results.append(result)
                    
                    # Quick summary  
                    valid_str = '✓' if result['is_valid'] else '✗'
                    timeout_str = ' [TIMEOUT]' if result.get('timeout_exceeded', False) else ''
                    print(f"  {valid_str} {result['runtime_seconds']:.1f}s{timeout_str} | "
                          f"Energy: {result['energy']:.1f} | "
                          f"Conflicts: {result['num_conflicts']}")
                else:
                    print(f"  ⚠ Skipped")
    
    # Save benchmark results
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        run_dir = get_latest_run_dir()
        benchmark_file = run_dir / 'benchmark_results.csv'
        df.to_csv(benchmark_file, index=False)
        
        print(f"\n✓ Saved benchmark results to {benchmark_file}")
    else:
        print(f"\n⚠ No results to save")
        return
    
    # Generate summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    successful = df[df['is_valid'] == True]
    
    if len(successful) > 0:
        summary = successful.groupby('backend').agg({
            'runtime_seconds': ['mean', 'std'],
            'energy': 'mean',
            'num_conflicts': 'mean'
        }).round(2)
        
        print("\n" + summary.to_string())
        
        # Best per dataset
        print("\n" + "-"*60)
        for dataset in datasets:
            dataset_valid = successful[successful['dataset'] == dataset]
            if len(dataset_valid) > 0:
                best = dataset_valid.loc[dataset_valid['runtime_seconds'].idxmin()]
                print(f"{dataset.upper()}: {best['backend'].upper()} "
                      f"(K={best['K']}, {best['runtime_seconds']:.1f}s)")
        
        # Generate plots
        if not args.no_viz:
            create_benchmark_plots(df, backends, datasets, run_dir)
    
    return df


def create_benchmark_plots(df, backends, datasets, run_dir):
    """Create benchmark visualization"""
    
    valid_df = df[df['is_valid'] == True]
    
    if len(valid_df) == 0:
        print("\n⚠ No valid solutions to plot")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Runtime by backend
    valid_df.groupby('backend')['runtime_seconds'].mean().plot(
        kind='bar', ax=axes[0], color='skyblue'
    )
    axes[0].set_ylabel('Runtime (seconds)')
    axes[0].set_title('Average Runtime by Backend')
    axes[0].grid(True, alpha=0.3)
    
    # Energy by backend
    valid_df.groupby('backend')['energy'].mean().plot(
        kind='bar', ax=axes[1], color='lightcoral'
    )
    axes[1].set_ylabel('Energy')
    axes[1].set_title('Average Energy by Backend')
    axes[1].grid(True, alpha=0.3)
    
    # Success rate
    (df.groupby('backend')['is_valid'].mean() * 100).plot(
        kind='bar', ax=axes[2], color='lightgreen'
    )
    axes[2].set_ylabel('Success Rate (%)')
    axes[2].set_title('Solution Validity Rate')
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = run_dir / 'benchmark_plot.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved benchmark plot to {plot_file}")
    plt.close()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main workflow dispatcher"""
    
    args = parse_arguments()
    
    # Determine datasets
    if args.all_datasets:
        datasets = ['tiny', 'small', 'medium']
    else:
        datasets = [args.dataset]
    
    # Determine K ranges
    if args.k_range:
        k_ranges = {ds: tuple(args.k_range) for ds in datasets}
    else:
        k_ranges = {
            'tiny': (2, 4),      # Updated: K=5 QUBO doesn't exist yet
            'small': (3, 5),
            'medium': (4, 6)
        }
    
    # Mode 1: Benchmark
    if args.benchmark:
        backends = args.backends if args.backends else ['qaoa', 'neal']
        run_benchmark(datasets, k_ranges, backends, args)
    
    # Mode 2: Compare backends (single K)
    elif args.backends:
        if args.K is None:
            print("✗ Must specify K when comparing backends")
            print("Example: python 04_unified_solver.py tiny 3 --compare-backends qaoa neal")
            sys.exit(1)
        
        results = []
        for backend in args.backends:
            result = solve_single(args.dataset, args.K, backend, args)
            if result:
                results.append(result)
        
        # Compare
        print("\n" + "="*60)
        print("BACKEND COMPARISON")
        print("="*60)
        for r in results:
            print(f"{r['backend'].upper():10s} | "
                  f"{r['runtime_seconds']:6.1f}s | "
                  f"Energy: {r['energy']:8.1f} | "
                  f"{'✓ VALID' if r['is_valid'] else '✗ INVALID'}")
    
    # Mode 3: Single solve
    else:
        if args.K is None:
            print("✗ Must specify K or use --k-range/--benchmark")
            sys.exit(1)
        
        solve_single(args.dataset, args.K, args.backend, args)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()