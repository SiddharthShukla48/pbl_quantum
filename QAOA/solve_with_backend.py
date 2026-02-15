"""
Unified Solver for Graph Coloring QUBO
=======================================

Supports multiple quantum/quantum-inspired backends:
1. QAOA (IBM Qiskit) - Gate-based quantum
2. D-Wave Neal - Simulated annealing (classical)
3. D-Wave QPU - Quantum annealing (real hardware)
4. D-Wave Hybrid - Hybrid classical-quantum

Usage:
    python 06_solve_with_backend.py tiny 3 --backend qaoa
    python 06_solve_with_backend.py tiny 3 --backend neal
    python 06_solve_with_backend.py tiny 3 --backend dwave
    python 06_solve_with_backend.py small 4 --backend hybrid
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import argparse
import sys

# Qiskit imports (for QAOA)
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
    from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    print("⚠ D-Wave Ocean SDK not available. D-Wave backends disabled.")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Unified solver for graph coloring QUBO',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('dataset', nargs='?', default='tiny',
                       choices=['tiny', 'small', 'medium'],
                       help='Dataset size')
    
    parser.add_argument('K', nargs='?', type=int, default=None,
                       help='Number of colors')
    
    parser.add_argument('--backend', type=str, default='qaoa',
                       choices=['qaoa', 'neal', 'dwave', 'hybrid'],
                       help='Solver backend (default: qaoa)')
    
    parser.add_argument('--reps', type=int, default=2,
                       help='QAOA depth (only for qaoa backend)')
    
    parser.add_argument('--maxiter', type=int, default=100,
                       help='Max optimizer iterations (qaoa only)')
    
    parser.add_argument('--num-reads', type=int, default=1000,
                       help='Number of reads (D-Wave backends)')
    
    parser.add_argument('--annealing-time', type=int, default=20,
                       help='Annealing time in μs (D-Wave QPU only)')
    
    args = parser.parse_args()
    
    # Auto-determine K if not specified
    if args.K is None:
        K_defaults = {'tiny': 3, 'small': 4, 'medium': 5}
        args.K = K_defaults[args.dataset]
    
    return args


def load_qubo_data(data_dir, K):
    """Load QUBO matrix and metadata"""
    data_path = Path(data_dir)
    
    Q = np.load(data_path / f'qubo_matrix_K{K}.npy')
    
    with open(data_path / f'qubo_metadata_K{K}.json', 'r') as f:
        metadata = json.load(f)
    
    courses = pd.read_csv(data_path / 'courses.csv')
    adjacency = pd.read_csv(data_path / 'conflict_adjacency.csv', index_col=0).values
    
    return Q, metadata, courses, adjacency


# ============================================================================
# QAOA SOLVER (IBM Qiskit)
# ============================================================================

class QAOASolver:
    """QAOA solver using IBM Qiskit"""
    
    def __init__(self, Q, reps=2, maxiter=100):
        self.Q = Q
        self.reps = reps
        self.maxiter = maxiter
        self.num_vars = Q.shape[0]
    
    def solve(self):
        """Solve QUBO with QAOA"""
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit not installed. Run: pip install qiskit qiskit-optimization")
        
        print("\n" + "="*60)
        print("QAOA SOLVER (IBM Qiskit)")
        print("="*60)
        print(f"Variables: {self.num_vars}")
        print(f"QAOA depth (p): {self.reps}")
        print(f"Max iterations: {self.maxiter}")
        
        # Convert QUBO to QuadraticProgram
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
        
        # Create QAOA
        sampler = Sampler()
        optimizer = COBYLA(maxiter=self.maxiter)
        
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=self.reps
        )
        
        qaoa_optimizer = MinimumEigenOptimizer(qaoa)
        
        # Solve
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
            'backend': 'qaoa',
            'raw_result': result
        }


# ============================================================================
# D-WAVE SOLVERS (Neal, QPU, Hybrid)
# ============================================================================

class DWaveSolverBase:
    """Base class for D-Wave solvers"""
    
    def __init__(self, Q, num_reads=1000):
        self.Q = Q
        self.num_reads = num_reads
        self.num_vars = Q.shape[0]
    
    def qubo_to_bqm(self):
        """Convert QUBO matrix to D-Wave BQM format"""
        linear = {}
        quadratic = {}
        
        for i in range(self.num_vars):
            if self.Q[i, i] != 0:
                linear[i] = self.Q[i, i]
            
            for j in range(i+1, self.num_vars):
                if self.Q[i, j] != 0:
                    quadratic[(i, j)] = self.Q[i, j]
        
        bqm = BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')
        
        return bqm
    
    def decode_solution(self, sample):
        """Convert D-Wave sample to solution vector"""
        solution = np.zeros(self.num_vars, dtype=int)
        for i in range(self.num_vars):
            solution[i] = sample.get(i, 0)
        return solution


class NealSolver(DWaveSolverBase):
    """D-Wave Neal simulated annealing solver"""
    
    def solve(self):
        """Solve with Neal simulator"""
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK not installed. Run: pip install dwave-ocean-sdk")
        
        print("\n" + "="*60)
        print("D-WAVE NEAL SIMULATOR (Simulated Annealing)")
        print("="*60)
        print(f"Variables: {self.num_vars}")
        print(f"Num reads: {self.num_reads}")
        
        bqm = self.qubo_to_bqm()
        
        print(f"BQM variables: {bqm.num_variables}")
        print(f"BQM interactions: {bqm.num_interactions}")
        
        # Solve
        print("\n⏳ Running simulated annealing...")
        start_time = time.time()
        
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=self.num_reads)
        
        runtime = time.time() - start_time
        
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        
        print(f"✓ Completed in {runtime:.2f}s")
        print(f"Energy: {best_energy:.2f}")
        print(f"Samples: {len(sampleset)}")
        
        solution = self.decode_solution(best_sample)
        
        return {
            'solution': solution,
            'energy': best_energy,
            'runtime': runtime,
            'backend': 'neal',
            'num_samples': len(sampleset),
            'raw_result': sampleset
        }


class DWaveQPUSolver(DWaveSolverBase):
    """D-Wave quantum annealer (real hardware)"""
    
    def __init__(self, Q, num_reads=1000, annealing_time=20):
        super().__init__(Q, num_reads)
        self.annealing_time = annealing_time
    
    def solve(self):
        """Solve with D-Wave QPU"""
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK not installed")
        
        print("\n" + "="*60)
        print("D-WAVE QUANTUM ANNEALER (QPU)")
        print("="*60)
        print(f"Variables: {self.num_vars}")
        print(f"Num reads: {self.num_reads}")
        print(f"Annealing time: {self.annealing_time} μs")
        
        bqm = self.qubo_to_bqm()
        
        print("\n⏳ Connecting to D-Wave QPU...")
        
        try:
            sampler = EmbeddingComposite(DWaveSampler())
            
            # Auto-calculate chain strength
            chain_strength = max(abs(self.Q.min()), abs(self.Q.max())) * 2
            
            print(f"Solver: {sampler.child.solver.name}")
            print(f"Chain strength: {chain_strength:.2f}")
            
            start_time = time.time()
            
            sampleset = sampler.sample(
                bqm,
                num_reads=self.num_reads,
                annealing_time=self.annealing_time,
                chain_strength=chain_strength
            )
            
            runtime = time.time() - start_time
            
            best_sample = sampleset.first.sample
            best_energy = sampleset.first.energy
            
            embedding_info = sampleset.info.get('embedding_context', {})
            
            print(f"✓ QPU execution complete")
            print(f"Runtime: {runtime:.2f}s")
            print(f"Energy: {best_energy:.2f}")
            print(f"Chain length: {embedding_info.get('chain_length', 'N/A')}")
            
            solution = self.decode_solution(best_sample)
            
            return {
                'solution': solution,
                'energy': best_energy,
                'runtime': runtime,
                'backend': 'dwave_qpu',
                'num_samples': len(sampleset),
                'embedding_info': embedding_info,
                'raw_result': sampleset
            }
            
        except Exception as e:
            print(f"✗ D-Wave QPU access failed: {e}")
            print("Falling back to Neal simulator...")
            
            # Fallback to Neal
            neal_solver = NealSolver(self.Q, self.num_reads)
            return neal_solver.solve()


class HybridSolver(DWaveSolverBase):
    """D-Wave Hybrid solver (classical + quantum)"""
    
    def solve(self):
        """Solve with D-Wave Hybrid"""
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK not installed")
        
        print("\n" + "="*60)
        print("D-WAVE HYBRID SOLVER (Classical + Quantum)")
        print("="*60)
        print(f"Variables: {self.num_vars}")
        
        bqm = self.qubo_to_bqm()
        
        print(f"BQM variables: {bqm.num_variables}")
        print(f"BQM interactions: {bqm.num_interactions}")
        
        print("\n⏳ Connecting to Hybrid solver...")
        
        try:
            sampler = LeapHybridSampler()
            
            print(f"Solver: {sampler.solver.name}")
            
            start_time = time.time()
            
            sampleset = sampler.sample(bqm)
            
            runtime = time.time() - start_time
            
            best_sample = sampleset.first.sample
            best_energy = sampleset.first.energy
            
            print(f"✓ Hybrid solver complete")
            print(f"Runtime: {runtime:.2f}s")
            print(f"Energy: {best_energy:.2f}")
            
            solution = self.decode_solution(best_sample)
            
            return {
                'solution': solution,
                'energy': best_energy,
                'runtime': runtime,
                'backend': 'hybrid',
                'raw_result': sampleset
            }
            
        except Exception as e:
            print(f"✗ Hybrid solver access failed: {e}")
            print("Falling back to Neal simulator...")
            
            # Fallback to Neal
            neal_solver = NealSolver(self.Q, self.num_reads)
            return neal_solver.solve()


# ============================================================================
# SOLUTION VALIDATION
# ============================================================================

def decode_coloring(solution, metadata):
    """Decode solution to exam coloring"""
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


def validate_coloring(coloring, adjacency, metadata):
    """Validate graph coloring solution"""
    n_exams = metadata['num_exams']
    violations = []
    
    # Check C1: Each exam assigned
    if len(coloring) != n_exams:
        violations.append(f"Only {len(coloring)}/{n_exams} exams colored")
    
    # Check C2: No conflicts
    conflict_count = 0
    for i in range(n_exams):
        for j in range(i+1, n_exams):
            if adjacency[i, j] > 0:
                if i in coloring and j in coloring:
                    if coloring[i] == coloring[j]:
                        conflict_count += 1
                        violations.append(f"Conflict: Exams {i},{j} both slot {coloring[i]}")
    
    is_valid = len(violations) == 0
    
    return is_valid, conflict_count, violations


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main unified solver workflow"""
    
    args = parse_arguments()
    
    print("="*60)
    print("UNIFIED QUBO SOLVER")
    print("="*60)
    print(f"\n📊 Dataset: {args.dataset.upper()}")
    print(f"🎨 Colors (K): {args.K}")
    print(f"🔧 Backend: {args.backend.upper()}\n")
    
    # Get run directory
    output_base = Path('./output')
    latest_run_file = output_base / 'latest_run.txt'
    
    if not latest_run_file.exists():
        print("⚠ No run directory found. Run 01_generate_dataset.py first.")
        return
    
    with open(latest_run_file, 'r') as f:
        run_dir = Path(f.read().strip())
    
    datasets_dir = run_dir / 'datasets'
    solutions_dir = run_dir / 'solutions'
    solutions_dir.mkdir(exist_ok=True)
    
    # Load QUBO
    data_dir = datasets_dir / f'exam_data_{args.dataset}'
    
    if not data_dir.exists():
        print(f"✗ Error: {data_dir} not found!")
        return
    
    print("Loading QUBO...")
    Q, metadata, courses, adjacency = load_qubo_data(data_dir, args.K)
    print(f"✓ Loaded QUBO: {Q.shape}")
    print(f"  Variables: {metadata['num_variables']}")
    print(f"  Exams: {metadata['num_exams']}")
    print(f"  Colors: {metadata['num_colors']}")
    
    # Select solver
    if args.backend == 'qaoa':
        solver = QAOASolver(Q, reps=args.reps, maxiter=args.maxiter)
    
    elif args.backend == 'neal':
        solver = NealSolver(Q, num_reads=args.num_reads)
    
    elif args.backend == 'dwave':
        solver = DWaveQPUSolver(Q, num_reads=args.num_reads, annealing_time=args.annealing_time)
    
    elif args.backend == 'hybrid':
        solver = HybridSolver(Q)
    
    else:
        print(f"✗ Unknown backend: {args.backend}")
        return
    
    # Solve
    try:
        result = solver.solve()
    except Exception as e:
        print(f"\n✗ Solver failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Decode and validate
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    coloring = decode_coloring(result['solution'], metadata)
    is_valid, num_conflicts, violations = validate_coloring(coloring, adjacency, metadata)
    
    if is_valid:
        print("✓ Solution is VALID!")
        print(f"✓ All {metadata['num_exams']} exams successfully colored")
        print(f"✓ Colors used: {len(set(coloring.values()))}/{args.K}")
    else:
        print("✗ Solution INVALID")
        print(f"✗ Conflicts: {num_conflicts}")
        if violations:
            print(f"\nFirst 5 violations:")
            for v in violations[:5]:
                print(f"  - {v}")
    
    # Save results
    result_data = {
        'dataset': args.dataset,
        'K': args.K,
        'backend': args.backend,
        'num_variables': metadata['num_variables'],
        'num_exams': metadata['num_exams'],
        'runtime_seconds': result['runtime'],
        'energy': float(result['energy']),
        'is_valid': is_valid,
        'num_conflicts': num_conflicts,
        'colors_used': len(set(coloring.values())),
        'coloring': {str(k): int(v) for k, v in coloring.items()},
        'backend_params': {
            'reps': args.reps if args.backend == 'qaoa' else None,
            'num_reads': args.num_reads if args.backend in ['neal', 'dwave'] else None,
            'annealing_time': args.annealing_time if args.backend == 'dwave' else None
        }
    }
    
    results_file = solutions_dir / f'{args.backend}_results_{args.dataset}_K{args.K}.json'
    
    with open(results_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✓ Saved results to {results_file}")
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Backend:          {args.backend.upper()}")
    print(f"Problem size:     {metadata['num_variables']} variables")
    print(f"Runtime:          {result['runtime']:.2f}s")
    print(f"Energy:           {result['energy']:.2f}")
    print(f"Result:           {'✓ VALID' if is_valid else '✗ INVALID'}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()