"""
QAOA Solver with Flexible K Selection
======================================

Usage:
    python 04_solve_qaoa.py                    # tiny dataset, K=3 (default)
    python 04_solve_qaoa.py tiny 4             # tiny dataset, K=4
    python 04_solve_qaoa.py small 3            # small dataset, K=3
    python 04_solve_qaoa.py medium 5 --reps 1  # medium, K=5, p=1 layer

This allows you to:
- Test different chromatic numbers
- Find maximum solvable size for your hardware
- Compare solution quality across K values
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import sys
import time
import argparse
import psutil  # For memory tracking

# Qiskit imports
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

try:
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.primitives import Sampler
    QISKIT_VERSION = "1.0+"
except ImportError:
    try:
        from qiskit.algorithms.minimum_eigensolvers import QAOA
        from qiskit.algorithms.optimizers import COBYLA, SPSA
        from qiskit.primitives import Sampler
        QISKIT_VERSION = "0.x (new)"
    except ImportError:
        from qiskit.algorithms import QAOA
        from qiskit.algorithms.optimizers import COBYLA, SPSA
        from qiskit.utils import QuantumInstance
        from qiskit import Aer
        QISKIT_VERSION = "0.x (old)"


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='QAOA solver for exam scheduling graph coloring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 04_solve_qaoa.py                        # Default: tiny, K=3
  python 04_solve_qaoa.py tiny 4                 # Tiny with 4 colors
  python 04_solve_qaoa.py small 3 --reps 1       # Small, K=3, p=1
  python 04_solve_qaoa.py tiny 5 --maxiter 50    # Fewer optimizer iterations
  python 04_solve_qaoa.py small 4 --timeout 300  # 5 minute timeout
        """
    )
    
    parser.add_argument('dataset', nargs='?', default='tiny',
                       choices=['tiny', 'small', 'medium'],
                       help='Dataset size (default: tiny)')
    
    parser.add_argument('K', nargs='?', type=int, default=None,
                       help='Number of colors (default: auto based on dataset)')
    
    parser.add_argument('--reps', type=int, default=2,
                       help='QAOA depth/layers (default: 2). Use 1 for larger problems.')
    
    parser.add_argument('--maxiter', type=int, default=100,
                       help='Max optimizer iterations (default: 100)')
    
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout in seconds (default: 600 = 10 min)')
    
    parser.add_argument('--optimizer', choices=['cobyla', 'spsa'], default='cobyla',
                       help='Classical optimizer (default: cobyla)')
    
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization to save time')
    
    args = parser.parse_args()
    
    # Auto-determine K if not specified
    if args.K is None:
        K_defaults = {'tiny': 3, 'small': 4, 'medium': 5}
        args.K = K_defaults[args.dataset]
    
    return args


def get_system_resources():
    """Get current system resource usage"""
    process = psutil.Process()
    return {
        'cpu_percent': psutil.cpu_percent(interval=0.1),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'memory_percent': process.memory_percent()
    }


def estimate_problem_complexity(num_vars, K):
    """
    Estimate if problem is solvable on current system
    
    Returns:
        dict with recommendations
    """
    recommendations = {
        'solvable': True,
        'warnings': [],
        'recommended_reps': 2,
        'recommended_maxiter': 100
    }
    
    # Memory estimate: QUBO matrix is num_vars x num_vars (float64)
    qubo_memory_mb = (num_vars ** 2) * 8 / (1024 * 1024)
    
    # Circuit depth estimate (rough)
    circuit_qubits = num_vars
    
    # Rules of thumb based on simulator limitations
    if num_vars > 100:
        recommendations['solvable'] = False
        recommendations['warnings'].append(
            f"⚠ {num_vars} variables too large for classical simulator"
        )
        recommendations['warnings'].append(
            "  Recommendation: Reduce K or use smaller dataset"
        )
    
    elif num_vars > 50:
        recommendations['warnings'].append(
            f"⚠ {num_vars} variables is challenging. Expected runtime: 10-30 min"
        )
        recommendations['recommended_reps'] = 1
        recommendations['recommended_maxiter'] = 50
    
    elif num_vars > 30:
        recommendations['warnings'].append(
            f"⚠ {num_vars} variables may be slow. Expected runtime: 2-10 min"
        )
        recommendations['recommended_reps'] = 1
    
    # Check available memory
    available_memory = psutil.virtual_memory().available / (1024 * 1024)
    if qubo_memory_mb > available_memory * 0.5:
        recommendations['solvable'] = False
        recommendations['warnings'].append(
            f"⚠ Insufficient memory! Need ~{qubo_memory_mb:.0f} MB, have {available_memory:.0f} MB"
        )
    
    return recommendations


def load_qubo(data_dir, K):
    """Load QUBO matrix and metadata"""
    data_path = Path(data_dir)
    
    Q = np.load(data_path / f'qubo_matrix_K{K}.npy')
    
    with open(data_path / f'qubo_metadata_K{K}.json', 'r') as f:
        metadata = json.load(f)
    
    courses = pd.read_csv(data_path / 'courses.csv')
    adjacency = pd.read_csv(data_path / 'conflict_adjacency.csv', index_col=0).values
    
    return Q, metadata, courses, adjacency


def qubo_to_quadratic_program(Q):
    """Convert QUBO matrix to Qiskit QuadraticProgram"""
    n = Q.shape[0]
    
    qp = QuadraticProgram()
    
    for i in range(n):
        qp.binary_var(name=f'x{i}')
    
    linear = {}
    quadratic = {}
    
    for i in range(n):
        if Q[i, i] != 0:
            linear[f'x{i}'] = Q[i, i]
        
        for j in range(i+1, n):
            if Q[i, j] != 0:
                quadratic[(f'x{i}', f'x{j}')] = 2 * Q[i, j]
    
    qp.minimize(linear=linear, quadratic=quadratic)
    
    return qp


def solve_with_qaoa_simulator(qp, reps=2, maxiter=100, optimizer_name='cobyla', timeout=600):
    """
    Solve QUBO with QAOA on local simulator
    
    Returns:
        result, runtime_seconds
    """
    print("\n" + "="*60)
    print("SOLVING WITH QAOA (LOCAL SIMULATOR)")
    print("="*60)
    print(f"Problem size: {qp.get_num_vars()} variables")
    print(f"QAOA depth (p): {reps}")
    print(f"Optimizer: {optimizer_name.upper()}")
    print(f"Max iterations: {maxiter}")
    print(f"Timeout: {timeout}s")
    
    resources_before = get_system_resources()
    print(f"\nSystem resources:")
    print(f"  Memory: {resources_before['memory_mb']:.1f} MB ({resources_before['memory_percent']:.1f}%)")
    print(f"  CPU: {resources_before['cpu_percent']:.1f}%")
    
    print("\n⏳ Starting optimization (this may take several minutes)...")
    
    # Create sampler
    sampler = Sampler()
    
    # Select optimizer
    if optimizer_name == 'cobyla':
        optimizer = COBYLA(maxiter=maxiter)
    else:
        optimizer = SPSA(maxiter=maxiter)
    
    # Create QAOA
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps
    )
    
    # Wrap in MinimumEigenOptimizer
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    
    # Solve with timeout
    start_time = time.time()
    
    try:
        # Note: qiskit doesn't have built-in timeout, so we track manually
        result = qaoa_optimizer.solve(qp)
        runtime = time.time() - start_time
        
        if runtime > timeout:
            print(f"\n⚠ WARNING: Exceeded timeout ({runtime:.1f}s > {timeout}s)")
        
    except Exception as e:
        print(f"\n✗ Error during optimization: {e}")
        raise
    
    resources_after = get_system_resources()
    memory_used = resources_after['memory_mb'] - resources_before['memory_mb']
    
    print(f"\n✓ Optimization complete!")
    print(f"  Runtime: {runtime:.1f}s")
    print(f"  Memory used: {memory_used:.1f} MB")
    print(f"  Final memory: {resources_after['memory_mb']:.1f} MB")
    
    return result, runtime


def decode_solution(result, metadata):
    """Decode QAOA result to exam coloring"""
    solution_bits = result.x
    
    n_exams = metadata['num_exams']
    K = metadata['num_colors']
    
    coloring = {}
    
    for exam in range(n_exams):
        for color in range(K):
            var_idx = exam * K + color
            
            if solution_bits[var_idx] == 1:
                coloring[exam] = color
                break
    
    return coloring


def validate_solution(coloring, adjacency, metadata):
    """Validate solution"""
    n_exams = metadata['num_exams']
    violations = []
    
    # C1: Each exam assigned
    if len(coloring) != n_exams:
        violations.append(f"Only {len(coloring)}/{n_exams} exams colored")
    
    # C2: No conflicts
    conflict_count = 0
    for i in range(n_exams):
        for j in range(i+1, n_exams):
            if adjacency[i, j] > 0:
                if i in coloring and j in coloring:
                    if coloring[i] == coloring[j]:
                        conflict_count += 1
                        violations.append(f"Conflict: Exams {i} and {j} both slot {coloring[i]}")
    
    is_valid = len(violations) == 0
    
    return is_valid, conflict_count, violations


def save_results(result_data, output_path):
    """Save detailed results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"✓ Saved results to {output_path}")


def main():
    """Main QAOA solving workflow"""
    
    # Parse arguments
    args = parse_arguments()
    
    print("="*60)
    print("QAOA SOLVER FOR EXAM SCHEDULING")
    print("="*60)
    print(f"\n📊 Dataset: {args.dataset.upper()}")
    print(f"🎨 Colors (K): {args.K}")
    print(f"📐 QAOA depth (p): {args.reps}")
    print(f"🔧 Optimizer: {args.optimizer.upper()}")
    print(f"⚙️  Max iterations: {args.maxiter}")
    print(f"⏱️  Timeout: {args.timeout}s\n")
    
    # Get latest run directory
    output_base = Path('./output')
    latest_run_file = output_base / 'latest_run.txt'
    
    if not latest_run_file.exists():
        print("⚠️ No run directory found. Please run 01_generate_dataset.py first.")
        return
    
    with open(latest_run_file, 'r') as f:
        run_dir = Path(f.read().strip())
    
    datasets_dir = run_dir / 'datasets'
    solutions_dir = run_dir / 'solutions'
    viz_dir = run_dir / 'visualizations'
    solutions_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)
    
    print(f"📁 Using run directory: {run_dir}\n")
    
    # Load QUBO
    data_dir = datasets_dir / f'exam_data_{args.dataset}'
    
    if not data_dir.exists():
        print(f"✗ Error: {data_dir} not found!")
        print("Available datasets:")
        for d in datasets_dir.glob('exam_data_*'):
            print(f"  - {d.name}")
        return
    
    print("Loading QUBO...")
    Q, metadata, courses, adjacency = load_qubo(data_dir, args.K)
    num_vars = metadata['num_variables']
    print(f"✓ Loaded QUBO: {Q.shape}")
    print(f"  Variables: {num_vars}")
    print(f"  Exams: {metadata['num_exams']}")
    print(f"  Colors: {metadata['num_colors']}")
    
    # Check if problem is solvable
    estimation = estimate_problem_complexity(num_vars, args.K)
    
    if estimation['warnings']:
        print("\n" + "="*60)
        print("PROBLEM COMPLEXITY ASSESSMENT")
        print("="*60)
        for warning in estimation['warnings']:
            print(warning)
        
        if not estimation['solvable']:
            print("\n✗ Problem too large for this system!")
            print("\nSuggestions:")
            print(f"  1. Try smaller K (currently K={args.K})")
            print(f"  2. Use smaller dataset (currently {args.dataset})")
            print(f"  3. Try K={args.K-1} or K={args.K-2}")
            return
        
        # Adjust parameters based on recommendations
        if args.reps > estimation['recommended_reps']:
            print(f"\n📝 Adjusting QAOA depth: {args.reps} → {estimation['recommended_reps']}")
            args.reps = estimation['recommended_reps']
        
        if args.maxiter > estimation['recommended_maxiter']:
            print(f"📝 Adjusting max iterations: {args.maxiter} → {estimation['recommended_maxiter']}")
            args.maxiter = estimation['recommended_maxiter']
    
    # Convert to QuadraticProgram
    print("\nConverting to Qiskit format...")
    qp = qubo_to_quadratic_program(Q)
    print(f"✓ Created QuadraticProgram with {qp.get_num_vars()} variables")
    
    # Solve
    try:
        result, runtime = solve_with_qaoa_simulator(
            qp, 
            reps=args.reps,
            maxiter=args.maxiter,
            optimizer_name=args.optimizer,
            timeout=args.timeout
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user!")
        return
    except Exception as e:
        print(f"\n✗ Solver failed: {e}")
        return
    
    # Decode and validate
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Objective value: {result.fval}")
    
    coloring = decode_solution(result, metadata)
    is_valid, num_conflicts, violations = validate_solution(coloring, adjacency, metadata)
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    if is_valid:
        print("✓ Solution is VALID! All constraints satisfied.")
        print(f"✓ All {metadata['num_exams']} exams successfully colored")
        print(f"✓ Colors used: {len(set(coloring.values()))}/{args.K}")
    else:
        print("✗ Solution INVALID")
        print(f"✗ Conflicts: {num_conflicts}")
        if num_conflicts > 0:
            print("\nFirst 5 violations:")
            for v in violations[:5]:
                print(f"  - {v}")
    
    # Save detailed results
    result_data = {
        'dataset': args.dataset,
        'K': args.K,
        'num_variables': num_vars,
        'num_exams': metadata['num_exams'],
        'qaoa_reps': args.reps,
        'optimizer': args.optimizer,
        'maxiter': args.maxiter,
        'runtime_seconds': runtime,
        'objective_value': float(result.fval),
        'is_valid': is_valid,
        'num_conflicts': num_conflicts,
        'colors_used': len(set(coloring.values())),
        'coloring': {str(k): int(v) for k, v in coloring.items()},
        'system_info': {
            'qiskit_version': QISKIT_VERSION,
            'final_memory_mb': get_system_resources()['memory_mb']
        }
    }
    
    results_file = solutions_dir / f'qaoa_results_{args.dataset}_K{args.K}_p{args.reps}.json'
    save_results(result_data, results_file)
    
    # Save coloring for room assignment
    coloring_file = solutions_dir / f'qaoa_solution_{args.dataset}_K{args.K}.json'
    with open(coloring_file, 'w') as f:
        json.dump({str(k): int(v) for k, v in coloring.items()}, f, indent=2)
    print(f"✓ Saved coloring to {coloring_file}")
    
    # Print performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Problem size:     {num_vars} variables ({metadata['num_exams']} exams × {args.K} colors)")
    print(f"Runtime:          {runtime:.1f}s ({runtime/60:.1f} minutes)")
    print(f"Variables/second: {num_vars/runtime:.1f}")
    print(f"Result:           {'✓ VALID' if is_valid else '✗ INVALID'}")
    
    if not args.no_viz:
        # Simple visualization
        print("\n📊 Generating visualization...")
        import networkx as nx
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Graph structure
        G = nx.Graph()
        for i in range(metadata['num_exams']):
            G.add_node(i)
        for i in range(metadata['num_exams']):
            for j in range(i+1, metadata['num_exams']):
                if adjacency[i, j] > 0:
                    G.add_edge(i, j, weight=adjacency[i, j])
        
        pos = nx.spring_layout(G, seed=42)
        
        # Original graph
        nx.draw(G, pos, ax=ax1, with_labels=True, node_color='lightblue',
                node_size=500, font_size=10, edge_color='gray')
        ax1.set_title(f"Conflict Graph ({metadata['num_exams']} exams)")
        
        # Colored solution
        node_colors = [coloring.get(i, -1) for i in range(metadata['num_exams'])]
        nx.draw(G, pos, ax=ax2, with_labels=True, node_color=node_colors,
                node_size=500, font_size=10, edge_color='gray', cmap='Set3')
        ax2.set_title(f"QAOA Solution (K={args.K}, {'Valid' if is_valid else 'Invalid'})")
        
        plt.tight_layout()
        viz_file = viz_dir / f'qaoa_solution_{args.dataset}_K{args.K}_p{args.reps}.png'
        plt.savefig(viz_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization to {viz_file}")
        plt.close()
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print(f"✓ Solution saved to: {coloring_file}")
    print(f"✓ Detailed results: {results_file}")
    
    if is_valid:
        print("\nTo assign rooms:")
        print(f"  python 05_assign_rooms.py {args.dataset} {args.K}")
    else:
        print("\n⚠ Solution has conflicts. Try:")
        print(f"  1. Increase K: python 04_solve_qaoa.py {args.dataset} {args.K + 1}")
        print(f"  2. More QAOA layers: python 04_solve_qaoa.py {args.dataset} {args.K} --reps 3")
        print(f"  3. More optimizer iterations: python 04_solve_qaoa.py {args.dataset} {args.K} --maxiter 200")


if __name__ == '__main__':
    main()
