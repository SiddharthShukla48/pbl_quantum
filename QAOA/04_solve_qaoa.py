"""
QAOA Solver using IBM Qiskit
=============================

This script solves the graph coloring QUBO using QAOA on IBM Quantum.

Workflow:
1. Load QUBO matrix
2. Convert QUBO → Qiskit QuadraticProgram
3. Set up QAOA with sampler and classical optimizer
4. Run on simulator (FREE, unlimited) or real hardware (10 min/month)
5. Decode solution and validate

IMPORTANT: 
- Start with SIMULATOR for debugging (this script uses simulator by default)
- Once working, uncomment the real hardware section
- Monitor your 10-minute quota at: https://quantum.cloud.ibm.com

Author: For quantum graph coloring exam scheduling research
Date: February 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import matplotlib.pyplot as plt
import sys

# Qiskit imports (Compatible with both old and new versions)
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# Try new imports first (Qiskit 1.0+), fallback to old (0.x)
try:
    # Qiskit 1.0+ (requires Python 3.9+)
    from qiskit_algorithms import QAOA
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.primitives import Sampler
    QISKIT_VERSION = "1.0+"
except ImportError:
    try:
        # Qiskit 0.x (works with Python 3.8)
        from qiskit.algorithms.minimum_eigensolvers import QAOA
        from qiskit.algorithms.optimizers import COBYLA, SPSA
        from qiskit.primitives import Sampler
        QISKIT_VERSION = "0.x (new style)"
    except ImportError:
        # Very old Qiskit 0.x
        from qiskit.algorithms import QAOA
        from qiskit.algorithms.optimizers import COBYLA, SPSA
        from qiskit.utils import QuantumInstance
        from qiskit import Aer
        QISKIT_VERSION = "0.x (old style)"
        USE_OLD_BACKEND = True

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except ImportError:
    QiskitRuntimeService = None  # Optional for simulator-only usage

# For visualization
import networkx as nx


def get_latest_run_dir(output_base='./output'):
    """Get the latest run directory"""
    latest_file = Path(output_base) / 'latest_run.txt'
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            return Path(f.read().strip())
    return None


def load_qubo(data_dir, K):
    """
    Load QUBO matrix and metadata
    
    Args:
        data_dir: Path to data directory
        K: Number of colors used
        
    Returns:
        tuple (Q_matrix, metadata, courses_df, adjacency)
    """
    data_path = Path(data_dir)
    
    # Load QUBO
    Q = np.load(data_path / f'qubo_matrix_K{K}.npy')
    
    # Load metadata
    with open(data_path / f'qubo_metadata_K{K}.json', 'r') as f:
        metadata = json.load(f)
    
    # Load courses and adjacency for validation
    courses = pd.read_csv(data_path / 'courses.csv')
    adjacency = pd.read_csv(data_path / 'conflict_adjacency.csv', index_col=0).values
    
    return Q, metadata, courses, adjacency


def qubo_to_quadratic_program(Q):
    """
    Convert QUBO matrix to Qiskit QuadraticProgram
    
    Args:
        Q: QUBO matrix (n×n)
        
    Returns:
        QuadraticProgram object
    """
    n = Q.shape[0]
    
    # Create quadratic program
    qp = QuadraticProgram()
    
    # Add binary variables
    for i in range(n):
        qp.binary_var(name=f'x{i}')
    
    # Set objective: minimize x^T Q x
    # Qiskit expects: linear terms + quadratic terms
    
    # Extract linear (diagonal) and quadratic (off-diagonal) terms
    linear = {}
    quadratic = {}
    
    for i in range(n):
        if Q[i, i] != 0:
            linear[f'x{i}'] = Q[i, i]
        
        for j in range(i+1, n):
            if Q[i, j] != 0:
                # QUBO has x[i]*x[j] with coefficient Q[i,j]
                # But in symmetric matrix, we have Q[i,j] + Q[j,i]
                # So use the sum
                coeff = Q[i, j] + Q[j, i]
                quadratic[(f'x{i}', f'x{j}')] = coeff
    
    # Set objective
    qp.minimize(linear=linear, quadratic=quadratic)
    
    return qp


def solve_with_qaoa_simulator(qp, reps=2, maxiter=100):
    """
    Solve QUBO using QAOA on local simulator
    
    Args:
        qp: QuadraticProgram
        reps: QAOA depth (p parameter, number of layers)
        maxiter: Max iterations for classical optimizer
        
    Returns:
        OptimizationResult
    """
    print("\n" + "="*60)
    print("SOLVING WITH QAOA (LOCAL SIMULATOR)")
    print("="*60)
    print(f"Problem size: {qp.get_num_vars()} variables")
    print(f"QAOA depth (p): {reps}")
    print(f"Classical optimizer: COBYLA with {maxiter} iterations")
    print("\nThis may take 1-5 minutes depending on problem size...")
    
    # Create local sampler (simulator)
    sampler = Sampler()
    
    # Classical optimizer
    # COBYLA is gradient-free, good for noisy optimization
    optimizer = COBYLA(maxiter=maxiter)
    
    # Create QAOA instance
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps  # Number of QAOA layers
    )
    
    # Wrap in MinimumEigenOptimizer
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    
    # Solve
    result = qaoa_optimizer.solve(qp)
    
    print("\n✓ QAOA optimization complete!")
    return result


def solve_with_qaoa_real_hardware(qp, reps=1, maxiter=50):
    """
    Solve QUBO using QAOA on REAL IBM quantum hardware
    
    ⚠ WARNING: This uses your 10-minute monthly quota!
    
    Args:
        qp: QuadraticProgram
        reps: QAOA depth (use p=1 for real hardware to save time)
        maxiter: Fewer iterations for real hardware
        
    Returns:
        OptimizationResult
    """
    print("\n" + "="*60)
    print("SOLVING WITH QAOA (REAL IBM QUANTUM HARDWARE)")
    print("="*60)
    print("⚠ This will use your 10-minute monthly quota!")
    print(f"Problem size: {qp.get_num_vars()} variables")
    print(f"QAOA depth (p): {reps}")
    
    # Initialize IBM Quantum service
    # First time: QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
    service = QiskitRuntimeService(channel="ibm_quantum")
    
    # Get least busy backend
    backend = service.least_busy(operational=True, simulator=False)
    print(f"Selected backend: {backend.name}")
    print(f"Queue length: {backend.status().pending_jobs}")
    
    # Create sampler for real hardware
    from qiskit_ibm_runtime import Sampler as RuntimeSampler
    sampler = RuntimeSampler(backend=backend)
    
    # Use SPSA optimizer (better for noisy hardware)
    optimizer = SPSA(maxiter=maxiter)
    
    # Create QAOA
    qaoa = QAOA(
        sampler=sampler,
        optimizer=optimizer,
        reps=reps
    )
    
    # Solve
    qaoa_optimizer = MinimumEigenOptimizer(qaoa)
    
    print("\n🚀 Submitting job to IBM Quantum...")
    result = qaoa_optimizer.solve(qp)
    
    print("✓ Job complete!")
    return result


def decode_solution(result, metadata):
    """
    Decode QAOA result to exam coloring
    
    Args:
        result: OptimizationResult from QAOA
        metadata: Dict with num_exams, num_colors
        
    Returns:
        dict {exam_id: color}
    """
    solution_bits = result.x  # Binary solution vector
    
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
    """
    Validate that solution satisfies constraints
    
    Args:
        coloring: Dict {exam_id: color}
        adjacency: Adjacency matrix
        metadata: Problem metadata
        
    Returns:
        tuple (is_valid, num_violations, details)
    """
    n_exams = metadata['num_exams']
    violations = []
    
    # Check C1: Each exam assigned exactly one color
    if len(coloring) != n_exams:
        violations.append(f"Only {len(coloring)}/{n_exams} exams colored")
    
    # Check C2: Adjacent exams have different colors
    conflict_count = 0
    for i in range(n_exams):
        for j in range(i+1, n_exams):
            if adjacency[i, j] > 0:  # Edge exists
                if i in coloring and j in coloring:
                    if coloring[i] == coloring[j]:
                        conflict_count += 1
                        violations.append(f"Conflict: Exams {i} and {j} both slot {coloring[i]}")
    
    is_valid = len(violations) == 0
    
    return is_valid, conflict_count, violations


def visualize_solution(coloring, adjacency, courses_df, title="QAOA Solution"):
    """
    Visualize the colored graph
    
    Args:
        coloring: Dict {exam_id: color}
        adjacency: Adjacency matrix
        courses_df: DataFrame with course info
        title: Plot title
    """
    # Build NetworkX graph
    G = nx.Graph()
    
    n = len(adjacency)
    for i in range(n):
        G.add_node(i, label=courses_df.iloc[i]['course_code'])
    
    for i in range(n):
        for j in range(i+1, n):
            if adjacency[i, j] > 0:
                G.add_edge(i, j)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Color map
    color_palette = ['red', 'blue', 'green', 'yellow', 'purple', 
                    'orange', 'pink', 'cyan', 'brown', 'lime']
    
    node_colors = [color_palette[coloring.get(i, 0) % len(color_palette)] 
                   for i in range(n)]
    
    # Draw
    plt.figure(figsize=(12, 8))
    
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=1500,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray')
    
    # Labels
    labels = {i: f"{G.nodes[i]['label']}\nSlot {coloring.get(i, '?')}" 
              for i in range(n)}
    nx.draw_networkx_labels(G, pos, labels, font_size=9, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()


def main():
    """
    Main QAOA solving workflow
    """
    print("="*60)
    print("QAOA SOLVER FOR EXAM SCHEDULING")
    print("="*60)
    
    # Parse command-line arguments
    if len(sys.argv) > 1:
        dataset = sys.argv[1].lower()
    else:
        dataset = 'tiny'  # Default
    
    if len(sys.argv) > 2:
        K = int(sys.argv[2])
    else:
        # Recommended K values
        K_defaults = {'tiny': 3, 'small': 4, 'medium': 5}
        K = K_defaults.get(dataset, 3)
    
    print(f"\n📊 Dataset: {dataset.upper()}")
    print(f"🎨 Colors (K): {K}\n")
    
    # Get latest run directory
    run_dir = get_latest_run_dir()
    if run_dir is None:
        print("⚠️ No run directory found. Please run 01_generate_dataset.py first.")
        return
    
    datasets_dir = run_dir / 'datasets'
    viz_dir = run_dir / 'visualizations'
    solutions_dir = run_dir / 'solutions'
    viz_dir.mkdir(exist_ok=True)
    solutions_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Using run directory: {run_dir}")
    print(f"📁 Solutions will be saved to: {solutions_dir}\n")
    
    # Load QUBO
    print("\nLoading QUBO...")
    Q, metadata, courses, adjacency = load_qubo(datasets_dir / f'exam_data_{dataset}', K)
    print(f"✓ Loaded QUBO: {Q.shape}")
    
    # Convert to QuadraticProgram
    print("\nConverting to Qiskit format...")
    qp = qubo_to_quadratic_program(Q)
    print(f"✓ Created QuadraticProgram with {qp.get_num_vars()} variables")
    
    # Solve with QAOA (SIMULATOR)
    result = solve_with_qaoa_simulator(qp, reps=2, maxiter=100)
    
    # Print result
    print("\n" + "="*60)
    print("QAOA RESULT")
    print("="*60)
    print(f"Objective value: {result.fval}")
    print(f"Solution vector: {result.x}")
    
    # Decode solution
    coloring = decode_solution(result, metadata)
    print(f"\nDecoded coloring:")
    for exam, color in sorted(coloring.items()):
        course_code = courses.iloc[exam]['course_code']
        print(f"  {course_code} (Exam {exam}) → Slot {color}")
    
    # Validate
    is_valid, num_conflicts, violations = validate_solution(coloring, adjacency, metadata)
    
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    if is_valid:
        print("✓ Solution is VALID! All constraints satisfied.")
        print(f"✓ Colors used: {len(set(coloring.values()))}/{K}")
    else:
        print("✗ Solution INVALID")
        print(f"✗ Number of conflicts: {num_conflicts}")
        print("\nViolations:")
        for v in violations[:10]:  # Show first 10
            print(f"  - {v}")
    
    # Visualize
    visualize_solution(coloring, adjacency, courses, 
                      title=f"QAOA Solution ({dataset.upper()}, K={K})")
    plt.savefig(viz_dir / f'qaoa_solution_{dataset}_K{K}.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {viz_dir}/qaoa_solution_{dataset}_K{K}.png")
    
    # Save solution to JSON for room assignment step
    solution_path = solutions_dir / f'qaoa_solution_{dataset}_K{K}.json'
    with open(solution_path, 'w') as f:
        json.dump(coloring, f, indent=2)
    print(f"✓ Saved solution to {solution_path}")
    
    plt.show()
    
    # Instructions for real hardware
    print("\n" + "="*60)
    print("TO RUN ON REAL IBM QUANTUM HARDWARE:")
    print("="*60)
    print("1. Get IBM Quantum account: https://quantum.cloud.ibm.com")
    print("2. Get your API token from account settings")
    print("3. Save token (first time only):")
    print("   from qiskit_ibm_runtime import QiskitRuntimeService")
    print('   QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")')
    print("4. In this script, comment out simulator section and uncomment:")
    print("   result = solve_with_qaoa_real_hardware(qp, reps=1, maxiter=30)")
    print("\n⚠ Real hardware uses your 10-minute monthly quota!")
    print("⚠ Start with TINY dataset (5 exams, 15 vars, ~1-2 min runtime)")


if __name__ == '__main__':
    main()
