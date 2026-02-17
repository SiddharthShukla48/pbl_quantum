"""
QUBO Builder for Graph Coloring
================================

This file builds the QUBO matrix for exam scheduling using graph coloring.

Mathematical Formulation:
-------------------------
Variables: x[i,c] = 1 if exam i gets color c (time slot c), else 0

Constraints:
1. Each exam exactly one color: Σ_c x[i,c] = 1  for all exams i
2. Adjacent exams different colors: x[i,c] * x[j,c] = 0  for all edges (i,j), all colors c

QUBO Objective:
H = λ₁ * Σᵢ (Σ_c x[i,c] - 1)²  +  λ₂ * Σ_{(i,j)∈E} Σ_c x[i,c] * x[j,c]

Where:
- λ₁ = penalty for not assigning exactly one color (HARD constraint)
- λ₂ = penalty for adjacent exams having same color (HARD constraint)

Author: For quantum graph coloring exam scheduling research
Date: February 2026
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


def get_latest_run_dir(output_base='./output'):
    """Get the latest run directory"""
    latest_file = Path(output_base) / 'latest_run.txt'
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            return Path(f.read().strip())
    return None


class GraphColoringQUBO:
    """
    Build QUBO matrix for graph coloring problem
    
    Key concept: We're converting the graph coloring problem into a
    binary optimization problem that quantum computers can solve.
    """
    
    def __init__(self, adjacency_matrix, num_colors, lambda1=10000, lambda2=5000):
        """
        Initialize QUBO builder
        
        Args:
            adjacency_matrix: n×n adjacency matrix (1 if edge exists)
            num_colors: K = number of colors (time slots) to use
            lambda1: Penalty for not assigning exactly one color (HARD)
            lambda2: Penalty for adjacent nodes same color (HARD)
        """
        # Convert to numpy if DataFrame
        if isinstance(adjacency_matrix, pd.DataFrame):
            self.adj = adjacency_matrix.values
        else:
            self.adj = adjacency_matrix
        
        self.n = len(self.adj)  # Number of nodes (exams)
        self.K = num_colors     # Number of colors (slots)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
        # Total number of binary variables: n * K
        # Variable x[i,c] stored at index: i*K + c
        self.num_vars = self.n * self.K
        
        print(f"QUBO Builder initialized:")
        print(f"  Nodes (exams):     {self.n}")
        print(f"  Colors (slots):    {self.K}")
        print(f"  Binary variables:  {self.num_vars}")
        print(f"  QUBO matrix size:  {self.num_vars} × {self.num_vars}")
    
    def get_variable_index(self, exam_id, color):
        """
        Convert (exam, color) pair to single variable index
        
        Args:
            exam_id: Exam number (0 to n-1)
            color: Color number (0 to K-1)
            
        Returns:
            Variable index (0 to n*K-1)
        
        Example:
            5 exams, 3 colors → 15 variables
            Exam 0: vars 0,1,2    (colors 0,1,2)
            Exam 1: vars 3,4,5    (colors 0,1,2)
            Exam 2: vars 6,7,8    (colors 0,1,2)
            ...
        """
        return exam_id * self.K + color
    
    def get_exam_color(self, var_idx):
        """
        Reverse: variable index → (exam, color)
        
        Args:
            var_idx: Variable index
            
        Returns:
            tuple (exam_id, color)
        """
        exam = var_idx // self.K
        color = var_idx % self.K
        return exam, color
    
    def build_constraint_one_color_per_exam(self):
        """
        Build QUBO for: Each exam gets exactly one color
        
        Mathematical form:
            For each exam i: (Σ_c x[i,c] - 1)² = penalty if not exactly 1
        
        Expanded form:
            (Σ_c x[i,c] - 1)² = Σ_c x[i,c]² + 2·Σ_{c<c'} x[i,c]·x[i,c'] - 2·Σ_c x[i,c] + 1
        
        For binary variables (x² = x):
            = Σ_c x[i,c] + 2·Σ_{c<c'} x[i,c]·x[i,c'] - 2·Σ_c x[i,c] + const
            = -Σ_c x[i,c] + 2·Σ_{c<c'} x[i,c]·x[i,c']
        
        QUBO contributions:
            Diagonal Q[var(i,c), var(i,c)]:     -λ₁
            Off-diag Q[var(i,c), var(i,c')]:   +2λ₁  (same exam, different colors)
        
        Returns:
            Q matrix (n*K × n*K)
        """
        Q = np.zeros((self.num_vars, self.num_vars))
        
        print("\nBuilding Constraint C1: One color per exam...")
        
        for exam in range(self.n):
            # Get all variable indices for this exam
            vars_for_exam = [self.get_variable_index(exam, c) for c in range(self.K)]
            
            # Diagonal terms: -λ₁ for each variable
            for var in vars_for_exam:
                Q[var, var] += -self.lambda1
            
            # Off-diagonal terms: +2λ₁ for each pair
            for i in range(len(vars_for_exam)):
                for j in range(i+1, len(vars_for_exam)):
                    var_i = vars_for_exam[i]
                    var_j = vars_for_exam[j]
                    
                    Q[var_i, var_j] += 2 * self.lambda1
                    Q[var_j, var_i] += 2 * self.lambda1  # Symmetric
        
        print(f"  ✓ Added penalties for {self.n} exams")
        return Q
    
    def build_constraint_adjacent_different_colors(self):
        """
        Build QUBO for: Adjacent exams must have different colors
        
        Mathematical form:
            For each edge (i,j) and each color c:
                x[i,c] * x[j,c] = penalty if both use color c
        
        QUBO contribution:
            Off-diagonal Q[var(i,c), var(j,c)]: +λ₂
            (Penalty if both exams i and j are assigned color c)
        
        Returns:
            Q matrix (n*K × n*K)
        """
        Q = np.zeros((self.num_vars, self.num_vars))
        
        print("\nBuilding Constraint C2: Adjacent exams different colors...")
        
        num_edges = 0
        
        # For each edge in the conflict graph
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.adj[i, j] > 0:  # Edge exists
                    num_edges += 1
                    
                    # For each color
                    for c in range(self.K):
                        var_i = self.get_variable_index(i, c)
                        var_j = self.get_variable_index(j, c)
                        
                        # Add penalty for both using color c
                        Q[var_i, var_j] += self.lambda2
                        Q[var_j, var_i] += self.lambda2  # Symmetric
        
        print(f"  ✓ Added penalties for {num_edges} edges × {self.K} colors")
        return Q
    
    def build_full_qubo(self):
        """
        Build complete QUBO matrix
        
        Returns:
            Q: Full QUBO matrix (n*K × n*K)
        """
        print("\n" + "="*60)
        print("BUILDING FULL QUBO MATRIX")
        print("="*60)
        
        # Initialize
        Q = np.zeros((self.num_vars, self.num_vars))
        
        # Add constraint 1: One color per exam
        Q1 = self.build_constraint_one_color_per_exam()
        Q += Q1
        
        # Add constraint 2: Adjacent exams different colors
        Q2 = self.build_constraint_adjacent_different_colors()
        Q += Q2
        
        # Make symmetric (some solvers require this)
        Q = (Q + Q.T) / 2
        
        print("\n" + "="*60)
        print(f"QUBO Matrix Complete: {Q.shape}")
        print(f"Non-zero entries: {np.count_nonzero(Q)}")
        print("="*60)
        
        return Q
    
    def decode_solution(self, solution_bits):
        """
        Decode binary solution to color assignment
        
        Args:
            solution_bits: Binary vector of length n*K
            
        Returns:
            dict: {exam_id: color}
        """
        coloring = {}
        
        for exam in range(self.n):
            for color in range(self.K):
                var_idx = self.get_variable_index(exam, color)
                
                if solution_bits[var_idx] == 1:
                    coloring[exam] = color
                    break  # Found color for this exam
        
        return coloring
    
    def validate_solution(self, coloring):
        """
        Check if coloring satisfies all constraints
        
        Args:
            coloring: dict {exam_id: color}
            
        Returns:
            tuple (is_valid, violations)
        """
        violations = []
        
        # Check C1: Each exam has exactly one color
        if len(coloring) != self.n:
            violations.append(f"Not all exams colored: {len(coloring)}/{self.n}")
        
        # Check C2: Adjacent exams different colors
        conflict_count = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                if self.adj[i, j] > 0:  # Edge exists
                    if i in coloring and j in coloring:
                        if coloring[i] == coloring[j]:
                            conflict_count += 1
                            violations.append(f"Exams {i} and {j} both color {coloring[i]}")
        
        is_valid = len(violations) == 0
        
        return is_valid, violations
    
    def compute_energy(self, solution_bits):
        """
        Compute QUBO energy for a solution
        
        Energy = x^T Q x = Σᵢⱼ Q[i,j] * x[i] * x[j]
        
        Args:
            solution_bits: Binary vector
            
        Returns:
            float: Energy value
        """
        Q = self.build_full_qubo()
        energy = solution_bits @ Q @ solution_bits
        return energy


def load_data(data_dir):
    """Load dataset"""
    data_path = Path(data_dir)
    
    return {
        'courses': pd.read_csv(data_path / 'courses.csv'),
        'conflict_adjacency': pd.read_csv(data_path / 'conflict_adjacency.csv', index_col=0),
    }


def main():
    """
    Main workflow: Build QUBO for all dataset sizes
    """
    print("="*60)
    print("QUBO BUILDER FOR GRAPH COLORING")
    print("="*60)
    
    # Get latest run directory
    run_dir = get_latest_run_dir()
    if run_dir is None:
        print("⚠️ No run directory found. Please run 01_generate_dataset.py first.")
        return
    
    datasets_dir = run_dir / 'datasets'
    print(f"\n📁 Using run directory: {run_dir}\n")
    
    # Dataset configurations - generate multiple K values for each dataset
    configs = [
        {'size': 'tiny', 'K_range': range(2, 5)},    # K=2,3,4 for tiny
        {'size': 'small', 'K_range': range(3, 6)},   # K=3,4,5 for small
        {'size': 'medium', 'K_range': range(4, 7)},  # K=4,5,6 for medium
    ]
    
    for config in configs:
        size = config['size']
        K_range = config['K_range']
        
        data_dir = datasets_dir / f'exam_data_{size}'
        
        if not data_dir.exists():
            print(f"\n⚠ {data_dir} not found. Skipping.")
            continue
        
        # Load data once per dataset
        data = load_data(data_dir)
        
        # Generate QUBO for each K value
        for K in K_range:
            print(f"\n{'#'*60}")
            print(f"# Processing {size.upper()} dataset with K={K} colors")
            print(f"{'#'*60}")
            
            # Build QUBO
            builder = GraphColoringQUBO(
                adjacency_matrix=data['conflict_adjacency'],
                num_colors=K,
                lambda1=10000,
                lambda2=5000
            )
            
            Q = builder.build_full_qubo()
            
            # Save QUBO matrix
            output_path = Path(data_dir)
            np.save(output_path / f'qubo_matrix_K{K}.npy', Q)
            print(f"\n✓ Saved QUBO matrix to {data_dir}/qubo_matrix_K{K}.npy")
            
            # Save builder metadata
            metadata = {
                'num_exams': builder.n,
                'num_colors': builder.K,
                'num_variables': builder.num_vars,
                'lambda1': builder.lambda1,
                'lambda2': builder.lambda2,
                'qubo_shape': list(Q.shape)
            }
            
            with open(output_path / f'qubo_metadata_K{K}.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✓ Saved metadata to {data_dir}/qubo_metadata_K{K}.json")
    
    print("\n" + "="*60)
    print("QUBO BUILDING COMPLETE!")
    print("="*60)
    print(f"Generated QUBO matrices:")
    print(f"  - tiny:   K=2, 3, 4")
    print(f"  - small:  K=3, 4, 5")
    print(f"  - medium: K=4, 5, 6")
    print(f"\nNext step: Run 04_unified_solver.py --benchmark")
    print("="*60)


if __name__ == '__main__':
    main()
