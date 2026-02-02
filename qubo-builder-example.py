"""
QUBO Implementation Example for Exam Scheduling
Shows how to build QUBO matrices step-by-step with explanations

Author: Created for exam scheduling QUBO research
Date: February 2026

WHAT IS QUBO?
=============
QUBO (Quadratic Unconstrained Binary Optimization) is a way to express 
optimization problems for quantum computers.

Format: minimize E(x) = Σ Q[i,j] * x[i] * x[j]
where:
- x is a vector of binary variables (0 or 1)
- Q is a square matrix of penalties
- E(x) is the energy/cost function

The quantum computer finds the x that minimizes E(x).
"""

import numpy as np
from typing import Tuple, Dict

# Import our data loader
# (Uncomment when you have the file)
# from data_loader import ExamSchedulingDataLoader


class Stage1QuboBuilder:
    """
    Build QUBO for Stage 1: Timeslot Allocation
    
    GOAL: Assign each exam to a (day, timeslot) combination
    
    VARIABLES: x[course_id, day, timeslot] = 1 if course scheduled at (day, slot)
    
    Example: 10 courses, 3 days, 2 slots = 10 * 3 * 2 = 60 variables
    """
    
    def __init__(self, data):
        """
        Initialize Stage 1 QUBO builder
        
        Args:
            data: ExamSchedulingData object from loader
        """
        self.data = data
        self.num_courses = len(data.courses)
        self.num_days = data.metadata['num_days']
        self.num_slots = data.metadata['num_timeslots']
        self.total_vars = self.num_courses * self.num_days * self.num_slots
        
        print(f"Stage 1: Scheduling {self.num_courses} courses")
        print(f"         across {self.num_days} days × {self.num_slots} slots")
        print(f"         Total variables: {self.total_vars}")
    
    def get_variable_index(self, course_id: int, day: int, slot: int) -> int:
        """
        Convert (course, day, slot) to single variable index
        
        This is crucial for building QUBO matrix!
        
        Mapping formula: var_idx = course * (days * slots) + day * slots + slot
        
        Example: Course 2, Day 1, Slot 0 with 3 days, 2 slots:
        var_idx = 2 * (3 * 2) + 1 * 2 + 0 = 12 + 2 + 0 = 14
        """
        return course_id * (self.num_days * self.num_slots) + day * self.num_slots + slot
    
    def get_course_day_slot(self, var_idx: int) -> Tuple[int, int, int]:
        """
        Reverse mapping: variable index -> (course, day, slot)
        
        Useful for interpreting QUBO solutions
        """
        course_id = var_idx // (self.num_days * self.num_slots)
        remainder = var_idx % (self.num_days * self.num_slots)
        day = remainder // self.num_slots
        slot = remainder % self.num_slots
        return course_id, day, slot
    
    def build_h1_no_conflict(self, lambda_h1: float = 10000) -> np.ndarray:
        """
        CONSTRAINT H1: No Student Conflicts
        ===================================
        
        RULE: If two courses have students in common, they CAN'T be 
              scheduled at the same (day, timeslot)
        
        HOW IT WORKS:
        1. For each pair of courses (ci, cj)
        2. If conflict_matrix[ci, cj] > 0 (students in both courses)
        3. Add penalty to Q matrix for scheduling both at same time
        
        PENALTY FORMULA:
        For each (day, slot) combination:
          If x[ci, d, t] = 1 AND x[cj, d, t] = 1:
            Add penalty = lambda_h1 * num_conflicting_students
        
        In QUBO form:
          Q[var_ci, var_cj] += lambda_h1 * conflict_size
        
        Args:
            lambda_h1: Penalty weight (large number like 10000)
        
        Returns:
            Q matrix contribution (numpy array)
        """
        print("\n  Building H1: No Student Conflicts...")
        
        Q = np.zeros((self.total_vars, self.total_vars))
        conflicts_added = 0
        
        # For each pair of courses
        for ci in range(self.num_courses):
            for cj in range(ci + 1, self.num_courses):  # ci < cj (avoid duplicates)
                
                # Check if they have conflicting students
                conflict_size = self.data.conflict_matrix[ci, cj]
                
                if conflict_size > 0:
                    # These courses CANNOT be at same timeslot
                    # Add penalty for each (day, slot) combination
                    
                    for day in range(self.num_days):
                        for slot in range(self.num_slots):
                            # Get variable indices
                            var_ci = self.get_variable_index(ci, day, slot)
                            var_cj = self.get_variable_index(cj, day, slot)
                            
                            # Add penalty: if both variables = 1, add huge cost
                            penalty = lambda_h1 * conflict_size
                            Q[var_ci, var_cj] += penalty
                            conflicts_added += 1
        
        print(f"    Added {conflicts_added} conflict penalties")
        return Q
    
    def build_h3_one_exam_one_slot(self, lambda_h3: float = 10000) -> np.ndarray:
        """
        CONSTRAINT H3: Each Exam Exactly One Slot
        ==========================================
        
        RULE: Every course must be scheduled EXACTLY ONCE
              (not zero times, not two times, exactly one time)
        
        HOW IT WORKS:
        For each course c:
          Σ(all days d, all slots t) x[c, d, t] = 1
        
        QUBO ENCODING:
        We want to minimize: (Σx - 1)²
        
        Expanding: (Σx - 1)² = Σx² - 2Σx + 1
                              = Σx (since x² = x for binary)
                              - 2Σx + 1
                              = -Σx + 1
        
        Wait, that's not quadratic! We need pairs:
        
        Actually: (Σx - 1)² = (x1 + x2 + ... + xn - 1)²
                             = x1² + x2² + ... + 2x1x2 + 2x1x3 + ...
                               - 2x1 - 2x2 - ... + 1
        
        For binary variables (x² = x):
                             = x1 + x2 + ... + 2Σ(i<j) xixj - 2Σxi + 1
        
        Dropping constant:
                             = Σxi + 2Σ(i<j) xixj - 2Σxi
                             = -Σxi + 2Σ(i<j) xixj
        
        QUBO form:
        - Diagonal Q[i,i]: coefficient for xi is -1 (but we add lambda: λ*(-1))
        - Off-diagonal Q[i,j]: coefficient for xixj is 2λ
        
        Actually simpler: (Σx - 1)² has:
        - Diagonal terms: 1 (from x²=x)
        - Off-diagonal terms: 2 (from 2xixj)
        - We multiply everything by λ
        
        Args:
            lambda_h3: Penalty weight
        
        Returns:
            Q matrix contribution
        """
        print("\n  Building H3: One Exam One Slot...")
        
        Q = np.zeros((self.total_vars, self.total_vars))
        
        for course in range(self.num_courses):
            # Get all variables for this course (all possible timeslots)
            course_vars = []
            for day in range(self.num_days):
                for slot in range(self.num_slots):
                    var_idx = self.get_variable_index(course, day, slot)
                    course_vars.append(var_idx)
            
            # Add diagonal terms (coefficient = 1 for (Σx - 1)²)
            for var_i in course_vars:
                Q[var_i, var_i] += lambda_h3
            
            # Add off-diagonal terms (coefficient = 2 for (Σx - 1)²)
            for i, var_i in enumerate(course_vars):
                for var_j in course_vars[i+1:]:
                    Q[var_i, var_j] += 2 * lambda_h3
        
        print(f"    Added one-exam constraints for {self.num_courses} courses")
        return Q
    
    def build_s1_spread_exams(self, mu_s1: float = 100) -> np.ndarray:
        """
        SOFT CONSTRAINT S1: Spread Exams Over Period
        ============================================
        
        RULE: Students should have GAP between their exams for revision
        
        PENALTY STRUCTURE:
        - Same day, different slot: penalty = 5
        - Consecutive days: penalty = 3
        - 2 days apart: penalty = 1
        - ≥3 days apart: penalty = 0
        
        HOW IT WORKS:
        For each pair of courses with conflicts:
          For each pair of (day1, slot1) and (day2, slot2):
            penalty = proximity_weight * num_conflicting_students
            Q[var_c1, var_c2] += mu_s1 * penalty
        
        Args:
            mu_s1: Soft penalty weight (smaller than hard constraints)
        
        Returns:
            Q matrix contribution
        """
        print("\n  Building S1: Spread Exams...")
        
        Q = np.zeros((self.total_vars, self.total_vars))
        
        for ci in range(self.num_courses):
            for cj in range(ci + 1, self.num_courses):
                conflict_size = self.data.conflict_matrix[ci, cj]
                
                if conflict_size > 0:
                    # For each pair of timeslots
                    for d1 in range(self.num_days):
                        for d2 in range(self.num_days):
                            gap = abs(d1 - d2)
                            
                            # Determine proximity penalty
                            if gap == 0:
                                proximity = 5  # Same day (bad)
                            elif gap == 1:
                                proximity = 3  # Next day (still bad)
                            elif gap == 2:
                                proximity = 1  # 2 days (okay)
                            else:
                                proximity = 0  # ≥3 days (good, no penalty)
                            
                            if proximity > 0:
                                for t1 in range(self.num_slots):
                                    for t2 in range(self.num_slots):
                                        var_ci = self.get_variable_index(ci, d1, t1)
                                        var_cj = self.get_variable_index(cj, d2, t2)
                                        
                                        penalty = mu_s1 * conflict_size * proximity
                                        
                                        if var_ci == var_cj:
                                            Q[var_ci, var_ci] += penalty
                                        else:
                                            Q[var_ci, var_cj] += penalty
        
        print(f"    Added spread-exam penalties")
        return Q
    
    def build_full_qubo(self, lambda_h1=10000, lambda_h3=10000, mu_s1=100) -> np.ndarray:
        """
        Build Complete Stage 1 QUBO Matrix
        ==================================
        
        OBJECTIVE FUNCTION:
        E(x) = λ_H1 * H1(x) + λ_H3 * H3(x) + μ_S1 * S1(x)
        
        Where:
        - H1 = no conflicts (HARD, large penalty)
        - H3 = one exam one slot (HARD, large penalty)
        - S1 = spread exams (SOFT, small penalty)
        
        PENALTY WEIGHTS:
        - λ (lambda): Hard constraints (10000+)
        - μ (mu): Soft constraints (100-500)
        
        Returns:
            Complete Q matrix ready for QAOA/D-Wave
        """
        print(f"\n{'='*60}")
        print(f"Building Stage 1 QUBO")
        print(f"{'='*60}")
        print(f"Problem size: {self.total_vars} variables")
        print(f"QUBO matrix: {self.total_vars} × {self.total_vars}")
        
        Q = np.zeros((self.total_vars, self.total_vars))
        
        # Add all constraints
        Q += self.build_h1_no_conflict(lambda_h1)
        Q += self.build_h3_one_exam_one_slot(lambda_h3)
        Q += self.build_s1_spread_exams(mu_s1)
        
        # Make symmetric (QUBO matrices must be symmetric for quantum solvers)
        Q = (Q + Q.T) / 2
        
        # Statistics
        nonzero = np.count_nonzero(Q)
        density = nonzero / (self.total_vars ** 2) * 100
        
        print(f"\n{'='*60}")
        print(f"QUBO Statistics:")
        print(f"  Variables:       {self.total_vars}")
        print(f"  Matrix size:     {Q.shape}")
        print(f"  Non-zero:        {nonzero}")
        print(f"  Density:         {density:.2f}%")
        print(f"  Min value:       {Q.min():.2f}")
        print(f"  Max value:       {Q.max():.2f}")
        print(f"{'='*60}")
        
        return Q


# Example usage
def example_tiny_dataset():
    """
    Example: Build QUBO for TINY dataset
    
    This shows the COMPLETE workflow from data loading to QUBO matrix
    """
    print("\n" + "#"*60)
    print("# EXAMPLE: Building QUBO for TINY Dataset")
    print("#"*60)
    
    # Create a tiny example dataset
    print("\nCreating sample data...")
    
    # Sample data structure
    class SampleData:
        def __init__(self):
            # Metadata
            self.metadata = {
                'num_courses': 5,
                'num_days': 2,
                'num_timeslots': 2
            }
            
            # Conflict matrix (5x5)
            # Shows how many students are in both courses
            self.conflict_matrix = np.array([
                [0, 10, 0,  5, 0],   # Course 0 conflicts with: 1 (10 students), 3 (5 students)
                [10, 0, 15, 0, 8],   # Course 1 conflicts with: 0 (10), 2 (15), 4 (8)
                [0, 15, 0,  3, 0],   # Course 2 conflicts with: 1 (15), 3 (3)
                [5, 0,  3,  0, 12],  # Course 3 conflicts with: 0 (5), 2 (3), 4 (12)
                [0, 8,  0,  12, 0]   # Course 4 conflicts with: 1 (8), 3 (12)
            ])
            
            print("\nConflict matrix:")
            print("    C0  C1  C2  C3  C4")
            for i in range(5):
                print(f"C{i}: {self.conflict_matrix[i]}")
    
    data = SampleData()
    
    # Build QUBO
    builder = Stage1QuboBuilder(data)
    Q = builder.build_full_qubo()
    
    print("\nWhat this QUBO does:")
    print("  ✓ Prevents conflicting courses in same timeslot")
    print("  ✓ Ensures each course scheduled exactly once")
    print("  ✓ Prefers spreading exams apart (soft constraint)")
    
    print("\nNext step: Solve this QUBO with:")
    print("  - D-Wave quantum annealer")
    print("  - Qiskit QAOA")
    print("  - Classical simulated annealing")
    
    return Q


if __name__ == '__main__':
    print("="*60)
    print("QUBO Implementation Example")
    print("="*60)
    
    # Run example
    Q = example_tiny_dataset()
    
    print("\n✓ QUBO matrix built successfully!")
    print("\nYou can now:")
    print("  1. Use this Q matrix with your quantum solver")
    print("  2. Generate datasets with dataset-generator.py")
    print("  3. Load data with data-loader.py")
    print("  4. Build QUBO for your actual data")
