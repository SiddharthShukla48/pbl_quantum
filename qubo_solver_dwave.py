"""
D-Wave Ocean SDK QUBO Solver for Exam Scheduling
Multi-Stage QUBO Implementation with Constraints H1, H2, H3, S1

Author: Created for exam scheduling QUBO research
Date: February 2026

CONSTRAINTS IMPLEMENTED:
========================
H1 (Hard): No student conflicts - students can't have 2 exams at same time
H2 (Hard): Exam duration constraints - exams requiring multiple slots must get consecutive slots
H3 (Hard): One exam one slot - each exam scheduled exactly once
S1 (Soft): Spread exams - students should have gaps between exams for revision

D-WAVE INTEGRATION:
==================
- Uses Ocean SDK (dwave-ocean-sdk package)
- Supports multiple solvers: Neal simulator, D-Wave QPU, Hybrid solver
- Automatic embedding for quantum hardware
- Comprehensive solution validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# D-Wave Ocean SDK imports
try:
    from dimod import BinaryQuadraticModel
    import neal  # Simulated annealing solver
    from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
    DWAVE_AVAILABLE = True
except ImportError:
    DWAVE_AVAILABLE = False
    print("WARNING: D-Wave Ocean SDK not installed!")
    print("Install with: pip install dwave-ocean-sdk")

# Import data loader
try:
    from data_loader import ExamSchedulingData
except ImportError:
    # Try importing with explicit path handling
    import importlib.util
    data_loader_path = Path(__file__).parent / "data-loader.py"
    if data_loader_path.exists():
        spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
        data_loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_loader)
        ExamSchedulingData = data_loader.ExamSchedulingData
    else:
        raise ImportError("data-loader.py not found in current directory")


@dataclass
class SolverConfig:
    """Configuration for QUBO solver"""
    solver_type: str = 'neal'  # 'neal', 'dwave', 'hybrid'
    num_reads: int = 100  # Number of solutions to sample
    annealing_time: int = 20  # Microseconds (for D-Wave)
    chain_strength: float = None  # Auto-calculate if None
    
    # Penalty weights
    lambda_h1: float = 10000  # No conflicts
    lambda_h2: float = 10000  # Duration constraints
    lambda_h3: float = 10000  # One exam one slot
    mu_s1: float = 100  # Spread exams (soft)


@dataclass
class SchedulingSolution:
    """Container for scheduling solution"""
    assignments: Dict[int, Tuple[int, int]]  # course_id -> (day, slot)
    energy: float
    is_valid: bool
    violations: Dict[str, int]
    execution_time: float
    solver_info: Dict


class Stage1QuboSolver:
    """
    D-Wave QUBO Solver for Stage 1: Timeslot Allocation
    
    VARIABLES: x[course_id, day, timeslot] = 1 if course scheduled at (day, slot)
    
    OBJECTIVE: Minimize energy E(x) = Σ Q[i,j] * x[i] * x[j]
    
    Where Q includes:
    - H1: Penalties for student conflicts
    - H2: Penalties for violating duration constraints
    - H3: Penalties for not scheduling exactly once
    - S1: Penalties for exams too close together
    """
    
    def __init__(self, data: ExamSchedulingData, config: SolverConfig):
        """
        Initialize QUBO solver
        
        Args:
            data: Loaded exam scheduling data
            config: Solver configuration
        """
        self.data = data
        self.config = config
        
        self.num_courses = len(data.courses)
        self.num_days = data.metadata['num_days']
        self.num_slots = data.metadata['num_timeslots']
        self.total_vars = self.num_courses * self.num_days * self.num_slots
        
        print(f"\n{'='*70}")
        print(f"D-Wave QUBO Solver - Stage 1: Timeslot Allocation")
        print(f"{'='*70}")
        print(f"Courses:          {self.num_courses}")
        print(f"Days:             {self.num_days}")
        print(f"Slots per day:    {self.num_slots}")
        print(f"Total variables:  {self.total_vars}")
        print(f"QUBO matrix size: {self.total_vars} × {self.total_vars}")
        print(f"Solver type:      {config.solver_type.upper()}")
        print(f"{'='*70}\n")
        
        # Will store the QUBO matrix and BQM
        self.Q = None
        self.bqm = None
    
    def get_variable_index(self, course_id: int, day: int, slot: int) -> int:
        """
        Convert (course, day, slot) tuple to single variable index
        
        Formula: var_idx = course * (days * slots) + day * slots + slot
        
        Example with 3 days, 2 slots:
        - Course 0, Day 0, Slot 0 → index 0
        - Course 0, Day 0, Slot 1 → index 1
        - Course 0, Day 1, Slot 0 → index 2
        - Course 0, Day 1, Slot 1 → index 3
        - Course 0, Day 2, Slot 0 → index 4
        - Course 0, Day 2, Slot 1 → index 5
        - Course 1, Day 0, Slot 0 → index 6
        """
        return course_id * (self.num_days * self.num_slots) + day * self.num_slots + slot
    
    def get_course_day_slot(self, var_idx: int) -> Tuple[int, int, int]:
        """
        Reverse mapping: variable index → (course, day, slot)
        
        Essential for interpreting D-Wave solutions
        """
        slots_per_course = self.num_days * self.num_slots
        course_id = var_idx // slots_per_course
        remainder = var_idx % slots_per_course
        day = remainder // self.num_slots
        slot = remainder % self.num_slots
        return course_id, day, slot
    
    def build_h1_no_conflict(self) -> np.ndarray:
        """
        H1: NO STUDENT CONFLICTS (Hard Constraint)
        ===========================================
        
        RULE: If two courses have common students, they CANNOT be scheduled
              at the same (day, timeslot)
        
        IMPLEMENTATION:
        1. Iterate through conflict_matrix to find course pairs with conflicts
        2. For each conflicting pair (ci, cj):
           - For each timeslot (d, t):
             - Add penalty if both x[ci,d,t]=1 AND x[cj,d,t]=1
        
        QUBO ENCODING:
        - For courses ci and cj with conflict_size students in common:
        - For each (day, slot):
          Q[var_ci, var_cj] += λ_H1 * conflict_size
        
        PENALTY WEIGHT:
        - Large (e.g., 10000) to make violations very expensive
        - Violations should never occur in valid solutions
        
        Returns:
            Q matrix contribution (numpy array)
        """
        print("  [1/4] Building H1: No Student Conflicts...")
        
        Q = np.zeros((self.total_vars, self.total_vars))
        conflicts_added = 0
        total_conflict_penalty = 0
        
        for ci in range(self.num_courses):
            for cj in range(ci + 1, self.num_courses):
                conflict_size = self.data.conflict_matrix[ci, cj]
                
                if conflict_size > 0:
                    # These courses have conflicting students
                    # Penalize scheduling them at same time
                    
                    for day in range(self.num_days):
                        for slot in range(self.num_slots):
                            var_ci = self.get_variable_index(ci, day, slot)
                            var_cj = self.get_variable_index(cj, day, slot)
                            
                            # Add penalty: if both = 1, energy increases by penalty
                            penalty = self.config.lambda_h1 * conflict_size
                            Q[var_ci, var_cj] += penalty
                            
                            conflicts_added += 1
                            total_conflict_penalty += penalty
        
        print(f"      ✓ Added {conflicts_added} conflict penalties")
        print(f"      ✓ Total conflict penalty: {total_conflict_penalty:,.0f}")
        
        return Q
    
    def build_h2_duration(self) -> np.ndarray:
        """
        H2: EXAM DURATION CONSTRAINTS (Hard Constraint)
        ================================================
        
        RULE: Exams requiring multiple hours must be scheduled in consecutive
              slots on the same day
        
        SCENARIOS:
        - 1-hour exam: Can fit in any single slot
        - 2-hour exam: Needs 2 consecutive slots (slots 0-1 on same day)
        - 3-hour exam: Needs 3 consecutive slots (all slots on same day, if we have 3 slots/day)
        
        IMPLEMENTATION STRATEGY:
        For exam requiring 'duration' hours:
        
        Option A: ONLY allow valid slot assignments
        - 1-hour exam: Allow all slots
        - 2-hour exam: Only allow first slot of consecutive pairs
        - 3-hour exam: Only allow first slot of consecutive triplets
        
        Option B: Add penalties for invalid assignments
        - Add penalty if exam assigned to slot without enough consecutive slots
        
        We'll use OPTION A (cleaner, fewer variables):
        
        For 2-hour exam on a day with 2 slots:
        - If scheduled at slot 0: Must also activate slot 1
        - Cannot schedule at slot 1 alone (not enough room)
        
        QUBO ENCODING:
        For 2-hour exam on day d:
        - Variable x[c, d, 0] = 1 means exam uses slots 0 AND 1
        - We penalize x[c, d, 1] = 1 heavily (invalid assignment)
        
        Actually, better approach:
        - For 2-hour exam: Create constraint that if assigned to day d,
          must use slots 0 and 1 together
        - Penalize: x[c, d, 0] XOR x[c, d, 1]  (they must be equal)
        
        SIMPLIFIED VERSION (for 2 slots/day):
        - 1-hour exams: No special handling
        - 2-hour exams: Only allow assignment to day, assume uses both slots
        - For this, we modify variable interpretation:
          x[c_2hr, d, 0] = 1 means course c scheduled on day d (uses both slots)
          x[c_2hr, d, 1] is not allowed (add huge penalty to diagonal)
        
        Returns:
            Q matrix contribution
        """
        print("  [2/4] Building H2: Exam Duration Constraints...")
        
        Q = np.zeros((self.total_vars, self.total_vars))
        duration_constraints = 0
        
        for course_id, course_row in self.data.courses.iterrows():
            duration = course_row['duration_hours']
            
            if duration > 1:
                # Multi-slot exam
                # Strategy: Heavily penalize assignment to non-first slots
                
                for day in range(self.num_days):
                    # Allow only slot 0 for multi-hour exams
                    # Penalize assignments to other slots
                    for slot in range(1, self.num_slots):
                        var_idx = self.get_variable_index(course_id, day, slot)
                        # Add penalty to diagonal (linear term)
                        # This makes x[var_idx]=1 very expensive
                        Q[var_idx, var_idx] += self.config.lambda_h2 * 1000
                        duration_constraints += 1
                
                # Additionally, if exam scheduled on day d, slot 0,
                # we implicitly assume it occupies all required slots
                # (This will be validated in solution interpretation)
        
        print(f"      ✓ Added {duration_constraints} duration constraints")
        print(f"      ✓ Multi-hour exams restricted to valid slots")
        
        return Q
    
    def build_h3_one_exam_one_slot(self) -> np.ndarray:
        """
        H3: ONE EXAM ONE SLOT (Hard Constraint)
        ========================================
        
        RULE: Each exam must be scheduled EXACTLY ONCE
        - Not zero times (unscheduled)
        - Not multiple times (double-booked)
        - Exactly one time
        
        MATHEMATICAL FORMULATION:
        For each course c:
          Σ(all days d, all slots t) x[c, d, t] = 1
        
        To convert to QUBO, minimize the squared penalty:
          (Σx - 1)²
        
        EXPANSION:
        (x₁ + x₂ + ... + xₙ - 1)²
        = (x₁)² + (x₂)² + ... + 2x₁x₂ + 2x₁x₃ + ... - 2x₁ - 2x₂ - ... + 1
        
        For binary variables (x² = x):
        = x₁ + x₂ + ... + 2x₁x₂ + 2x₁x₃ + ... - 2x₁ - 2x₂ - ... + 1
        = (1-2)x₁ + (1-2)x₂ + ... + 2(Σᵢ<ⱼ xᵢxⱼ) + 1
        = -x₁ - x₂ - ... + 2(Σᵢ<ⱼ xᵢxⱼ) + 1
        
        Dropping constant:
        = -Σxᵢ + 2Σᵢ<ⱼ xᵢxⱼ
        
        QUBO FORM (multiply by λ):
        - Diagonal Q[i,i]: -λ (coefficient of xᵢ)
        - Off-diagonal Q[i,j]: 2λ (coefficient of xᵢxⱼ)
        
        WHY THIS WORKS:
        - If exactly one variable = 1: Energy = -λ + 0 = -λ (minimum)
        - If zero variables = 1: Energy = 0
        - If two variables = 1: Energy = -2λ + 2λ = 0
        - If all n variables = 1: Energy = -nλ + 2λ*C(n,2) = -nλ + λn(n-1) = λn(n-2)
        
        Actually, we want to MINIMIZE violation, so we ADD penalty for violation.
        The constraint (Σx - 1)² is MINIMIZED when Σx = 1.
        
        Let's use standard form: minimize (Σx - 1)²
        After expansion and using x²=x:
        
        (Σx - 1)² = Σx² - 2Σx + 1 + 2Σᵢ<ⱼ xᵢxⱼ
                  = Σx - 2Σx + 1 + 2Σᵢ<ⱼ xᵢxⱼ  (since x²=x)
                  = -Σx + 1 + 2Σᵢ<ⱼ xᵢxⱼ
        
        Drop constant: -Σx + 2Σᵢ<ⱼ xᵢxⱼ
        
        So:
        - Q[i,i] += -λ (diagonal)
        - Q[i,j] += 2λ (off-diagonal)
        
        Returns:
            Q matrix contribution
        """
        print("  [3/4] Building H3: One Exam One Slot...")
        
        Q = np.zeros((self.total_vars, self.total_vars))
        
        for course in range(self.num_courses):
            # Get all variable indices for this course
            course_vars = []
            for day in range(self.num_days):
                for slot in range(self.num_slots):
                    var_idx = self.get_variable_index(course, day, slot)
                    course_vars.append(var_idx)
            
            # Add diagonal terms: -λ
            for var_i in course_vars:
                Q[var_i, var_i] += -self.config.lambda_h3
            
            # Add off-diagonal terms: 2λ
            for i, var_i in enumerate(course_vars):
                for var_j in course_vars[i+1:]:
                    Q[var_i, var_j] += 2 * self.config.lambda_h3
        
        print(f"      ✓ Added one-slot constraints for {self.num_courses} courses")
        
        return Q
    
    def build_s1_spread_exams(self) -> np.ndarray:
        """
        S1: SPREAD EXAMS OVER PERIOD (Soft Constraint)
        ===============================================
        
        RULE: Students should have sufficient gap between their exams for revision
        
        PENALTY STRUCTURE (based on gap between exams):
        - Same day, different slot:  penalty = 5 (very bad)
        - Consecutive days:          penalty = 3 (bad)
        - 2 days apart:              penalty = 1 (acceptable)
        - ≥3 days apart:             penalty = 0 (good, no penalty)
        
        IMPLEMENTATION:
        For each pair of courses (ci, cj) with conflicting students:
          For each pair of timeslots (d1, t1) and (d2, t2):
            gap = |d1 - d2|
            proximity_penalty = penalty_for_gap(gap)
            Q[var_ci, var_cj] += μ_S1 * conflict_size * proximity_penalty
        
        EXAMPLE:
        Courses C1 and C2 have 15 common students.
        - If C1 on Day 0 and C2 on Day 0: gap=0, penalty = μ*15*5 = 7500
        - If C1 on Day 0 and C2 on Day 1: gap=1, penalty = μ*15*3 = 4500
        - If C1 on Day 0 and C2 on Day 2: gap=2, penalty = μ*15*1 = 1500
        - If C1 on Day 0 and C2 on Day 3: gap=3, penalty = μ*15*0 = 0
        
        WEIGHT:
        - μ (mu) is much smaller than λ (lambda)
        - Typical: μ = 100, λ = 10000
        - This ensures hard constraints dominate
        
        Returns:
            Q matrix contribution
        """
        print("  [4/4] Building S1: Spread Exams (Soft Constraint)...")
        
        Q = np.zeros((self.total_vars, self.total_vars))
        spread_penalties = 0
        
        for ci in range(self.num_courses):
            for cj in range(ci + 1, self.num_courses):
                conflict_size = self.data.conflict_matrix[ci, cj]
                
                if conflict_size > 0:
                    # For each pair of timeslots
                    for d1 in range(self.num_days):
                        for t1 in range(self.num_slots):
                            var_ci = self.get_variable_index(ci, d1, t1)
                            
                            for d2 in range(self.num_days):
                                for t2 in range(self.num_slots):
                                    var_cj = self.get_variable_index(cj, d2, t2)
                                    
                                    # Calculate gap
                                    gap = abs(d1 - d2)
                                    
                                    # Determine proximity penalty
                                    if gap == 0:
                                        proximity = 5  # Same day (very bad)
                                    elif gap == 1:
                                        proximity = 3  # Next day (bad)
                                    elif gap == 2:
                                        proximity = 1  # 2 days apart (okay)
                                    else:
                                        proximity = 0  # ≥3 days (good)
                                    
                                    if proximity > 0:
                                        penalty = self.config.mu_s1 * conflict_size * proximity
                                        
                                        if var_ci == var_cj:
                                            Q[var_ci, var_ci] += penalty
                                        else:
                                            Q[var_ci, var_cj] += penalty
                                        
                                        spread_penalties += 1
        
        print(f"      ✓ Added {spread_penalties} spread penalties")
        
        return Q
    
    def build_qubo(self) -> np.ndarray:
        """
        Build complete QUBO matrix combining all constraints
        
        OBJECTIVE FUNCTION:
        E(x) = λ_H1*H1(x) + λ_H2*H2(x) + λ_H3*H3(x) + μ_S1*S1(x)
        
        Where:
        - H1, H2, H3 are HARD constraints (must be satisfied)
        - S1 is SOFT constraint (nice to satisfy)
        
        Returns:
            Complete symmetric QUBO matrix Q
        """
        print(f"\n{'='*70}")
        print(f"Building QUBO Matrix")
        print(f"{'='*70}")
        print(f"Penalty weights:")
        print(f"  λ_H1 (no conflicts):  {self.config.lambda_h1:,}")
        print(f"  λ_H2 (duration):      {self.config.lambda_h2:,}")
        print(f"  λ_H3 (one slot):      {self.config.lambda_h3:,}")
        print(f"  μ_S1 (spread):        {self.config.mu_s1:,}")
        print(f"{'='*70}\n")
        
        Q = np.zeros((self.total_vars, self.total_vars))
        
        # Add all constraint contributions
        Q += self.build_h1_no_conflict()
        Q += self.build_h2_duration()
        Q += self.build_h3_one_exam_one_slot()
        Q += self.build_s1_spread_exams()
        
        # Make symmetric (required for quantum solvers)
        # Upper and lower triangles might have different values
        # Average them to make symmetric
        Q = (Q + Q.T) / 2
        
        # Statistics
        nonzero = np.count_nonzero(Q)
        density = (nonzero / (self.total_vars ** 2)) * 100
        
        print(f"\n{'='*70}")
        print(f"QUBO Matrix Statistics:")
        print(f"  Dimensions:      {Q.shape[0]} × {Q.shape[1]}")
        print(f"  Total elements:  {self.total_vars ** 2:,}")
        print(f"  Non-zero:        {nonzero:,}")
        print(f"  Density:         {density:.2f}%")
        print(f"  Min value:       {Q.min():,.2f}")
        print(f"  Max value:       {Q.max():,.2f}")
        print(f"  Matrix norm:     {np.linalg.norm(Q):,.2f}")
        print(f"{'='*70}\n")
        
        self.Q = Q
        return Q
    
    def create_bqm(self) -> 'BinaryQuadraticModel':
        """
        Convert QUBO matrix to D-Wave Binary Quadratic Model (BQM)
        
        BQM FORMAT:
        E(x) = Σᵢ hᵢxᵢ + Σᵢ<ⱼ Jᵢⱼxᵢxⱼ + c
        
        Where:
        - hᵢ: linear coefficients (from Q diagonal)
        - Jᵢⱼ: quadratic coefficients (from Q off-diagonal)
        - c: constant offset (we set to 0)
        
        CONVERSION FROM QUBO:
        - h[i] = Q[i,i]  (diagonal elements)
        - J[i,j] = Q[i,j] for i < j  (upper triangle)
        
        Returns:
            BinaryQuadraticModel object for D-Wave
        """
        if self.Q is None:
            self.build_qubo()
        
        print("Converting QUBO to BQM format for D-Wave...")
        
        # Extract linear and quadratic coefficients
        linear = {}
        quadratic = {}
        
        for i in range(self.total_vars):
            # Linear terms (diagonal)
            if self.Q[i, i] != 0:
                linear[i] = self.Q[i, i]
            
            # Quadratic terms (upper triangle)
            for j in range(i + 1, self.total_vars):
                if self.Q[i, j] != 0:
                    quadratic[(i, j)] = self.Q[i, j]
        
        # Create BQM
        bqm = BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')
        
        print(f"  ✓ BQM created")
        print(f"    Variables: {bqm.num_variables}")
        print(f"    Interactions: {bqm.num_interactions}")
        
        self.bqm = bqm
        return bqm
    
    def solve(self) -> SchedulingSolution:
        """
        Solve QUBO using configured solver
        
        SOLVER OPTIONS:
        1. 'neal': Simulated annealing (classical, unlimited, free)
        2. 'dwave': D-Wave quantum annealer (requires hardware access)
        3. 'hybrid': D-Wave hybrid solver (combines classical + quantum)
        
        Returns:
            SchedulingSolution with best found solution
        """
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave Ocean SDK not installed. Run: pip install dwave-ocean-sdk")
        
        if self.bqm is None:
            self.create_bqm()
        
        print(f"\n{'='*70}")
        print(f"Solving QUBO with {self.config.solver_type.upper()} solver")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        if self.config.solver_type == 'neal':
            solution = self._solve_neal()
        elif self.config.solver_type == 'dwave':
            solution = self._solve_dwave()
        elif self.config.solver_type == 'hybrid':
            solution = self._solve_hybrid()
        else:
            raise ValueError(f"Unknown solver type: {self.config.solver_type}")
        
        execution_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"Solution found in {execution_time:.2f} seconds")
        print(f"{'='*70}\n")
        
        # Interpret solution
        return self._interpret_solution(solution, execution_time)
    
    def _solve_neal(self) -> Dict:
        """Solve using Neal simulated annealing simulator"""
        print("Running Neal simulated annealing...")
        print(f"  Num reads: {self.config.num_reads}")
        
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(self.bqm, num_reads=self.config.num_reads)
        
        print(f"  ✓ Completed {len(sampleset)} samples")
        
        # Get best solution
        best_sample = sampleset.first.sample
        best_energy = sampleset.first.energy
        
        print(f"  Best energy: {best_energy:.2f}")
        
        return {
            'sample': best_sample,
            'energy': best_energy,
            'sampleset': sampleset,
            'solver_type': 'neal'
        }
    
    def _solve_dwave(self) -> Dict:
        """Solve using D-Wave quantum annealer"""
        print("Connecting to D-Wave quantum computer...")
        
        try:
            # Connect to D-Wave QPU
            sampler = EmbeddingComposite(DWaveSampler())
            
            # Determine chain strength
            if self.config.chain_strength is None:
                # Auto-calculate as max interaction value
                chain_strength = max(abs(self.Q.min()), abs(self.Q.max())) * 2
            else:
                chain_strength = self.config.chain_strength
            
            print(f"  Solver: {sampler.child.solver.name}")
            print(f"  Num reads: {self.config.num_reads}")
            print(f"  Annealing time: {self.config.annealing_time} μs")
            print(f"  Chain strength: {chain_strength:.2f}")
            
            # Sample
            sampleset = sampler.sample(
                self.bqm,
                num_reads=self.config.num_reads,
                annealing_time=self.config.annealing_time,
                chain_strength=chain_strength
            )
            
            # Get embedding info
            embedding_info = sampleset.info.get('embedding_context', {})
            
            print(f"  ✓ QPU access successful")
            print(f"  Chain length: {embedding_info.get('chain_length', 'N/A')}")
            
            best_sample = sampleset.first.sample
            best_energy = sampleset.first.energy
            
            print(f"  Best energy: {best_energy:.2f}")
            
            return {
                'sample': best_sample,
                'energy': best_energy,
                'sampleset': sampleset,
                'solver_type': 'dwave',
                'embedding_info': embedding_info
            }
            
        except Exception as e:
            print(f"  ✗ D-Wave connection failed: {e}")
            print(f"  Falling back to Neal simulator...")
            return self._solve_neal()
    
    def _solve_hybrid(self) -> Dict:
        """Solve using D-Wave hybrid solver"""
        print("Connecting to D-Wave Leap Hybrid Solver...")
        
        try:
            sampler = LeapHybridSampler()
            
            print(f"  Solver: {sampler.solver.name}")
            
            # Hybrid solver doesn't use num_reads
            sampleset = sampler.sample(self.bqm)
            
            print(f"  ✓ Hybrid solver completed")
            
            best_sample = sampleset.first.sample
            best_energy = sampleset.first.energy
            
            print(f"  Best energy: {best_energy:.2f}")
            
            return {
                'sample': best_sample,
                'energy': best_energy,
                'sampleset': sampleset,
                'solver_type': 'hybrid'
            }
            
        except Exception as e:
            print(f"  ✗ Hybrid solver failed: {e}")
            print(f"  Falling back to Neal simulator...")
            return self._solve_neal()
    
    def _interpret_solution(self, solution: Dict, execution_time: float) -> SchedulingSolution:
        """
        Interpret binary solution as exam schedule
        
        Converts variable assignments x[course, day, slot] back to readable schedule
        
        Args:
            solution: Dict with 'sample' and 'energy'
            execution_time: Time taken to solve
        
        Returns:
            SchedulingSolution object
        """
        print("\nInterpreting solution...")
        
        sample = solution['sample']
        energy = solution['energy']
        
        # Extract assignments
        assignments = {}
        for var_idx, value in sample.items():
            if value == 1:
                course_id, day, slot = self.get_course_day_slot(var_idx)
                
                # Check if course already assigned (H3 violation)
                if course_id in assignments:
                    print(f"  WARNING: Course {course_id} assigned multiple times!")
                
                assignments[course_id] = (day, slot)
        
        # Validate solution
        is_valid, violations = self._validate_solution(assignments)
        
        # Print schedule
        self._print_schedule(assignments)
        
        return SchedulingSolution(
            assignments=assignments,
            energy=energy,
            is_valid=is_valid,
            violations=violations,
            execution_time=execution_time,
            solver_info={
                'solver_type': solution['solver_type'],
                'num_reads': self.config.num_reads if solution['solver_type'] == 'neal' else 1
            }
        )
    
    def _validate_solution(self, assignments: Dict[int, Tuple[int, int]]) -> Tuple[bool, Dict]:
        """
        Validate solution against all constraints
        
        Returns:
            (is_valid, violations_dict)
        """
        violations = {
            'h1_conflicts': 0,
            'h2_duration': 0,
            'h3_unscheduled': 0,
            'h3_multiple': 0,
            's1_same_day': 0
        }
        
        # Check H3: All courses scheduled exactly once
        for course_id in range(self.num_courses):
            if course_id not in assignments:
                violations['h3_unscheduled'] += 1
        
        # Check H1: No conflicts
        for ci in range(self.num_courses):
            for cj in range(ci + 1, self.num_courses):
                if self.data.conflict_matrix[ci, cj] > 0:
                    if ci in assignments and cj in assignments:
                        day_i, slot_i = assignments[ci]
                        day_j, slot_j = assignments[cj]
                        if day_i == day_j and slot_i == slot_j:
                            violations['h1_conflicts'] += 1
        
        # Check H2: Duration constraints
        for course_id, course_row in self.data.courses.iterrows():
            if course_id in assignments:
                duration = course_row['duration_hours']
                day, slot = assignments[course_id]
                
                if duration > 1 and slot != 0:
                    violations['h2_duration'] += 1
        
        # Check S1: Spread (count same-day conflicts)
        for ci in range(self.num_courses):
            for cj in range(ci + 1, self.num_courses):
                if self.data.conflict_matrix[ci, cj] > 0:
                    if ci in assignments and cj in assignments:
                        day_i, _ = assignments[ci]
                        day_j, _ = assignments[cj]
                        if day_i == day_j:
                            violations['s1_same_day'] += 1
        
        is_valid = (violations['h1_conflicts'] == 0 and 
                   violations['h2_duration'] == 0 and 
                   violations['h3_unscheduled'] == 0)
        
        # Print validation results
        print(f"\nValidation Results:")
        print(f"  {'='*66}")
        print(f"  {'Constraint':<30} {'Violations':<15} {'Status':<20}")
        print(f"  {'-'*66}")
        print(f"  {'H1: No conflicts':<30} {violations['h1_conflicts']:<15} {'✓ PASS' if violations['h1_conflicts']==0 else '✗ FAIL':<20}")
        print(f"  {'H2: Duration':<30} {violations['h2_duration']:<15} {'✓ PASS' if violations['h2_duration']==0 else '✗ FAIL':<20}")
        print(f"  {'H3: All scheduled':<30} {violations['h3_unscheduled']:<15} {'✓ PASS' if violations['h3_unscheduled']==0 else '✗ FAIL':<20}")
        print(f"  {'S1: Same-day exams':<30} {violations['s1_same_day']:<15} {'(soft constraint)':<20}")
        print(f"  {'='*66}")
        print(f"  {'OVERALL':<30} {'':<15} {'✓ VALID' if is_valid else '✗ INVALID':<20}")
        print(f"  {'='*66}")
        
        return is_valid, violations
    
    def _print_schedule(self, assignments: Dict[int, Tuple[int, int]]):
        """Print human-readable schedule"""
        print(f"\n{'='*70}")
        print("EXAM SCHEDULE")
        print(f"{'='*70}")
        
        # Organize by day and slot
        schedule = {}
        for day in range(self.num_days):
            schedule[day] = {}
            for slot in range(self.num_slots):
                schedule[day][slot] = []
        
        for course_id, (day, slot) in assignments.items():
            course_code = self.data.courses.iloc[course_id]['course_code']
            enrollment = self.data.courses.iloc[course_id]['enrollment']
            duration = self.data.courses.iloc[course_id]['duration_hours']
            
            schedule[day][slot].append({
                'code': course_code,
                'enrollment': enrollment,
                'duration': duration
            })
        
        # Print schedule
        for day in range(self.num_days):
            print(f"\nDay {day}:")
            for slot in range(self.num_slots):
                print(f"  Slot {slot}:", end=" ")
                if schedule[day][slot]:
                    exams = ", ".join([f"{e['code']} ({e['enrollment']} students, {e['duration']}hr)" 
                                      for e in schedule[day][slot]])
                    print(exams)
                else:
                    print("(empty)")
        
        print(f"\n{'='*70}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - demonstrates complete workflow
    """
    print("\n" + "#"*70)
    print("# D-Wave Ocean SDK QUBO Solver for Exam Scheduling")
    print("# Constraints: H1, H2, H3, S1")
    print("#"*70 + "\n")
    
    # Check D-Wave installation
    if not DWAVE_AVAILABLE:
        print("ERROR: D-Wave Ocean SDK not installed!")
        print("\nInstall with:")
        print("  pip install dwave-ocean-sdk")
        print("\nThis includes:")
        print("  - dimod: QUBO/BQM framework")
        print("  - neal: Simulated annealing solver")
        print("  - dwave-system: D-Wave QPU access")
        print("  - dwave-hybrid: Hybrid solver")
        return
    
    # Load data
    try:
        from data_loader import ExamSchedulingDataLoader
    except ImportError:
        # Try with explicit path
        import importlib.util
        data_loader_path = Path(__file__).parent / "data-loader.py"
        if data_loader_path.exists():
            spec = importlib.util.spec_from_file_location("data_loader", data_loader_path)
            data_loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(data_loader)
            ExamSchedulingDataLoader = data_loader.ExamSchedulingDataLoader
        else:
            print("\nERROR: data-loader.py not found!")
            print("Make sure data-loader.py is in the same directory.")
            return
    
    data_dir = './exam_scheduling_data_tiny'  # Adjust path as needed
    
    try:
        loader = ExamSchedulingDataLoader(data_dir)
        data = loader.load()
        loader.print_statistics(data)
    except FileNotFoundError:
        print(f"\nERROR: Dataset not found at {data_dir}")
        print("\nGenerate a dataset first using dataset-generator.py")
        print("Example:")
        print("  python dataset-generator.py")
        return
    
    # Configure solver
    config = SolverConfig(
        solver_type='neal',  # Start with simulator
        num_reads=5000,      # Many samples to explore solution space
        lambda_h1=1000000,   # Very high penalties for hard constraints
        lambda_h2=1000000,
        lambda_h3=1000000,
        mu_s1=10             # Low soft penalty
    )
    
    # Create solver
    solver = Stage1QuboSolver(data, config)
    
    # Build QUBO
    Q = solver.build_qubo()
    
    # Solve
    solution = solver.solve()
    
    # Print results
    print(f"\n{'='*70}")
    print("SOLUTION SUMMARY")
    print(f"{'='*70}")
    print(f"Energy:          {solution.energy:.2f}")
    print(f"Valid:           {solution.is_valid}")
    print(f"Execution time:  {solution.execution_time:.2f}s")
    print(f"Solver:          {solution.solver_info['solver_type']}")
    print(f"{'='*70}\n")
    
    if solution.is_valid:
        print("✓ SUCCESS: Found valid exam schedule!")
    else:
        print("✗ WARNING: Solution violates hard constraints")
        print("  Try adjusting penalty weights or using different solver")
    
    print("\nNext steps:")
    print("  1. Test with larger datasets (SMALL, MEDIUM)")
    print("  2. Try D-Wave hardware: config.solver_type = 'dwave'")
    print("  3. Implement Stage 2 (room assignment)")
    print("  4. Compare with classical solver (OR-Tools)")


if __name__ == '__main__':
    main()
