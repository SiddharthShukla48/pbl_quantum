"""
Quick Test Script - Verify QUBO Solver Setup
Tests the solver with a minimal example before using real datasets
"""

import numpy as np
from typing import Dict

# Test imports
def test_imports():
    """Test that all required packages are installed"""
    print("Testing imports...")
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy - Run: pip install numpy")
        return False
    
    try:
        import pandas
        print("  ✓ pandas")
    except ImportError:
        print("  ✗ pandas - Run: pip install pandas")
        return False
    
    try:
        from dimod import BinaryQuadraticModel
        print("  ✓ dimod")
    except ImportError:
        print("  ✗ dimod - Run: pip install dwave-ocean-sdk")
        return False
    
    try:
        import neal
        print("  ✓ neal")
    except ImportError:
        print("  ✗ neal - Run: pip install dwave-ocean-sdk")
        return False
    
    try:
        from dwave.system import DWaveSampler
        print("  ✓ dwave-system")
    except ImportError:
        print("  ✗ dwave-system - Run: pip install dwave-ocean-sdk")
        return False
    
    print("\n✓ All imports successful!\n")
    return True


# Minimal QUBO test
def test_minimal_qubo():
    """Test QUBO building with tiny problem"""
    print("Testing QUBO construction...")
    
    try:
        from dimod import BinaryQuadraticModel
        import neal
        
        # Tiny problem: 2 courses, 2 days, 1 slot = 4 variables
        # Variables: x[0]=C0D0, x[1]=C0D1, x[2]=C1D0, x[3]=C1D1
        
        # Constraint: Each course scheduled exactly once
        # (x[0] + x[1] - 1)² + (x[2] + x[3] - 1)²
        
        Q = {}
        
        # Course 0: (x0 + x1 - 1)²
        Q[(0, 0)] = -10000  # -λ
        Q[(1, 1)] = -10000  # -λ
        Q[(0, 1)] = 20000   # 2λ
        
        # Course 1: (x2 + x3 - 1)²
        Q[(2, 2)] = -10000
        Q[(3, 3)] = -10000
        Q[(2, 3)] = 20000
        
        # Create BQM
        bqm = BinaryQuadraticModel.from_qubo(Q)
        
        print(f"  Created BQM with {bqm.num_variables} variables")
        print(f"  Interactions: {bqm.num_interactions}")
        
        # Solve with Neal
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=10)
        
        best = sampleset.first
        print(f"  Best solution: {dict(best.sample)}")
        print(f"  Energy: {best.energy}")
        
        # Validate: Should have exactly one variable per course = 1
        course0_vars = best.sample[0] + best.sample[1]
        course1_vars = best.sample[2] + best.sample[3]
        
        if course0_vars == 1 and course1_vars == 1:
            print("  ✓ Valid solution (each course scheduled once)")
            return True
        else:
            print(f"  ✗ Invalid solution (C0: {course0_vars}, C1: {course1_vars})")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


# Test dataset loading
def test_dataset_loading():
    """Test if datasets exist and can be loaded"""
    print("Testing dataset loading...")
    
    try:
        import sys
        from pathlib import Path
        
        # Add current directory to path for imports
        sys.path.insert(0, str(Path(__file__).parent))
        
        from data_loader import ExamSchedulingDataLoader
        
        data_dir = Path('./exam_scheduling_TINY')
        
        if not data_dir.exists():
            print(f"  ⚠ Dataset not found at {data_dir}")
            print(f"    Run: uv run python dataset-generator.py")
            return False
        
        loader = ExamSchedulingDataLoader(str(data_dir))
        data = loader.load()
        
        print(f"  ✓ Loaded {len(data.courses)} courses")
        print(f"  ✓ Loaded {len(data.students)} students")
        print(f"  ✓ Conflict matrix shape: {data.conflict_matrix.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


# Test D-Wave connection (optional)
def test_dwave_connection():
    """Test D-Wave cloud connection"""
    print("Testing D-Wave connection (optional)...")
    
    try:
        from dwave.system import DWaveSampler
        
        # Just test connection, don't run anything
        sampler = DWaveSampler()
        print(f"  ✓ Connected to {sampler.solver.name}")
        print(f"    Qubits: {sampler.solver.properties.get('num_qubits', 'N/A')}")
        print(f"    Topology: {sampler.solver.properties.get('topology', {}).get('type', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"  ⚠ D-Wave not configured: {e}")
        print(f"    This is OK - you can use Neal simulator")
        print(f"    To setup D-Wave: dwave config create")
        return False


# Main test runner
def main():
    print("="*70)
    print("QUBO Solver Test Suite")
    print("="*70 + "\n")
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    
    if not results['imports']:
        print("\n✗ Import test failed!")
        print("Install dependencies: pip install -r requirements.txt")
        return
    
    # Test 2: QUBO construction
    results['qubo'] = test_minimal_qubo()
    print()
    
    # Test 3: Dataset loading
    results['dataset'] = test_dataset_loading()
    print()
    
    # Test 4: D-Wave (optional)
    results['dwave'] = test_dwave_connection()
    print()
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Imports:     {'✓ PASS' if results['imports'] else '✗ FAIL'}")
    print(f"  QUBO:        {'✓ PASS' if results['qubo'] else '✗ FAIL'}")
    print(f"  Dataset:     {'✓ PASS' if results['dataset'] else '⚠ SKIP'}")
    print(f"  D-Wave:      {'✓ PASS' if results['dwave'] else '⚠ SKIP (optional)'}")
    print("="*70)
    
    essential_passed = results['imports'] and results['qubo']
    
    if essential_passed:
        print("\n✓ Essential tests passed!")
        print("\nYou can now run:")
        if not results['dataset']:
            print("  1. uv run python dataset-generator.py  (generate datasets)")
            print("  2. uv run python qubo_solver_dwave.py  (run solver)")
        else:
            print("  uv run python qubo_solver_dwave.py")
    else:
        print("\n✗ Some tests failed - see above for details")


if __name__ == '__main__':
    main()
