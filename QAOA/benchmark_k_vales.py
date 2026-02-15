"""
Benchmark Script: Find Maximum Solvable K
==========================================

This script automatically tests different K values to determine:
1. Which K values work on your system
2. Runtime scaling with problem size
3. Solution quality vs. K tradeoff

Usage:
    python benchmark_k_values.py tiny            # Test K=2,3,4,5 on tiny
    python benchmark_k_values.py small --max-k 4 # Test K=2,3,4 on small
    python benchmark_k_values.py --all           # Test all datasets
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np
import sys

# Import QUBO builder
sys.path.append('.')
from importlib import import_module


def build_qubo_for_k(dataset, K):
    """
    Build QUBO matrix for given dataset and K value
    
    Args:
        dataset: Dataset name ('tiny', 'small', 'medium')
        K: Number of colors
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Get latest run directory
    output_base = Path('./output')
    latest_file = output_base / 'latest_run.txt'
    if not latest_file.exists():
        print("⚠️ No run directory found")
        return False
    
    with open(latest_file, 'r') as f:
        run_dir = Path(f.read().strip())
    
    data_dir = run_dir / 'datasets' / f'exam_data_{dataset}'
    
    if not data_dir.exists():
        print(f"⚠️ Dataset {dataset} not found")
        return False
    
    # Check if QUBO already exists
    qubo_file = data_dir / f'qubo_matrix_K{K}.npy'
    if qubo_file.exists():
        return True
    
    print(f"  Building QUBO matrix for K={K}...")
    
    try:
        # Load conflict adjacency matrix
        adjacency_file = data_dir / 'conflict_adjacency.csv'
        adjacency = pd.read_csv(adjacency_file, index_col=0).values
        
        # Import GraphColoringQUBO class
        import_module('03_build_qubo')
        from importlib import reload
        build_qubo_module = reload(sys.modules['03_build_qubo'])
        GraphColoringQUBO = build_qubo_module.GraphColoringQUBO
        
        # Build QUBO
        builder = GraphColoringQUBO(
            adjacency_matrix=adjacency,
            num_colors=K,
            lambda1=10000,
            lambda2=5000
        )
        
        Q = builder.build_full_qubo()
        
        # Save QUBO
        np.save(qubo_file, Q)
        
        # Save metadata
        metadata = {
            'num_exams': builder.n,
            'num_colors': builder.K,
            'num_variables': builder.num_vars,
            'lambda1': builder.lambda1,
            'lambda2': builder.lambda2,
            'qubo_shape': list(Q.shape)
        }
        
        metadata_file = data_dir / f'qubo_metadata_K{K}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Built and saved QUBO for K={K}")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to build QUBO: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark QAOA with different K values')
    parser.add_argument('dataset', nargs='?', default='tiny',
                       choices=['tiny', 'small', 'medium', 'all'],
                       help='Dataset to benchmark')
    parser.add_argument('--min-k', type=int, default=2,
                       help='Minimum K to test (default: 2)')
    parser.add_argument('--max-k', type=int, default=6,
                       help='Maximum K to test (default: 6)')
    parser.add_argument('--reps', type=int, default=2,
                       help='QAOA depth (default: 2)')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per run in seconds (default: 600)')
    return parser.parse_args()


def run_qaoa(dataset, K, reps, timeout):
    """
    Run QAOA for given parameters
    
    Returns:
        dict with results or None if failed
    """
    print(f"\n{'='*60}")
    print(f"Testing: {dataset.upper()} with K={K}")
    print(f"{'='*60}")
    
    # Build QUBO matrix for this K if it doesn't exist
    if not build_qubo_for_k(dataset, K):
        return {
            'dataset': dataset,
            'K': K,
            'status': 'qubo_build_failed'
        }
    
    cmd = [
        'python', '04_solve_qaoa.py',
        dataset, str(K),
        '--reps', str(reps),
        '--timeout', str(timeout),
        '--no-viz'  # Skip viz for benchmarking
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+10)
        runtime = time.time() - start_time
        
        # Try to load results file
        output_base = Path('./output')
        with open(output_base / 'latest_run.txt', 'r') as f:
            run_dir = Path(f.read().strip())
        
        results_file = run_dir / 'solutions' / f'qaoa_results_{dataset}_K{K}_p{reps}.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                detailed_results = json.load(f)
            
            # Check if all required fields exist
            required_fields = ['is_valid', 'runtime_seconds', 'num_variables', 
                              'num_conflicts', 'objective_value', 'colors_used']
            
            if all(field in detailed_results for field in required_fields):
                return {
                    'dataset': dataset,
                    'K': K,
                    'status': 'success',
                    'is_valid': detailed_results['is_valid'],
                    'runtime': detailed_results['runtime_seconds'],
                    'num_variables': detailed_results['num_variables'],
                    'num_conflicts': detailed_results['num_conflicts'],
                    'objective_value': detailed_results['objective_value'],
                    'colors_used': detailed_results['colors_used']
                }
            else:
                print(f"⚠ Results file missing fields: {results_file}")
                return {
                    'dataset': dataset,
                    'K': K,
                    'status': 'incomplete_results',
                    'runtime': runtime
                }
        else:
            print(f"⚠ Results file not found: {results_file}")
            return {
                'dataset': dataset,
                'K': K,
                'status': 'completed_no_results',
                'runtime': runtime
            }
    
    except subprocess.TimeoutExpired:
        print(f"⚠ Timeout after {timeout}s")
        return {
            'dataset': dataset,
            'K': K,
            'status': 'timeout',
            'runtime': timeout
        }
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            'dataset': dataset,
            'K': K,
            'status': 'error',
            'error': str(e)
        }


def main():
    args = parse_args()
    
    print("="*60)
    print("QAOA K-VALUE BENCHMARK")
    print("="*60)
    print(f"Dataset(s): {args.dataset}")
    print(f"K range: {args.min_k} to {args.max_k}")
    print(f"QAOA depth: {args.reps}")
    print(f"Timeout: {args.timeout}s per run")
    
    # Determine datasets to test
    if args.dataset == 'all':
        datasets = ['tiny', 'small', 'medium']
    else:
        datasets = [args.dataset]
    
    all_results = []
    
    for dataset in datasets:
        print(f"\n{'#'*60}")
        print(f"# BENCHMARKING {dataset.upper()}")
        print(f"{'#'*60}")
        
        for K in range(args.min_k, args.max_k + 1):
            result = run_qaoa(dataset, K, args.reps, args.timeout)
            
            if result:
                all_results.append(result)
                
                # Print summary
                if result['status'] == 'success':
                    valid_str = '✓' if result['is_valid'] else '✗'
                    print(f"{valid_str} K={K}: {result['runtime']:.1f}s, "
                          f"{result['num_variables']} vars, "
                          f"conflicts={result['num_conflicts']}")
                else:
                    print(f"✗ K={K}: {result['status']}")
                
                # Stop if timeout or error
                if result['status'] in ['timeout', 'error']:
                    print(f"\n⚠ Stopping benchmark for {dataset} due to {result['status']}")
                    break
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_file = Path('./output') / 'benchmark_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved benchmark results to {output_file}")
    
    # Generate summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    for dataset in datasets:
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        if len(dataset_results) == 0:
            continue
        
        print(f"\n{dataset.upper()}:")
        
        # Find max solvable K
        successful = dataset_results[dataset_results['status'] == 'success']
        
        if len(successful) == 0:
            print(f"  ✗ No successful runs")
            continue
        
        valid_solutions = successful[successful['is_valid'] == True]
        
        if len(valid_solutions) > 0:
            max_valid_K = valid_solutions['K'].max()
            print(f"  ✓ Maximum valid K: {max_valid_K}")
            
            # Runtime scaling
            if len(successful) > 1:
                print(f"  Runtime scaling:")
                for _, row in successful.iterrows():
                    vars_per_sec = row['num_variables'] / row['runtime']
                    print(f"    K={row['K']}: {row['runtime']:.1f}s ({vars_per_sec:.1f} vars/s)")
        else:
            print(f"  ✗ No valid solutions found")
        
        # Timeouts
        timeouts = dataset_results[dataset_results['status'] == 'timeout']
        if len(timeouts) > 0:
            print(f"  ⚠ Timeouts at K={list(timeouts['K'])}")
    
    # Plot results
    successful_results = results_df[results_df['status'] == 'success']
    
    if len(successful_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Runtime vs K
        for dataset in datasets:
            data = successful_results[successful_results['dataset'] == dataset]
            if len(data) > 0:
                axes[0].plot(data['K'], data['runtime'], 'o-', label=dataset.upper())
        
        axes[0].set_xlabel('K (number of colors)')
        axes[0].set_ylabel('Runtime (seconds)')
        axes[0].set_title('QAOA Runtime vs K')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Variables vs Runtime
        axes[1].scatter(successful_results['num_variables'], 
                       successful_results['runtime'],
                       c=successful_results['K'],
                       cmap='viridis',
                       s=100,
                       alpha=0.6)
        axes[1].set_xlabel('Number of Variables')
        axes[1].set_ylabel('Runtime (seconds)')
        axes[1].set_title('Scaling: Variables vs Runtime')
        axes[1].grid(True, alpha=0.3)
        
        cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
        cbar.set_label('K')
        
        plt.tight_layout()
        plot_file = Path('./output') / 'benchmark_plot.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved benchmark plot to {plot_file}")
        plt.close()
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR YOUR SYSTEM")
    print("="*60)
    
    if len(successful_results) > 0:
        max_vars = successful_results['num_variables'].max()
        max_runtime = successful_results['runtime'].max()
        
        print(f"Maximum solvable problem size: {max_vars} variables")
        print(f"Longest successful runtime: {max_runtime:.1f}s ({max_runtime/60:.1f} min)")
        
        print("\nRecommended parameters:")
        if max_vars < 20:
            print("  - Start with K=3 on TINY dataset")
            print("  - Your system can handle small problems only")
        elif max_vars < 40:
            print("  - Use TINY with K=3-5")
            print("  - SMALL with K=3 may work")
        elif max_vars < 60:
            print("  - TINY works well (K up to 6)")
            print("  - SMALL with K=3-4 should work")
        else:
            print("  - Your system is powerful!")
            print("  - Try MEDIUM dataset")


if __name__ == '__main__':
    main()