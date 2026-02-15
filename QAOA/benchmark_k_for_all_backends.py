"""
Unified Benchmark Script for All Quantum Backends
==================================================

Benchmarks QAOA (IBM Qiskit) vs D-Wave (Neal/QPU/Hybrid) solvers.

Usage:
    python benchmark_all_backends.py tiny --backends qaoa neal
    python benchmark_all_backends.py small --backends qaoa neal dwave
    python benchmark_all_backends.py --all-datasets --all-backends
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark all quantum backends',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('dataset', nargs='?', default='tiny',
                       choices=['tiny', 'small', 'medium', 'all'],
                       help='Dataset to test (default: tiny)')
    
    parser.add_argument('--backends', nargs='+',
                       default=['qaoa', 'neal'],
                       choices=['qaoa', 'qaoa_old', 'neal', 'dwave', 'hybrid'],
                       help='Backends to compare (default: qaoa neal)')
    
    parser.add_argument('--k-range', nargs=2, type=int,
                       metavar=('MIN', 'MAX'),
                       help='K value range to test (e.g., --k-range 3 5)')
    
    parser.add_argument('--runs', type=int, default=1,
                       help='Runs per configuration (default: 1)')
    
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per run in seconds (default: 600)')
    
    parser.add_argument('--all-datasets', action='store_true',
                       help='Test all datasets (tiny, small, medium)')
    
    parser.add_argument('--all-backends', action='store_true',
                       help='Test all available backends')
    
    return parser.parse_args()


def get_k_range(dataset):
    """Get default K range for dataset"""
    defaults = {
        'tiny': (2, 5),
        'small': (3, 5),
        'medium': (4, 6)
    }
    return defaults.get(dataset, (3, 4))


def run_solver(dataset, K, backend, run_id, timeout):
    """
    Run a solver with specific backend
    
    Returns:
        dict with results
    """
    print(f"\n{'='*60}")
    print(f"Run {run_id+1}: {backend.upper()} | {dataset} | K={K}")
    print(f"{'='*60}")
    
    # Select appropriate script
    if backend == 'qaoa_old':
        # Use original 04_solve_qaoa.py
        cmd = [
            'python', '04_solve_qaoa.py',
            dataset, str(K),
            '--timeout', str(timeout),
            '--no-viz'
        ]
        backend_label = 'qaoa_original'
    
    else:
        # Use unified solve_with_backend.py
        cmd = [
            'python', 'solve_with_backend.py',
            dataset, str(K),
            '--backend', backend if backend != 'qaoa' else 'qaoa'
        ]
        
        # Add D-Wave specific parameters
        if backend in ['neal', 'dwave']:
            cmd.extend(['--num-reads', '1000'])
        
        backend_label = backend
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 10
        )
        
        wall_time = time.time() - start_time
        
        # Parse results
        output_base = Path('./output')
        with open(output_base / 'latest_run.txt', 'r') as f:
            run_dir = Path(f.read().strip())
        
        solutions_dir = run_dir / 'solutions'
        
        # Try to load results file
        if backend == 'qaoa_old':
            results_file = solutions_dir / f'qaoa_results_{dataset}_K{K}_p2.json'
        else:
            results_file = solutions_dir / f'{backend}_results_{dataset}_K{K}.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                solver_data = json.load(f)
            
            return {
                'dataset': dataset,
                'K': K,
                'backend': backend_label,
                'run_id': run_id,
                'status': 'success',
                'is_valid': solver_data.get('is_valid', False),
                'runtime_solver': solver_data.get('runtime_seconds', wall_time),
                'runtime_wall': wall_time,
                'energy': solver_data.get('energy', None),
                'num_conflicts': solver_data.get('num_conflicts', None),
                'num_variables': solver_data.get('num_variables', None),
                'colors_used': solver_data.get('colors_used', None)
            }
        else:
            # Debug: print why no results
            print(f"  ⚠ Results file not found: {results_file}")
            if result.returncode != 0:
                print(f"  ⚠ Solver exited with code {result.returncode}")
                if result.stderr:
                    print(f"  Error: {result.stderr[:200]}")
            
            return {
                'dataset': dataset,
                'K': K,
                'backend': backend_label,
                'run_id': run_id,
                'status': 'no_results',
                'runtime_wall': wall_time
            }
    
    except subprocess.TimeoutExpired:
        return {
            'dataset': dataset,
            'K': K,
            'backend': backend_label,
            'run_id': run_id,
            'status': 'timeout',
            'runtime_wall': timeout
        }
    
    except Exception as e:
        return {
            'dataset': dataset,
            'K': K,
            'backend': backend_label,
            'run_id': run_id,
            'status': 'error',
            'error': str(e)
        }


def main():
    args = parse_args()
    
    # Determine datasets to test
    if args.all_datasets:
        datasets = ['tiny', 'small', 'medium']
    elif args.dataset == 'all':
        datasets = ['tiny', 'small', 'medium']
    else:
        datasets = [args.dataset]
    
    # Determine backends
    if args.all_backends:
        backends = ['qaoa', 'qaoa_old', 'neal', 'dwave', 'hybrid']
    else:
        backends = args.backends
    
    print("="*60)
    print("UNIFIED QUANTUM BACKEND BENCHMARK")
    print("="*60)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Backends: {', '.join(backends)}")
    print(f"Runs per config: {args.runs}")
    print(f"Timeout: {args.timeout}s")
    
    all_results = []
    
    # Run benchmarks
    for dataset in datasets:
        # Determine K range
        if args.k_range:
            k_min, k_max = args.k_range
        else:
            k_min, k_max = get_k_range(dataset)
        
        print(f"\n{'#'*60}")
        print(f"# DATASET: {dataset.upper()}")
        print(f"# K Range: {k_min} to {k_max}")
        print(f"{'#'*60}")
        
        for K in range(k_min, k_max + 1):
            for backend in backends:
                for run_id in range(args.runs):
                    result = run_solver(dataset, K, backend, run_id, args.timeout)
                    all_results.append(result)
                    
                    # Print quick summary
                    if result['status'] == 'success':
                        valid_str = '✓' if result['is_valid'] else '✗'
                        print(f"  {valid_str} {result['runtime_solver']:.1f}s | "
                              f"{result['num_variables']} vars | "
                              f"conflicts: {result['num_conflicts']}")
                    else:
                        print(f"  ✗ {result['status']}")
                    
                    # Stop backend if timeout
                    if result['status'] == 'timeout':
                        print(f"  ⚠ {backend} timed out, skipping remaining runs")
                        break
    
    # Save results
    df = pd.DataFrame(all_results)
    output_file = Path('./output') / 'benchmark_all_backends.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved results to {output_file}")
    
    # Generate analysis
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    successful = df[df['status'] == 'success']
    
    if len(successful) > 0:
        # Summary by backend
        summary = successful.groupby('backend').agg({
            'runtime_solver': ['mean', 'std', 'min', 'max'],
            'energy': 'mean',
            'is_valid': ['sum', 'count'],
            'num_conflicts': 'mean'
        }).round(2)
        
        print("\nBy Backend:")
        print(summary.to_string())
        
        # Best backend for each dataset
        print("\n" + "="*60)
        print("BEST BACKEND PER DATASET")
        print("="*60)
        
        for dataset in datasets:
            dataset_results = successful[successful['dataset'] == dataset]
            
            if len(dataset_results) > 0:
                # Find fastest valid solution
                valid = dataset_results[dataset_results['is_valid'] == True]
                
                if len(valid) > 0:
                    fastest = valid.loc[valid['runtime_solver'].idxmin()]
                    print(f"\n{dataset.upper()}:")
                    print(f"  Winner: {fastest['backend'].upper()}")
                    print(f"  Runtime: {fastest['runtime_solver']:.2f}s")
                    print(f"  K: {fastest['K']}")
                else:
                    print(f"\n{dataset.upper()}: No valid solutions found")
        
        # Visualization
        create_comparison_plots(successful, backends, datasets)
    
    else:
        print("\n⚠ No successful runs")


def create_comparison_plots(df, backends, datasets):
    """Create comparison visualizations"""
    
    # Filter to successful valid solutions
    valid_df = df[df['is_valid'] == True]
    
    if len(valid_df) == 0:
        print("\n⚠ No valid solutions to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Runtime comparison
    ax1 = axes[0, 0]
    backend_runtimes = valid_df.groupby('backend')['runtime_solver'].mean()
    backend_runtimes.plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.set_title('Average Runtime by Backend')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Energy comparison
    ax2 = axes[0, 1]
    backend_energies = valid_df.groupby('backend')['energy'].mean()
    backend_energies.plot(kind='bar', ax=ax2, color='lightcoral')
    ax2.set_ylabel('Energy')
    ax2.set_title('Average Energy by Backend (lower = better)')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Success rate
    ax3 = axes[1, 0]
    success_rate = df.groupby('backend')['is_valid'].mean() * 100
    success_rate.plot(kind='bar', ax=ax3, color='lightgreen')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_title('Solution Validity Rate')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scaling (runtime vs variables)
    ax4 = axes[1, 1]
    for backend in backends:
        backend_data = valid_df[valid_df['backend'] == backend]
        if len(backend_data) > 0:
            ax4.scatter(backend_data['num_variables'], 
                       backend_data['runtime_solver'],
                       label=backend.upper(), alpha=0.6, s=100)
    
    ax4.set_xlabel('Number of Variables')
    ax4.set_ylabel('Runtime (seconds)')
    ax4.set_title('Scaling: Variables vs Runtime')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = Path('./output') / 'benchmark_all_backends_plot.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {plot_file}")
    plt.close()
    
    # Create per-dataset comparison
    if len(datasets) > 1:
        fig2, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(datasets))
        bar_width = 0.8 / len(backends)
        
        for i, backend in enumerate(backends):
            runtimes = []
            for dataset in datasets:
                data = valid_df[(valid_df['backend'] == backend) & (valid_df['dataset'] == dataset)]
                if len(data) > 0:
                    runtimes.append(data['runtime_solver'].mean())
                else:
                    runtimes.append(0)
            
            ax.bar(x + i * bar_width, runtimes, bar_width, 
                   label=backend.upper(), alpha=0.7)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Runtime (seconds)')
        ax.set_title('Runtime Comparison Across Datasets')
        ax.set_xticks(x + bar_width * (len(backends)-1) / 2)
        ax.set_xticklabels([d.upper() for d in datasets])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_file2 = Path('./output') / 'benchmark_datasets_comparison.png'
        plt.savefig(plot_file2, dpi=150, bbox_inches='tight')
        print(f"✓ Saved dataset comparison to {plot_file2}")
        plt.close()


if __name__ == '__main__':
    main()