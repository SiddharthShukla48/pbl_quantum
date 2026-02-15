"""
Backend Comparison Tool
=======================

Compare QAOA vs D-Wave solvers on same problem instances.

Usage:
    python 07_compare_backends.py tiny 3
    python 07_compare_backends.py small 4 --backends qaoa neal
"""

import subprocess
import json
import pandas as pd
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Compare solver backends')
    parser.add_argument('dataset', choices=['tiny', 'small', 'medium'])
    parser.add_argument('K', type=int)
    parser.add_argument('--backends', nargs='+', 
                       default=['qaoa', 'neal'],
                       choices=['qaoa', 'neal', 'dwave', 'hybrid'],
                       help='Backends to compare')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs per backend')
    return parser.parse_args()


def run_solver(dataset, K, backend, run_id=0):
    """Run solver with specific backend"""
    print(f"\n{'='*60}")
    print(f"Run {run_id+1}: {backend.upper()} on {dataset} (K={K})")
    print(f"{'='*60}")
    
    cmd = [
        'python', 'QAOA/06_solve_with_backend.py',
        dataset, str(K),
        '--backend', backend
    ]
    
    # Backend-specific parameters
    if backend == 'qaoa':
        cmd.extend(['--reps', '2', '--maxiter', '100'])
    elif backend in ['neal', 'dwave']:
        cmd.extend(['--num-reads', '1000'])
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        runtime = time.time() - start_time
        
        # Load results
        output_base = Path('./output')
        with open(output_base / 'latest_run.txt', 'r') as f:
            run_dir = Path(f.read().strip())
        
        results_file = run_dir / 'solutions' / f'{backend}_results_{dataset}_K{K}.json'
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            return {
                'backend': backend,
                'dataset': dataset,
                'K': K,
                'run_id': run_id,
                'success': True,
                'runtime': data['runtime_seconds'],
                'energy': data['energy'],
                'is_valid': data['is_valid'],
                'num_conflicts': data['num_conflicts'],
                'colors_used': data['colors_used']
            }
        else:
            return {
                'backend': backend,
                'dataset': dataset,
                'K': K,
                'run_id': run_id,
                'success': False,
                'error': 'Results file not found'
            }
    
    except subprocess.TimeoutExpired:
        return {
            'backend': backend,
            'dataset': dataset,
            'K': K,
            'run_id': run_id,
            'success': False,
            'error': 'Timeout'
        }
    
    except Exception as e:
        return {
            'backend': backend,
            'dataset': dataset,
            'K': K,
            'run_id': run_id,
            'success': False,
            'error': str(e)
        }


def main():
    args = parse_args()
    
    print("="*60)
    print("BACKEND COMPARISON")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"K: {args.K}")
    print(f"Backends: {', '.join(args.backends)}")
    print(f"Runs per backend: {args.runs}")
    
    results = []
    
    for backend in args.backends:
        for run_id in range(args.runs):
            result = run_solver(args.dataset, args.K, backend, run_id)
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_file = Path('./output') / f'backend_comparison_{args.dataset}_K{args.K}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Saved comparison to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    successful = df[df['success'] == True]
    
    if len(successful) > 0:
        summary = successful.groupby('backend').agg({
            'runtime': ['mean', 'std'],
            'energy': ['mean', 'std'],
            'is_valid': 'sum',
            'num_conflicts': 'mean'
        }).round(2)
        
        print("\n" + summary.to_string())
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Runtime comparison
        for backend in args.backends:
            backend_data = successful[successful['backend'] == backend]
            if len(backend_data) > 0:
                axes[0].bar(backend, backend_data['runtime'].mean(), 
                           yerr=backend_data['runtime'].std(),
                           alpha=0.7, capsize=5)
        
        axes[0].set_ylabel('Runtime (seconds)')
        axes[0].set_title('Runtime Comparison')
        axes[0].grid(True, alpha=0.3)
        
        # Energy comparison
        for backend in args.backends:
            backend_data = successful[successful['backend'] == backend]
            if len(backend_data) > 0:
                axes[1].bar(backend, backend_data['energy'].mean(),
                           yerr=backend_data['energy'].std(),
                           alpha=0.7, capsize=5)
        
        axes[1].set_ylabel('Energy')
        axes[1].set_title('Energy Comparison (lower = better)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = Path('./output') / f'backend_comparison_{args.dataset}_K{args.K}.png'
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot to {plot_file}")
        plt.close()
        
        # Winner determination
        print("\n" + "="*60)
        print("WINNER")
        print("="*60)
        
        valid_results = successful[successful['is_valid'] == True]
        
        if len(valid_results) > 0:
            fastest_idx = valid_results['runtime'].idxmin()
            fastest = valid_results.loc[fastest_idx]
            
            best_energy_idx = valid_results['energy'].idxmin()
            best_energy = valid_results.loc[best_energy_idx]
            
            print(f"Fastest valid solution: {fastest['backend'].upper()}")
            print(f"  Runtime: {fastest['runtime']:.2f}s")
            print(f"  Energy: {fastest['energy']:.2f}")
            
            print(f"\nBest energy (valid): {best_energy['backend'].upper()}")
            print(f"  Runtime: {best_energy['runtime']:.2f}s")
            print(f"  Energy: {best_energy['energy']:.2f}")
        else:
            print("⚠ No valid solutions found by any backend")
    else:
        print("⚠ No successful runs")


if __name__ == '__main__':
    main()