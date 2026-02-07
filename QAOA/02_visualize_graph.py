"""
Graph Visualization for Exam Conflict Graph
============================================

This script visualizes the conflict graph created from student enrollments.

What it shows:
1. Conflict graph structure (nodes = exams, edges = conflicts)
2. Graph statistics (density, degree distribution)
3. Interactive graph visualization with NetworkX

Author: For quantum graph coloring exam scheduling research
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import json


def get_latest_run_dir(output_base='./output'):
    """Get the latest run directory"""
    latest_file = Path(output_base) / 'latest_run.txt'
    if latest_file.exists():
        with open(latest_file, 'r') as f:
            return Path(f.read().strip())
    return None


def load_data(data_dir):
    """
    Load generated dataset
    
    Args:
        data_dir: Path to directory containing CSV files
        
    Returns:
        dict with all loaded data
    """
    data_path = Path(data_dir)
    
    data = {
        'courses': pd.read_csv(data_path / 'courses.csv'),
        'conflict_adjacency': pd.read_csv(data_path / 'conflict_adjacency.csv', index_col=0),
        'conflict_counts': pd.read_csv(data_path / 'conflict_counts.csv', index_col=0),
    }
    
    # Load metadata if exists
    metadata_path = data_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            data['metadata'] = json.load(f)
    
    return data


def visualize_conflict_graph(adjacency_matrix, courses_df, title="Exam Conflict Graph"):
    """
    Visualize the conflict graph
    
    Args:
        adjacency_matrix: NumPy array or DataFrame, adjacency matrix
        courses_df: DataFrame with course information
        title: Plot title
    """
    # Convert to numpy if DataFrame
    if isinstance(adjacency_matrix, pd.DataFrame):
        adj = adjacency_matrix.values
    else:
        adj = adjacency_matrix
    
    # Create NetworkX graph
    G = nx.Graph()
    
    n = len(adj)
    
    # Add nodes with course names
    for i in range(n):
        course_code = courses_df.iloc[i]['course_code']
        enrollment = courses_df.iloc[i]['enrollment']
        G.add_node(i, 
                  label=course_code,
                  enrollment=enrollment)
    
    # Add edges where conflicts exist
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                G.add_edge(i, j, weight=adj[i, j])
    
    # Calculate layout
    # Use spring layout for nice visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Draw nodes
    # Node size proportional to enrollment
    node_sizes = [G.nodes[i]['enrollment'] * 20 for i in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos,
                          node_color='lightblue',
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    # Draw edges
    # Edge width proportional to number of conflicts
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    max_weight = max(weights) if weights else 1
    
    # Normalize weights for visualization
    edge_widths = [2 + 4 * (w / max_weight) for w in weights]
    
    nx.draw_networkx_edges(G, pos,
                          width=edge_widths,
                          alpha=0.5,
                          edge_color='gray')
    
    # Draw labels
    labels = {i: G.nodes[i]['label'] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos,
                           labels,
                           font_size=10,
                           font_weight='bold')
    
    # Add edge labels (conflict counts) - only for edges with high conflicts
    edge_labels = {(u, v): f"{G[u][v]['weight']}" 
                   for u, v in G.edges() if G[u][v]['weight'] >= 5}
    
    nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels,
                                 font_size=8,
                                 font_color='red')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    return G


def print_graph_statistics(G, adjacency_matrix):
    """
    Print detailed graph statistics
    
    Args:
        G: NetworkX graph
        adjacency_matrix: Adjacency matrix (numpy array or DataFrame)
    """
    if isinstance(adjacency_matrix, pd.DataFrame):
        adj = adjacency_matrix.values
    else:
        adj = adjacency_matrix
    
    n = len(adj)
    num_edges = G.number_of_edges()
    
    print("\n" + "="*60)
    print("CONFLICT GRAPH STATISTICS")
    print("="*60)
    
    print(f"\nNodes (Exams):        {n}")
    print(f"Edges (Conflicts):    {num_edges}")
    
    # Graph density
    max_edges = n * (n - 1) / 2
    density = num_edges / max_edges if max_edges > 0 else 0
    print(f"Graph Density:        {density:.2%}")
    
    # Degree statistics
    degrees = [d for _, d in G.degree()]
    print(f"\nDegree Statistics:")
    print(f"  Min degree:         {min(degrees)}")
    print(f"  Max degree:         {max(degrees)}")
    print(f"  Average degree:     {np.mean(degrees):.2f}")
    
    # Graph coloring bounds
    print(f"\nChromatic Number Bounds:")
    print(f"  Lower bound (max degree + 1):  {max(degrees) + 1}")
    
    # Greedy coloring upper bound
    greedy_coloring = nx.greedy_color(G, strategy='largest_first')
    num_colors_greedy = max(greedy_coloring.values()) + 1
    print(f"  Upper bound (greedy coloring): {num_colors_greedy}")
    
    # Check if graph is complete
    if density == 1.0:
        print("\n  ⚠ Graph is COMPLETE - every exam conflicts with every other!")
        print("  Chromatic number = n (need as many slots as exams)")
    
    # Check connected components
    num_components = nx.number_connected_components(G)
    print(f"\nConnected Components: {num_components}")
    
    if num_components > 1:
        print("  (Some exams can be scheduled independently)")
    
    print("="*60)
    
    return greedy_coloring


def plot_degree_distribution(G):
    """Plot degree distribution histogram"""
    degrees = [d for _, d in G.degree()]
    
    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=range(min(degrees), max(degrees)+2), 
             alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Degree (Number of Conflicts)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Degree Distribution of Conflict Graph', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def visualize_greedy_coloring(G, pos, coloring):
    """
    Visualize greedy coloring solution
    
    Args:
        G: NetworkX graph
        pos: Node positions from layout
        coloring: Dict mapping node -> color
    """
    plt.figure(figsize=(14, 10))
    
    # Color map
    color_palette = ['red', 'blue', 'green', 'yellow', 'purple', 
                    'orange', 'pink', 'cyan', 'brown', 'lime']
    
    # Map color numbers to actual colors
    node_colors = [color_palette[coloring[node] % len(color_palette)] 
                   for node in G.nodes()]
    
    # Node sizes
    enrollments = [G.nodes[i].get('enrollment', 50) for i in G.nodes()]
    node_sizes = [e * 20 for e in enrollments]
    
    # Draw
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=node_sizes,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    nx.draw_networkx_edges(G, pos,
                          alpha=0.3,
                          edge_color='gray')
    
    # Labels showing course + slot
    labels = {i: f"{G.nodes[i]['label']}\nSlot {coloring[i]}" 
              for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=9,
                           font_weight='bold')
    
    plt.title('Greedy Graph Coloring Solution (Classical Baseline)', 
             fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()


def main():
    """Main visualization workflow"""
    
    # Get latest run directory
    run_dir = get_latest_run_dir()
    if run_dir is None:
        print("⚠️ No run directory found. Please run 01_generate_dataset.py first.")
        return
    
    datasets_dir = run_dir / 'datasets'
    viz_dir = run_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    
    print(f"📁 Using run directory: {run_dir}")
    print(f"📁 Visualizations will be saved to: {viz_dir}\n")
    
    # Choose dataset size
    dataset_sizes = ['tiny', 'small', 'medium']
    
    print("="*60)
    print("EXAM CONFLICT GRAPH VISUALIZER")
    print("="*60)
    
    for size in dataset_sizes:
        data_dir = datasets_dir / f'exam_data_{size}'
        
        # Check if directory exists
        if not data_dir.exists():
            print(f"\n⚠ {data_dir} not found. Skipping.")
            continue
        
        print(f"\n{'#'*60}")
        print(f"# Processing {size.upper()} dataset")
        print(f"{'#'*60}")
        
        # Load data
        data = load_data(data_dir)
        
        # Visualize conflict graph
        G = visualize_conflict_graph(
            data['conflict_adjacency'],
            data['courses'],
            title=f"Exam Conflict Graph ({size.upper()})"
        )
        
        plt.savefig(viz_dir / f'conflict_graph_{size}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved {viz_dir}/conflict_graph_{size}.png")
        
        # Print statistics
        greedy_coloring = print_graph_statistics(G, data['conflict_adjacency'])
        
        # Degree distribution
        plot_degree_distribution(G)
        plt.savefig(viz_dir / f'degree_distribution_{size}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved {viz_dir}/degree_distribution_{size}.png")
        
        # Visualize greedy coloring solution
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        visualize_greedy_coloring(G, pos, greedy_coloring)
        plt.savefig(viz_dir / f'greedy_coloring_{size}.png', dpi=150, bbox_inches='tight')
        print(f"✓ Saved {viz_dir}/greedy_coloring_{size}.png")
        
        plt.close('all')  # Close all figures
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("Next step: Run 03_build_qubo.py")
    print("="*60)


if __name__ == '__main__':
    main()
