"""Batch clustering analysis for all datasets."""

import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from modules.clustering import Clustering

# Constants
IMG_REZ = 400
EDGE_REZ = 100
FIG_SIZE = (12, 12)
DPI = 150
CMAP_CLUSTERS = 'tab20'


def load_graphml_file(filepath: Union[str, Path]) -> nx.Graph:
    """
    Load GraphML file and convert attributes to the expected format.

    Args:
        filepath: Path to the GraphML file.

    Returns:
        NetworkX graph with standardized 'X', 'Y' node attributes and edge coordinates.
    """
    print(f"  Loading: {filepath}")
    G = nx.read_graphml(str(filepath))

    # Convert node attributes to proper format
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        # Normalize coordinates to X, Y
        if 'X' not in node_data and 'x' in node_data:
            node_data['X'] = float(node_data['x'])
        if 'Y' not in node_data and 'y' in node_data:
            node_data['Y'] = float(node_data['y'])
            
        # Ensure ID exists
        if 'id' not in node_data:
            try:
                node_data['id'] = int(node_id)
            except ValueError:
                node_data['id'] = node_id

    # Convert edge attributes
    for u, v in G.edges():
        edge_data = G[u][v]
        if 'X' not in edge_data or 'Y' not in edge_data:
            if 'Spline_X' in edge_data and 'Spline_Y' in edge_data:
                # Parse spline control points
                edge_data['X'] = [float(x) for x in edge_data['Spline_X'].split()]
                edge_data['Y'] = [float(y) for y in edge_data['Spline_Y'].split()]
            else:
                # Fallback to straight line
                edge_data['X'] = [G.nodes[u]['X'], G.nodes[v]['X']]
                edge_data['Y'] = [G.nodes[u]['Y'], G.nodes[v]['Y']]

    return G


def create_straight_version(G: nx.Graph) -> nx.Graph:
    """
    Create a straight-line version of the graph (ignoring bundling).

    Args:
        G: Input graph.

    Returns:
        A copy of the graph with straight edges.
    """
    G_straight = G.copy()

    for u, v in G_straight.edges():
        edge_data = G_straight[u][v]
        # Replace edge coordinates with straight line segments
        edge_data['X'] = [G_straight.nodes[u]['X'], G_straight.nodes[v]['X']]
        edge_data['Y'] = [G_straight.nodes[u]['Y'], G_straight.nodes[v]['Y']]

    return G_straight


def visualize_clusters(
    G: nx.Graph,
    vertex_clusters: np.ndarray,
    vertices: List[Any],
    save_path: Path
) -> None:
    """
    Visualize vertex clusters on the graph and save the image.

    Args:
        G: The graph object (for drawing background edges).
        vertex_clusters: Array of cluster IDs for each vertex.
        vertices: List of vertex coordinates.
        save_path: Path to save the visualization.
    """
    if vertex_clusters is None or len(vertex_clusters) == 0:
        print("    No cluster data to visualize")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    # Draw background edges
    for _, _, data in G.edges(data=True):
        ax.plot(data['X'], data['Y'], 'gray', alpha=0.2, linewidth=0.5)

    # Filter valid clusters (exclude noise -1)
    unique_clusters = np.unique(vertex_clusters)
    valid_clusters = unique_clusters[unique_clusters != -1]
    num_clusters = len(valid_clusters)
    
    # Generate colors
    colors = plt.cm.get_cmap(CMAP_CLUSTERS)(np.linspace(0, 1, max(20, num_clusters)))

    # Draw vertices
    vertices_arr = np.array(vertices) # Assuming vertices is list of lists/tuples
    
    # 1. Draw noise points (cluster -1)
    noise_mask = vertex_clusters == -1
    if np.any(noise_mask):
        noise_points = vertices_arr[noise_mask]
        # vertices structure is likely [x, y, node_id] based on original code usage
        ax.plot(
            noise_points[:, 0], 
            noise_points[:, 1], 
            'x', color='black', markersize=8, markeredgewidth=2, label='Noise'
        )

    # 2. Draw clusters
    for i, cluster_id in enumerate(valid_clusters):
        cluster_mask = vertex_clusters == cluster_id
        cluster_points = vertices_arr[cluster_mask]
        color = colors[i % len(colors)]
        
        ax.plot(
            cluster_points[:, 0], 
            cluster_points[:, 1], 
            'o', color=color, markersize=10,
            markeredgecolor='black', markeredgewidth=1.5,
            label=f'Cluster {cluster_id}' if num_clusters < 10 else None
        )

    ax.set_aspect('equal')
    ax.set_title(f'Vertex Clustering: {num_clusters} clusters', fontsize=12, weight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


def save_density_heatmap(matrix: np.ndarray, output_path: Path, title: str) -> None:
    """Save the density matrix as a heatmap image."""
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(label='Density Value')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI)
    plt.close()


def save_clustering_results(
    output_dir: Path,
    algo_name: str,
    vertex_clusters: np.ndarray,
    vertices: List[Any],
    G: nx.Graph,
    time_taken: float
) -> Dict[str, Any]:
    """
    Save all results for a specific clustering algorithm (DBSCAN, MST, etc.).
    
    Returns:
        Dictionary containing summary statistics.
    """
    unique_clusters = np.unique(vertex_clusters)
    valid_clusters = unique_clusters[unique_clusters != -1]
    num_clusters = len(valid_clusters)
    num_noise = np.sum(vertex_clusters == -1)

    print(f"    {algo_name.upper()}: {num_clusters} clusters, {num_noise} noise points ({time_taken:.2f}s)")

    # 1. Save raw clusters
    np.save(output_dir / f'{algo_name}_clusters.npy', vertex_clusters)

    # 2. Visualize
    visualize_clusters(G, vertex_clusters, vertices, output_dir / f'{algo_name}_clusters.png')

    # 3. Prepare detailed vertex data
    vertex_data = []
    for i, vertex in enumerate(vertices):
        # vertex is likely [x, y, node_id]
        node_id = int(vertex[2]) if len(vertex) > 2 else i
        vertex_data.append({
            'vertex_id': i,
            'node_id': node_id,
            'x': float(vertex[0]),
            'y': float(vertex[1]),
            'cluster_id': int(vertex_clusters[i])
        })

    # 4. Save details to JSON
    with open(output_dir / f'{algo_name}_vertex_details.json', 'w') as f:
        json.dump(vertex_data, f, indent=2)

    # 5. Save details to CSV
    with open(output_dir / f'{algo_name}_vertex_details.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['vertex_id', 'node_id', 'x', 'y', 'cluster_id'])
        writer.writeheader()
        writer.writerows(vertex_data)

    return {
        'num_clusters': int(num_clusters),
        'num_noise': int(num_noise),
        'time_seconds': time_taken,
        'cluster_sizes': [int(np.sum(vertex_clusters == i)) for i in valid_clusters]
    }


def process_dataset(
    dataset_name: str,
    graph_path: Path,
    graph_type: str,
    output_base_dir: Path
) -> Dict[str, Any]:
    """
    Process a single dataset with both DBSCAN and MST clustering.
    """
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_name} ({graph_type})")
    print(f"{'='*80}")

    start_time = time.time()
    results = {
        'dataset': dataset_name,
        'graph_type': graph_type,
        'graph_path': str(graph_path),
        'status': 'failed',
        'error': None
    }

    try:
        # 1. Load and Prepare Graph
        G = load_graphml_file(graph_path)
        if graph_type == 'straight':
            G = create_straight_version(G)

        results['num_nodes'] = G.number_of_nodes()
        results['num_edges'] = G.number_of_edges()
        print(f"  Graph: {results['num_nodes']} nodes, {results['num_edges']} edges")

        # 2. Initialize Clustering
        clustering = Clustering(G)
        polylines = clustering.all_edges(G)
        vertices = clustering.init_Points()
        
        print(f"  Extracted {len(polylines)} polylines, {len(vertices)} vertices")
        results['num_polylines'] = len(polylines)
        results['num_vertices'] = len(vertices)

        # 3. Compute Density Matrix
        print("  Computing density matrix...")
        matrix = clustering.init_matrix(polylines)
        matrix = clustering.calcMatrix(matrix)

        # Prepare output directory
        output_dir = output_base_dir / dataset_name / graph_type
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Heatmap
        save_density_heatmap(
            matrix, 
            output_dir / 'density_heatmap.png', 
            f'{dataset_name} - {graph_type} - Density Heatmap'
        )

        # 4. Run DBSCAN
        print("  Running DBSCAN clustering...")
        t0 = time.time()
        vertex_clusters_dbscan = clustering.get_clusters(polylines, matrix, vertices)
        results['dbscan'] = save_clustering_results(
            output_dir, 'dbscan', vertex_clusters_dbscan, vertices, G, time.time() - t0
        )

        # 5. Run MST
        print("  Running MST clustering...")
        t0 = time.time()
        vertex_clusters_mst = clustering.cluster_mst(polylines, matrix, vertices)
        results['mst'] = save_clustering_results(
            output_dir, 'mst', vertex_clusters_mst, vertices, G, time.time() - t0
        )

        results['status'] = 'success'
        results['total_time_seconds'] = time.time() - start_time
        print(f"  ✓ Completed in {results['total_time_seconds']:.2f}s")

    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        results['error'] = str(e)
        results['traceback'] = traceback.format_exc()

    # Save overall results
    output_dir = output_base_dir / dataset_name / graph_type
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    """Main entry point to process all datasets."""
    input_base = Path('inputs/all_outputs')
    output_base = Path('all_outputs')
    output_base.mkdir(exist_ok=True)

    # Discover datasets
    datasets = []
    if input_base.exists():
        for dataset_dir in sorted(input_base.iterdir()):
            if not dataset_dir.is_dir():
                continue
            
            cubu_file = dataset_dir / 'cubu.graphml'
            if cubu_file.exists():
                datasets.append({
                    'name': dataset_dir.name,
                    'cubu_path': cubu_file
                })
    else:
        print(f"Input directory {input_base} does not exist.")
        return

    print(f"\nFound {len(datasets)} datasets to process")
    print(f"Output directory: {output_base}/")

    all_results = []
    successful = 0
    failed = 0

    for i, dataset_info in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Dataset: {dataset_info['name']}")

        # 1. Process Bundled (CUBu)
        res_cubu = process_dataset(
            dataset_info['name'],
            dataset_info['cubu_path'],
            'cubu',
            output_base
        )
        all_results.append(res_cubu)
        if res_cubu['status'] == 'success':
            successful += 1
        else:
            failed += 1

        # 2. Process Straight
        res_straight = process_dataset(
            dataset_info['name'],
            dataset_info['cubu_path'],
            'straight',
            output_base
        )
        all_results.append(res_straight)
        if res_straight['status'] == 'success':
            successful += 1
        else:
            failed += 1

    # Save Summary
    summary = {
        'total_datasets': len(datasets),
        'total_processed': len(all_results),
        'successful': successful,
        'failed': failed,
        'results': all_results
    }

    with open(output_base / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total runs: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Summary file: {output_base}/summary.json")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
