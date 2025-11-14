"""Batch clustering analysis for all datasets."""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from modules.clustering import Clustering
import os
import json
import time
from pathlib import Path

IMG_REZ = 400
EDGE_REZ = 100


def load_graphml_file(filepath):
    """Load GraphML file and convert to expected format."""
    print(f"  Loading: {filepath}")
    G = nx.read_graphml(filepath)
    
    # Convert node attributes to proper format
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        if 'X' not in node_data or 'Y' not in node_data:
            # Use the coordinates from the GraphML file
            if 'x' in node_data:
                node_data['X'] = float(node_data['x'])
            if 'y' in node_data:
                node_data['Y'] = float(node_data['y'])
        if 'id' not in node_data:
            node_data['id'] = int(node_id)
    
    # Convert edge attributes if they have spline data
    for u, v in G.edges():
        edge_data = G[u][v]
        if 'X' not in edge_data or 'Y' not in edge_data:
            # Check for Spline_X and Spline_Y
            if 'Spline_X' in edge_data and 'Spline_Y' in edge_data:
                edge_data['X'] = [float(x) for x in edge_data['Spline_X'].split()]
                edge_data['Y'] = [float(y) for y in edge_data['Spline_Y'].split()]
            else:
                # Use straight line between nodes
                edge_data['X'] = [G.nodes[u]['X'], G.nodes[v]['X']]
                edge_data['Y'] = [G.nodes[u]['Y'], G.nodes[v]['Y']]
    
    return G


def create_straight_version(G):
    """Create straight-line version of the graph (no bundling)."""
    G_straight = G.copy()
    
    for u, v in G_straight.edges():
        edge_data = G_straight[u][v]
        # Replace edge coordinates with straight line
        edge_data['X'] = [G_straight.nodes[u]['X'], G_straight.nodes[v]['X']]
        edge_data['Y'] = [G_straight.nodes[u]['Y'], G_straight.nodes[v]['Y']]
    
    return G_straight


def visualize_clusters(G, vertex_clusters, vertices, save_path):
    """Visualize vertex clusters on the graph."""
    if vertex_clusters is None or len(vertex_clusters) == 0:
        print("    No cluster data to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw edges
    for u, v, data in G.edges(data=True):
        ax.plot(data['X'], data['Y'], 'gray', alpha=0.2, linewidth=0.5)
    
    # Get unique clusters
    unique_clusters = np.unique(vertex_clusters)
    num_clusters = len(unique_clusters[unique_clusters != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, num_clusters)))
    
    # Draw vertices colored by cluster
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Noise points
            cluster_mask = vertex_clusters == cluster_id
            cluster_vertices = [vertices[i] for i in range(len(vertices)) if cluster_mask[i]]
            for v in cluster_vertices:
                ax.plot(v[0], v[1], 'x', color='black', markersize=8, markeredgewidth=2)
        else:
            cluster_mask = vertex_clusters == cluster_id
            cluster_vertices = [vertices[i] for i in range(len(vertices)) if cluster_mask[i]]
            color = colors[cluster_id % len(colors)]
            for v in cluster_vertices:
                ax.plot(v[0], v[1], 'o', color=color, markersize=10, 
                       markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_aspect('equal')
    ax.set_title(f'Vertex Clustering: {num_clusters} clusters', fontsize=12, weight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_dataset(dataset_name, graph_path, graph_type, output_base_dir):
    """Process a single dataset with both DBSCAN and MST clustering."""
    
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
        # Load graph
        G = load_graphml_file(graph_path)
        
        if graph_type == 'straight':
            G = create_straight_version(G)
        
        results['num_nodes'] = G.number_of_nodes()
        results['num_edges'] = G.number_of_edges()
        
        print(f"  Graph: {results['num_nodes']} nodes, {results['num_edges']} edges")
        
        # Initialize clustering
        clustering = Clustering(G)
        
        # Process graph data
        polylines = clustering.all_edges(G)
        vertices = clustering.init_Points()
        print(f"  Extracted {len(polylines)} polylines, {len(vertices)} vertices")
        
        results['num_polylines'] = len(polylines)
        results['num_vertices'] = len(vertices)
        
        # Compute density matrix
        print("  Computing density matrix...")
        matrix = clustering.init_matrix(polylines)
        matrix = clustering.calcMatrix(matrix)
        
        # Create output directory
        output_dir = output_base_dir / dataset_name / graph_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save density heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(matrix, cmap='hot', interpolation='nearest', origin='lower')
        plt.colorbar(label='Density Value')
        plt.title(f'{dataset_name} - {graph_type} - Density Heatmap')
        plt.tight_layout()
        plt.savefig(output_dir / 'density_heatmap.png', dpi=150)
        plt.close()
        
        # DBSCAN Clustering
        print("  Running DBSCAN clustering...")
        dbscan_start = time.time()
        vertex_clusters_dbscan = clustering.get_clusters(polylines, matrix, vertices)
        dbscan_time = time.time() - dbscan_start
        
        unique_clusters_dbscan = np.unique(vertex_clusters_dbscan)
        num_clusters_dbscan = len(unique_clusters_dbscan[unique_clusters_dbscan != -1])
        num_noise_dbscan = np.sum(vertex_clusters_dbscan == -1)
        
        print(f"    DBSCAN: {num_clusters_dbscan} clusters, {num_noise_dbscan} noise points ({dbscan_time:.2f}s)")
        
        results['dbscan'] = {
            'num_clusters': int(num_clusters_dbscan),
            'num_noise': int(num_noise_dbscan),
            'time_seconds': dbscan_time,
            'cluster_sizes': [int(np.sum(vertex_clusters_dbscan == i)) 
                            for i in unique_clusters_dbscan if i != -1]
        }
        
        # Save DBSCAN results
        np.save(output_dir / 'dbscan_clusters.npy', vertex_clusters_dbscan)
        visualize_clusters(G, vertex_clusters_dbscan, vertices, 
                          output_dir / 'dbscan_clusters.png')
        
        # Save detailed vertex data for DBSCAN
        vertex_data_dbscan = []
        for i, vertex in enumerate(vertices):
            vertex_data_dbscan.append({
                'vertex_id': i,
                'node_id': int(vertex[2]) if len(vertex) > 2 else i,
                'x': float(vertex[0]),
                'y': float(vertex[1]),
                'cluster_id': int(vertex_clusters_dbscan[i])
            })
        
        with open(output_dir / 'dbscan_vertex_details.json', 'w') as f:
            json.dump(vertex_data_dbscan, f, indent=2)
        
        # Also save as CSV for easier analysis
        import csv
        with open(output_dir / 'dbscan_vertex_details.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['vertex_id', 'node_id', 'x', 'y', 'cluster_id'])
            writer.writeheader()
            writer.writerows(vertex_data_dbscan)
        
        # MST Clustering
        print("  Running MST clustering...")
        mst_start = time.time()
        vertex_clusters_mst = clustering.cluster_mst(polylines, matrix, vertices)
        mst_time = time.time() - mst_start
        
        unique_clusters_mst = np.unique(vertex_clusters_mst)
        num_clusters_mst = len(unique_clusters_mst[unique_clusters_mst != -1])
        num_noise_mst = np.sum(vertex_clusters_mst == -1)
        
        print(f"    MST: {num_clusters_mst} clusters, {num_noise_mst} noise points ({mst_time:.2f}s)")
        
        results['mst'] = {
            'num_clusters': int(num_clusters_mst),
            'num_noise': int(num_noise_mst),
            'time_seconds': mst_time,
            'cluster_sizes': [int(np.sum(vertex_clusters_mst == i)) 
                            for i in unique_clusters_mst if i != -1]
        }
        
        # Save MST results
        np.save(output_dir / 'mst_clusters.npy', vertex_clusters_mst)
        visualize_clusters(G, vertex_clusters_mst, vertices, 
                          output_dir / 'mst_clusters.png')
        
        # Save detailed vertex data for MST
        vertex_data_mst = []
        for i, vertex in enumerate(vertices):
            vertex_data_mst.append({
                'vertex_id': i,
                'node_id': int(vertex[2]) if len(vertex) > 2 else i,
                'x': float(vertex[0]),
                'y': float(vertex[1]),
                'cluster_id': int(vertex_clusters_mst[i])
            })
        
        with open(output_dir / 'mst_vertex_details.json', 'w') as f:
            json.dump(vertex_data_mst, f, indent=2)
        
        # Also save as CSV for easier analysis
        import csv
        with open(output_dir / 'mst_vertex_details.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['vertex_id', 'node_id', 'x', 'y', 'cluster_id'])
            writer.writeheader()
            writer.writerows(vertex_data_mst)
        
        results['status'] = 'success'
        results['total_time_seconds'] = time.time() - start_time
        
        print(f"  ✓ Completed in {results['total_time_seconds']:.2f}s")
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        results['error'] = str(e)
        import traceback
        results['traceback'] = traceback.format_exc()
    
    # Save results JSON
    output_dir = output_base_dir / dataset_name / graph_type
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    """Process all datasets in inputs/all_outputs."""
    
    input_base = Path('inputs/all_outputs')
    output_base = Path('all_outputs')
    output_base.mkdir(exist_ok=True)
    
    # Find all datasets
    datasets = []
    for dataset_dir in sorted(input_base.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        cubu_file = dataset_dir / 'cubu.graphml'
        if cubu_file.exists():
            datasets.append({
                'name': dataset_dir.name,
                'cubu_path': cubu_file
            })
    
    print(f"\nFound {len(datasets)} datasets to process")
    print(f"Output directory: {output_base}/")
    
    # Summary tracking
    all_results = []
    successful = 0
    failed = 0
    
    # Process each dataset
    for i, dataset_info in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Dataset: {dataset_info['name']}")
        
        # Process CuBu version
        result_cubu = process_dataset(
            dataset_info['name'],
            dataset_info['cubu_path'],
            'cubu',
            output_base
        )
        all_results.append(result_cubu)
        if result_cubu['status'] == 'success':
            successful += 1
        else:
            failed += 1
        
        # Process straight version
        result_straight = process_dataset(
            dataset_info['name'],
            dataset_info['cubu_path'],  # We'll convert it to straight
            'straight',
            output_base
        )
        all_results.append(result_straight)
        if result_straight['status'] == 'success':
            successful += 1
        else:
            failed += 1
    
    # Save summary
    summary = {
        'total_datasets': len(datasets),
        'total_processed': len(all_results),
        'successful': successful,
        'failed': failed,
        'results': all_results
    }
    
    with open(output_base / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total datasets: {len(datasets)}")
    print(f"Total runs (cubu + straight): {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_base}/")
    print(f"Summary file: {output_base}/summary.json")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    results = main()
