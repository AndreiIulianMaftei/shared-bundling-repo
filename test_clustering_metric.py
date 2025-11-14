"""Test clustering metric with sample data and visualizations."""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from modules.clustering import Clustering
import seaborn as sns
from matplotlib.patches import Rectangle
import os

IMG_REZ = 1550
EDGE_REZ = 100

def create_sample_graph(num_nodes=10, graph_type='bundled', seed=42):
    """Create sample graph with spatial coordinates and bundled edges."""
    np.random.seed(seed)
    G = nx.Graph()
    
    num_clusters = max(3, num_nodes // 5)
    cluster_centers = []
    
    for _ in range(num_clusters):
        cx = np.random.uniform(80, 320)
        cy = np.random.uniform(80, 320)
        cluster_centers.append((cx, cy))
    
    for i in range(num_nodes):
        cluster_idx = i % num_clusters
        cx, cy = cluster_centers[cluster_idx]
        
        radius = np.random.uniform(20, 60)
        angle = np.random.uniform(0, 2 * np.pi)
        
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        x = max(10, min(390, x))
        y = max(10, min(390, y))
        
        G.add_node(i, X=int(x), Y=int(y), id=i)
    
    edges_to_add = []
    
    # Random connections
    for i in range(num_nodes):
        num_connections = np.random.randint(1, 3)
        for _ in range(num_connections):
            target = np.random.randint(0, num_nodes)
            if target != i and (i, target) not in edges_to_add and (target, i) not in edges_to_add:
                edges_to_add.append((i, target))
    
    # Cross-cluster connections (creates bundling)
    for i in range(num_nodes):
        cluster_i = i % num_clusters
        for j in range(i + 1, num_nodes):
            cluster_j = j % num_clusters
            if cluster_i != cluster_j and np.random.random() < 0.15:
                if (i, j) not in edges_to_add and (j, i) not in edges_to_add:
                    edges_to_add.append((i, j))
    
    # Hub nodes
    num_hubs = max(1, num_nodes // 15)
    hub_nodes = np.random.choice(num_nodes, num_hubs, replace=False)
    for hub in hub_nodes:
        num_hub_connections = np.random.randint(3, 6)
        targets = np.random.choice(num_nodes, num_hub_connections, replace=False)
        for target in targets:
            if target != hub and (hub, target) not in edges_to_add and (target, hub) not in edges_to_add:
                edges_to_add.append((hub, target))
    
    edges_to_add = list(set(edges_to_add))
    
    # Add edges with simulated bundled paths
    for u, v in edges_to_add:
        x_u, y_u = G.nodes[u]['X'], G.nodes[u]['Y']
        x_v, y_v = G.nodes[v]['X'], G.nodes[v]['Y']
        
        if graph_type == 'bundled':
            num_points = 20
            t = np.linspace(0, 1, num_points)
            
            if np.random.random() < 0.7:
                convergence_x = np.random.uniform(150, 250)
                convergence_y = np.random.uniform(150, 250)
                cx = (x_u + x_v) / 2 + (convergence_x - (x_u + x_v) / 2) * 0.8
                cy = (y_u + y_v) / 2 + (convergence_y - (y_u + y_v) / 2) * 0.8
            else:
                cx = (x_u + x_v) / 2 + np.random.uniform(-40, 40)
                cy = (y_u + y_v) / 2 + np.random.uniform(-40, 40)
            
            x_points = []
            y_points = []
            for ti in t:
                x = (1-ti)**2 * x_u + 2*(1-ti)*ti*cx + ti**2 * x_v
                y = (1-ti)**2 * y_u + 2*(1-ti)*ti*cy + ti**2 * y_v
                x_points.append(x)
                y_points.append(y)
        else:
            x_points = [x_u, x_v]
            y_points = [y_u, y_v]
        
        G.add_edge(u, v, X=x_points, Y=y_points)
    
    G.graph['name'] = f'sample_{graph_type}'
    G.graph['xmax'] = IMG_REZ
    G.graph['ymax'] = IMG_REZ
    
    return G


def visualize_graph(G, save_path='sample_graph.png'):
    """Visualize input graph with nodes and edges."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    for u, v, data in G.edges(data=True):
        ax.plot(data['X'], data['Y'], 'b-', alpha=0.5, linewidth=1)
    
    for node, data in G.nodes(data=True):
        ax.plot(data['X'], data['Y'], 'ro', markersize=1)
        
    
    ax.set_xlim(0, IMG_REZ)
    ax.set_ylim(0, IMG_REZ)
    ax.set_aspect('equal')
    ax.set_title('Input Graph Structure')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Graph visualization saved to: {save_path}")
    plt.close()


def visualize_heatmap(matrix, save_path='clustering_heatmap.png'):
    """Visualize density matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if matrix.shape[0] > 1000:
        non_zero_rows = np.any(matrix > 0, axis=1)
        non_zero_cols = np.any(matrix > 0, axis=0)
        
        row_indices = np.where(non_zero_rows)[0]
        col_indices = np.where(non_zero_cols)[0]
        
        if len(row_indices) > 0 and len(col_indices) > 0:
            row_min, row_max = max(0, row_indices[0] - 50), min(matrix.shape[0], row_indices[-1] + 50)
            col_min, col_max = max(0, col_indices[0] - 50), min(matrix.shape[1], col_indices[-1] + 50)
            matrix_subset = matrix[row_min:row_max, col_min:col_max]
        else:
            matrix_subset = matrix
    else:
        matrix_subset = matrix
    
    im = ax.imshow(matrix_subset, cmap='hot', interpolation='nearest', origin='lower')
    plt.colorbar(im, ax=ax, label='Density Value')
    
    ax.set_title('Edge Density Heatmap (after convolution)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Heatmap saved to: {save_path}")
    plt.close()


def visualize_vertex_clusters(G, vertex_clusters, vertices, save_path='vertex_clusters.png'):
    """Visualize vertex clusters on the graph."""
    if vertex_clusters is None or len(vertex_clusters) == 0:
        print("No cluster data to visualize")
        return
    
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw edges
    for u, v, data in G.edges(data=True):
        ax.plot(data['X'], data['Y'], 'gray', alpha=0.3, linewidth=1)
    
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
                ax.plot(v[0], v[1], 'x', color='black', markersize=1, markeredgewidth=2)
        else:
            cluster_mask = vertex_clusters == cluster_id
            cluster_vertices = [vertices[i] for i in range(len(vertices)) if cluster_mask[i]]
            color = colors[cluster_id % len(colors)]
            for v in cluster_vertices:
                ax.plot(v[0], v[1], 'o', color=color, markersize=1)
                
    
    ax.set_xlim(0, IMG_REZ)
    ax.set_ylim(0, IMG_REZ)
    ax.set_aspect('equal')
    ax.set_title(f'Vertex Clustering: {num_clusters} clusters found', fontsize=14, weight='bold')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Vertex cluster visualization saved to: {save_path}")
    plt.close()


def print_cluster_summary(vertex_clusters, vertices, method_name=""):
    """Print text summary of clustering results."""
    unique_clusters = np.unique(vertex_clusters)
    num_clusters = len(unique_clusters[unique_clusters != -1])
    num_noise = np.sum(vertex_clusters == -1)
    
    print(f"\n{method_name} Results:")
    print(f"  Vertices: {len(vertices)}, Clusters: {num_clusters}, Noise: {num_noise}")
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            continue
        cluster_size = np.sum(vertex_clusters == cluster_id)
        print(f"  Cluster {cluster_id}: {cluster_size} vertices")


def main():
    """Run clustering test pipeline."""
    output_dir = "clustering_test_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}/")
    
    print("\n[1/7] Loading graph from GraphML file...")
    G = nx.read_graphml('inputs/mafia_cubu.graphml')
    print(f"   Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Convert node attributes to proper format if needed
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
    
    G_straight_edge = G.copy()
    for u, v in G_straight_edge.edges():
        edge_data = G_straight_edge[u][v]
        edge_data['X'] = [G_straight_edge.nodes[u]['X'], G_straight_edge.nodes[v]['X']]
        edge_data['Y'] = [G_straight_edge.nodes[u]['Y'], G_straight_edge.nodes[v]['Y']]
    
    
    print("\n[2/7] Visualizing input graph...")
    visualize_graph(G, save_path=f'{output_dir}/1_input_graph.png')
    visualize_graph(G_straight_edge, save_path=f'{output_dir}/1_input_graph_straight_edges.png')
    
    print("\n[3/7] Initializing Clustering class...")
    clustering = Clustering(G)
    clustering_straight = Clustering(G_straight_edge)
    
    print("\n[4/7] Processing graph data...")
    polylines = clustering.all_edges(G)
    vertices = clustering.init_Points()

    polylines_straight = clustering_straight.all_edges(G_straight_edge)
    vertices_straight = clustering_straight.init_Points()

    print(f"   Extracted {len(polylines)} polylines")
    print(f"   Extracted {len(vertices)} vertices")
    
    print("\n[5/7] Computing density matrix...")
    matrix = clustering.init_matrix(polylines)
    matrix_straight = clustering_straight.init_matrix(polylines_straight)
    print(f"   Initial matrix shape: {matrix.shape}")
    print(f"   Initial matrix max value: {matrix.max():.2f}")
    
    matrix = clustering.calcMatrix(matrix)
    matrix_straight = clustering_straight.calcMatrix(matrix_straight)
    print(f"   Convolved matrix max value: {matrix.max():.2f}")
    
    visualize_heatmap(matrix, save_path=f'{output_dir}/2_density_heatmap.png')
    visualize_heatmap(matrix_straight, save_path=f'{output_dir}/2_density_heatmap_straight_edges.png')
    
    # print("\n[6/7] Computing clusters with DBSCAN...")
    # vertex_clusters_dbscan = clustering.get_clusters(polylines, matrix, vertices)
    # vertex_clusters_dbscan_straight = clustering_straight.get_clusters(polylines_straight, matrix_straight, vertices_straight)
    # print_cluster_summary(vertex_clusters_dbscan, vertices, "DBSCAN")
    # visualize_vertex_clusters(G, vertex_clusters_dbscan, vertices, 
    #                           save_path=f'{output_dir}/3_clusters_dbscan.png')
    # visualize_vertex_clusters(G_straight_edge, vertex_clusters_dbscan_straight, vertices_straight, 
    #                           save_path=f'{output_dir}/3_clusters_dbscan_straight_edges.png')
    
    print("\n[7/7] Computing clusters with MST...")
    vertex_clusters_mst = clustering.cluster_mst(polylines, matrix, vertices)
    print_cluster_summary(vertex_clusters_mst, vertices, "MST")
    visualize_vertex_clusters(G, vertex_clusters_mst, vertices, 
                              save_path=f'{output_dir}/4_clusters_mst.png')
    
    print("\n" + "="*70)
    print("CLUSTERING TEST COMPLETED SUCCESSFULLY!")
    print(f"All outputs saved to: {output_dir}/")
    print("="*70)
    
    return G, clustering, vertex_clusters_mst #,vertex_clusters_dbscan 


if __name__ == "__main__":
    #G, clustering, clusters_dbscan, clusters_mst = main()
    G, clustering, clusters_dbscan = main()
    
    print("\nVariables available for further analysis:")
    print("  - G: NetworkX graph")
    print("  - clustering: Clustering object")
    print("  - clusters_dbscan: DBSCAN clustering results")
    print("  - clusters_mst: MST clustering results")
