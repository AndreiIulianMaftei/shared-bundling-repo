"""
Clustering module for edge bundling analysis.
"""

import copy
import math
import os
import pickle
import re
from typing import List, Dict, Tuple, Optional, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.path import Path
import networkx
import numpy as np
import pandas as pd
import pylab
import requests
import seaborn as sbn
from frechetdist import frdist
from networkx.drawing.nx_pydot import graphviz_layout
from pdf2image import convert_from_path
from PIL import Image as Image
from plotnine import ggplot, aes, geom_violin, geom_boxplot, theme, element_text, labs, element_blank
from scipy import signal
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from modules.EPB.experiments import Experiment


BIG_THRESHOLD = 10
MEDIUM_THRESHOLD = 5
SMALL_THRESHOLD = 2

TIDE_MAX = 100
TIDE_MIN = 1

IMG_REZ = 1601
EDGE_REZ = 100

CONVOLUTION_ITERATIONS = 1

DBSCAN_EPS = 0.65
DBSCAN_MIN_SAMPLES = 2
DBSCAN_DEPTH_WEIGHT = 0.0
DBSCAN_DIST_WEIGHT = 1.0

MST_K_SIGMA = 2.0
MST_NEIGHBORHOOD_DEPTH = 5
MST_DEPTH_WEIGHT = 0.0
MST_DIST_WEIGHT = 1.0

MAX_NORMALIZED_DEPTH = 10

KMEANS_N_CLUSTERS = 'auto'
KMEANS_MIN_CLUSTERS = 2
KMEANS_MAX_CLUSTERS = 10
KMEANS_CLUSTER_BOOST = 500
KMEANS_INTERIOR_FACTOR = 0.3

VERTEX_DEPTH_BOOST = 90
EDGE_DEPTH_PRIMARY = 10
EDGE_DEPTH_SECONDARY = 2

CONVOLUTION_KERNEL_SIZE = 12


class Clustering:
    """Clustering analysis for edge bundling visualization."""

    class Pixel:
        x: int
        y: int
        depth: int

    class Node:
        x: int
        y: int
        depth: int
        id: int

    class Cluster:
        id: int
        depth: int
        x: int
        y: int
        children: List['Clustering.Cluster']
        parent: List['Clustering.Cluster']
        contains: int

    def __init__(self, G: networkx.Graph):
        self.G = G

    def draw_heatMaps(self, matrix: np.ndarray, vertices: List[Tuple]) -> None:
        """Generate heatmap visualizations for each depth level."""
        max_depth = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = int(matrix[i][j])
                max_depth = max(max_depth, matrix[i][j])
        
        for depth in range(int(max_depth), -1, -1):
            check_matrix = np.zeros((IMG_REZ, IMG_REZ))
            
            for i in range(len(matrix)):
                for j in range(len(matrix[i])):
                    if matrix[i][j] >= depth:
                        searched_pos = [(i, j)]
                        search_stack = [(i, j)]
                        
                        while search_stack:
                            current_pos = search_stack.pop()
                            I, J = current_pos
                            
                            for x in range(-1, 2):
                                for y in range(-1, 2):
                                    ni, nj = I + x, J + y
                                    if (0 <= ni < len(matrix) and 
                                        0 <= nj < len(matrix[i]) and 
                                        check_matrix[ni][nj] == 0):
                                        if matrix[ni][nj] >= depth:
                                            searched_pos.append((ni, nj))
                                            search_stack.append((ni, nj))
                                            check_matrix[ni][nj] = 1
                                    check_matrix[I][J] = 1
                        
                        for pos in searched_pos:
                            matrix[pos[0]][pos[1]] -= 1
            
            plt.imshow(check_matrix, cmap='hot', interpolation='nearest')
            plt.savefig(f"heatMap_{depth}.png")
            plt.close() 
    def draw_clusters(self, clusters: List['Clustering.Cluster']) -> None:
        """Visualize clusters as a directed tree using Graphviz layout."""
        G = networkx.DiGraph()

        cluster_map = {cluster: i for i, cluster in enumerate(clusters)}

        for cluster in clusters:
            G.add_node(cluster_map[cluster], depth=cluster.depth)

        for cluster in clusters:
            for child in cluster.children:
                if child in cluster_map:
                    G.add_edge(cluster_map[cluster], cluster_map[child])

        root_idx = cluster_map[clusters[-1]]
        pos = graphviz_layout(G, prog='dot', root=root_idx)

        depths = [c.depth for c in clusters]
        min_depth = min(depths)
        max_depth = max(depths)

        cmap = plt.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=min_depth, vmax=max_depth)
        node_colors = [norm(d) for d in depths]

        fig, ax = plt.subplots(figsize=(8, 6))

        networkx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=400,
            cmap=cmap,
            ax=ax
        )
        networkx.draw_networkx_edges(
            G, pos,
            arrowstyle='-|>',
            arrowsize=12,
            ax=ax
        )
        networkx.draw_networkx_labels(
            G, pos,
            labels={cluster_map[c]: f"d={c.depth}" for c in clusters},
            font_color='white',
            ax=ax
        )

        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Depth')

        ax.set_title("Clusters as a Tree (Root = Last Cluster)")
        ax.axis('off')
        plt.tight_layout()
        
        plt.savefig("clusters_by_depth.png", dpi=300)
        plt.close()

    def cluster_mst(
        self, 
        polylines: List[List[Tuple]], 
        matrix: np.ndarray, 
        vertices: List[Tuple]
    ) -> np.ndarray:
        """Cluster vertices using Minimum Spanning Tree approach."""
        max_depth = np.max(matrix)
        print(f"Computing max depth: {max_depth}")
        
        if max_depth > 0:
            matrix = np.where(matrix > 0, (matrix / max_depth) * MAX_NORMALIZED_DEPTH, 0).astype(int)
        
        print(f"Max depth: {max_depth}, normalized to {MAX_NORMALIZED_DEPTH}")

        num_nodes = len(vertices)
        pairwise_distance = np.zeros((num_nodes, num_nodes))
        pairwise_avg_depth = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    pairwise_distance[i][j] = np.sqrt(
                        (vertices[i][0] - vertices[j][0])**2 + 
                        (vertices[i][1] - vertices[j][1])**2
                    )
                    
                    x1, y1 = int(vertices[i][0]), int(vertices[i][1])
                    x2, y2 = int(vertices[j][0]), int(vertices[j][1])
                    depths = []
                    steps = max(abs(x2 - x1), abs(y2 - y1))
                    
                    if steps > 0:
                        for step in range(steps + 1):
                            t = step / steps
                            x = int(x1 + t * (x2 - x1))
                            y = int(y1 + t * (y2 - y1))
                            if 0 <= x < IMG_REZ and 0 <= y < IMG_REZ:
                                depths.append(matrix[x][y])
                        
                        avg_depth = np.mean(depths) if depths else 0
                        pairwise_avg_depth[i][j] = avg_depth
                        pairwise_avg_depth[j][i] = avg_depth
        
        max_avg_depth = np.max(pairwise_avg_depth)
        normalized_depth = (pairwise_avg_depth / MAX_NORMALIZED_DEPTH if max_avg_depth > 0 
                           else pairwise_avg_depth)
        if np.max(normalized_depth) > 0:
            normalized_depth = normalized_depth / np.max(normalized_depth)
        
        max_distance = np.max(pairwise_distance)
        normalized_distance = (pairwise_distance / max_distance if max_distance > 0 
                              else pairwise_distance)
        
        pairwise_distance_score = (
            MST_DEPTH_WEIGHT * (1 - normalized_depth) + 
            MST_DIST_WEIGHT * normalized_distance
        )

        G = networkx.Graph()
        
        for i in range(num_nodes):
            G.add_node(i, pos=vertices[i])
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = pairwise_distance_score[i][j]
                G.add_edge(i, j, weight=weight)

        mst = networkx.minimum_spanning_tree(G, weight='weight')
        print(f"\nMST computed with {mst.number_of_edges()} edges")
        
        inconsistent_edges = self._find_inconsistent_edges(mst)
        
        mst_clustered = mst.copy()
        for u, v, _, _, _ in inconsistent_edges:
            mst_clustered.remove_edge(u, v)
        
        clusters = list(networkx.connected_components(mst_clustered))
        vertex_clusters = np.full(num_nodes, -1)
        
        for cluster_id, cluster_nodes in enumerate(clusters):
            for node in cluster_nodes:
                vertex_clusters[node] = cluster_id
        
        print(f"\nMST Clustering: {len(clusters)} clusters found, "
              f"{len(inconsistent_edges)} edges removed")
        
        return vertex_clusters

    def _find_inconsistent_edges(
        self, 
        mst: networkx.Graph
    ) -> List[Tuple[int, int, float, float, float]]:
        """Find edges in MST that are inconsistent with their neighborhood."""
        inconsistent_edges = []
        
        for u, v, data in mst.edges(data=True):
            edge_weight = data['weight']
            
            neighbor_weights = []
            visited = {u, v}
            queue = [(u, 0), (v, 0)]
            
            while queue:
                node, dist = queue.pop(0)
                if dist < MST_NEIGHBORHOOD_DEPTH:
                    for neighbor in mst.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
                            
                            edge_data = mst.get_edge_data(node, neighbor)
                            if edge_data and 'weight' in edge_data:
                                neighbor_weights.append(edge_data['weight'])
            
            if len(neighbor_weights) > 1:
                mean_weight = np.mean(neighbor_weights)
                std_weight = np.std(neighbor_weights, ddof=1)
                threshold = mean_weight + MST_K_SIGMA * std_weight
                
                if edge_weight > threshold:
                    inconsistent_edges.append((u, v, edge_weight, mean_weight, std_weight))
        
        return inconsistent_edges

    def get_clusters(
        self, 
        polylines: List[List[Tuple]], 
        matrix: np.ndarray, 
        vertices: List[Tuple]
    ) -> np.ndarray:
        """
        Cluster vertices using HDBSCAN with combined distance metric.
        
        Computes pairwise distances combining spatial distance and density
        along paths between vertices, then applies HDBSCAN clustering.
        
        Args:
            polylines: List of edge polylines
            matrix: Density matrix
            vertices: List of vertex positions
            
        Returns:
            Array of cluster IDs for each vertex (-1 for noise)
        """
        # Normalize matrix to 0-10 range
        max_depth = np.max(matrix)
        print(f"Computing max depth: {max_depth}")
        
        if max_depth > 0:
            matrix = np.where(matrix > 0, (matrix / max_depth) * MAX_NORMALIZED_DEPTH, 0).astype(int)
        
        print(f"Max depth: {max_depth}, normalized to {MAX_NORMALIZED_DEPTH}")

        # Compute pairwise distance and average depth between nodes
        num_nodes = len(vertices)
        pairwise_distance = np.zeros((num_nodes, num_nodes))
        pairwise_avg_depth = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Euclidean distance
                    pairwise_distance[i][j] = np.sqrt(
                        (vertices[i][0] - vertices[j][0])**2 + 
                        (vertices[i][1] - vertices[j][1])**2
                    )
                    
                    # Average depth along line between nodes
                    x1, y1 = int(vertices[i][0]), int(vertices[i][1])
                    x2, y2 = int(vertices[j][0]), int(vertices[j][1])
                    depths = []
                    steps = max(abs(x2 - x1), abs(y2 - y1))
                    
                    if steps > 0:
                        for step in range(steps + 1):
                            t = step / steps
                            x = int(x1 + t * (x2 - x1))
                            y = int(y1 + t * (y2 - y1))
                            if 0 <= x < IMG_REZ and 0 <= y < IMG_REZ:
                                depths.append(matrix[x][y])
                        
                        avg_depth = np.mean(depths) if depths else 0
                        pairwise_avg_depth[i][j] = avg_depth
                        pairwise_avg_depth[j][i] = avg_depth
        
        # Normalize depth and distance to [0, 1]
        max_avg_depth = np.max(pairwise_avg_depth)
        normalized_depth = (pairwise_avg_depth / MAX_NORMALIZED_DEPTH 
                           if max_avg_depth > 0 else pairwise_avg_depth)
        if np.max(normalized_depth) > 0:
            normalized_depth = normalized_depth / np.max(normalized_depth)

        max_distance = np.max(pairwise_distance)
        normalized_distance = (pairwise_distance / max_distance 
                              if max_distance > 0 else pairwise_distance)
        
        # Check symmetry for debugging
        from scipy.linalg import issymmetric
        print(f"Distance matrix symmetric: {issymmetric(normalized_distance)}")
        print(f"Depth matrix symmetric: {issymmetric(normalized_depth)}")

        # Combined distance score (higher depth = more connected = lower distance)
        pairwise_distance_score = (
            DBSCAN_DEPTH_WEIGHT * (1 - normalized_depth) + 
            DBSCAN_DIST_WEIGHT * normalized_distance
        )

        # Apply HDBSCAN clustering with precomputed distance matrix
        from sklearn.cluster import HDBSCAN
        
        clusterer = HDBSCAN(
            min_samples=DBSCAN_MIN_SAMPLES,
            metric='precomputed'
        )
        vertex_clusters = clusterer.fit_predict(pairwise_distance_score)
        
        num_clusters = len(np.unique(vertex_clusters[vertex_clusters != -1]))
        num_noise = np.sum(vertex_clusters == -1)
        
        print(f"HDBSCAN Clustering: {num_clusters} clusters found, {num_noise} noise points")
        
        return vertex_clusters

            

    def all_edges(self, G: networkx.Graph) -> List[List[Tuple[float, float]]]:
        """
        Extract all edge polylines from the graph.
        
        Args:
            G: NetworkX graph with edge coordinate data
            
        Returns:
            List of polylines, where each polyline is a list of (x, y) coordinates
        """
        list_edges = list(self.G.edges(data=True))
        polylines = []
        
        for u, v, data in list_edges:
            numbers_x = [float(num) for num in data.get('X')]
            numbers_y = [float(num) for num in data.get('Y')]
            polyline = [(numbers_x[i], numbers_y[i]) for i in range(len(numbers_x))]
            polylines.append(polyline)
        
        return polylines

    def init_matrix(
        self, 
        polylines: List[List[Tuple[float, float]]], 
        n_clusters: str = 'auto', 
        cluster_depth_boost: int = KMEANS_CLUSTER_BOOST
    ) -> np.ndarray:
        """
        Initialize density matrix from polylines.
        
        Creates a density matrix by rasterizing edge polylines and adding
        depth values based on edge overlap. Optionally performs k-means
        clustering on vertices to boost cluster regions.
        
        Args:
            polylines: List of polylines (each is a list of (x, y) coordinates)
            n_clusters: Number of k-means clusters or 'auto' for automatic detection
            cluster_depth_boost: Depth value to add to cluster centers
            
        Returns:
            Density matrix (IMG_REZ x IMG_REZ)
        """
        matrix = np.zeros((IMG_REZ, IMG_REZ))

        # Step 1: Collect vertices (endpoints of polylines)
        vertices = []
        for polyline in polylines:
            if len(polyline) >= 2:
                vertices.append([polyline[0][0], polyline[0][1]])   # Start
                vertices.append([polyline[-1][0], polyline[-1][1]])  # End
        
        vertices = np.array(vertices)
        print(f"Collected {len(vertices)} vertices for clustering")
        
        # Step 2: K-means clustering (currently disabled by default)
        # Uncomment to enable vertex clustering with depth boost
        # if n_clusters == 'auto':
        #     n_clusters = self._find_optimal_clusters(vertices)
        #     print(f"Automatically determined optimal clusters: {n_clusters}")
        # 
        # if len(vertices) > 0 and len(vertices) >= n_clusters:
        #     self._apply_kmeans_boost(matrix, vertices, n_clusters, cluster_depth_boost)

        # Step 3: Rasterize polylines into density matrix
        for polyline in polylines:
            check_matrix = np.zeros((IMG_REZ, IMG_REZ))
            
            # Boost vertex positions
            start_x, start_y = int(polyline[0][0]), int(polyline[0][1])
            end_x, end_y = int(polyline[-1][0]), int(polyline[-1][1])
            matrix[start_x][start_y] += VERTEX_DEPTH_BOOST
            matrix[end_x][end_y] += VERTEX_DEPTH_BOOST
            
            # Interpolate and rasterize edge segments
            for i in range(len(polyline) - 1):
                x1, y1 = int(polyline[i][0]), int(polyline[i][1])
                x2, y2 = int(polyline[i+1][0]), int(polyline[i+1][1])
                
                # Interpolate EDGE_REZ points between segment endpoints
                x = np.linspace(x1, x2, EDGE_REZ)
                y = np.linspace(y1, y2, EDGE_REZ)
                
                for j in range(EDGE_REZ):
                    xi, yi = int(x[j]), int(y[j])
                    if check_matrix[xi][yi] == 0:
                        matrix[xi][yi] += EDGE_DEPTH_PRIMARY
                    else:
                        matrix[xi][yi] += EDGE_DEPTH_SECONDARY
                    check_matrix[xi][yi] = 1

        return matrix
    
    def _apply_kmeans_boost(
        self,
        matrix: np.ndarray,
        vertices: np.ndarray,
        n_clusters: int,
        cluster_depth_boost: int
    ) -> None:
        """
        Apply k-means clustering to vertices and boost density at cluster regions.
        
        Args:
            matrix: Density matrix to modify (in-place)
            vertices: Array of vertex coordinates
            n_clusters: Number of clusters
            cluster_depth_boost: Depth boost to apply
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(vertices)
        cluster_labels = kmeans.labels_
        
        print(f"K-means clustering completed with {n_clusters} clusters")
        
        for cluster_id in range(n_clusters):
            cluster_vertices = vertices[cluster_labels == cluster_id]
            
            if len(cluster_vertices) < 3:
                # Not enough points for convex hull - boost individual points
                for vertex in cluster_vertices:
                    x = int(np.clip(vertex[0], 0, IMG_REZ - 1))
                    y = int(np.clip(vertex[1], 0, IMG_REZ - 1))
                    matrix[x][y] += cluster_depth_boost
                print(f"Cluster {cluster_id}: Boosted {len(cluster_vertices)} points")
            else:
                # Create convex hull and boost interior
                try:
                    hull = ConvexHull(cluster_vertices)
                    hull_vertices = cluster_vertices[hull.vertices]
                    hull_path = Path(hull_vertices)
                    
                    # Find bounding box
                    min_x = int(max(0, np.floor(hull_vertices[:, 0].min())))
                    max_x = int(min(IMG_REZ - 1, np.ceil(hull_vertices[:, 0].max())))
                    min_y = int(max(0, np.floor(hull_vertices[:, 1].min())))
                    max_y = int(min(IMG_REZ - 1, np.ceil(hull_vertices[:, 1].max())))
                    
                    # Fill interior with reduced boost
                    interior_boost = cluster_depth_boost * KMEANS_INTERIOR_FACTOR
                    for x in range(min_x, max_x + 1):
                        for y in range(min_y, max_y + 1):
                            if hull_path.contains_point((x, y)):
                                matrix[x][y] += interior_boost
                    
                    print(f"Cluster {cluster_id}: Filled hull with {interior_boost:.1f} boost")
                except Exception as e:
                    print(f"Cluster {cluster_id}: Hull failed ({e}), boosting points")
                    for vertex in cluster_vertices:
                        x = int(np.clip(vertex[0], 0, IMG_REZ - 1))
                        y = int(np.clip(vertex[1], 0, IMG_REZ - 1))
                        matrix[x][y] += cluster_depth_boost
    
    def _find_optimal_clusters(
        self, 
        vertices: np.ndarray, 
        min_clusters: int = KMEANS_MIN_CLUSTERS, 
        max_clusters: int = KMEANS_MAX_CLUSTERS
    ) -> int:
        """
        Determine optimal number of clusters using silhouette score.
        
        Args:
            vertices: Array of vertex coordinates
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Optimal number of clusters
        """
        if len(vertices) < min_clusters:
            return max(1, len(vertices))
        
        max_clusters = min(max_clusters, len(vertices) - 1)
        
        if max_clusters < min_clusters:
            return min_clusters
        
        best_score = -1
        best_k = min_clusters
        
        for k in range(min_clusters, max_clusters + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(vertices)
                score = silhouette_score(vertices, labels)
                
                print(f"  Testing k={k}: silhouette score = {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_k = k
            except Exception as e:
                print(f"  Could not evaluate k={k}: {e}")
                continue
        
        print(f"  Best silhouette score: {best_score:.3f} at k={best_k}")
        return best_k
    
    def calcMatrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Apply convolution smoothing to the density matrix.
        
        Args:
            matrix: Raw density matrix
            
        Returns:
            Smoothed density matrix
        """
        # Apply Gaussian-like smoothing via convolution
        kernel = np.ones((CONVOLUTION_KERNEL_SIZE, CONVOLUTION_KERNEL_SIZE))
        
        for _ in range(CONVOLUTION_ITERATIONS):
            matrix = signal.convolve2d(matrix, kernel, mode='same')

        # Save heatmap visualization
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.savefig("heatMap.png")
        plt.close()
        
        return matrix
    
    def init_Points(self) -> List[Tuple[int, int, int]]:
        """
        Extract vertex positions from graph nodes.
        
        Returns:
            List of tuples (x, y, node_id) for each vertex
        """
        vertices = []
        
        for node_id, node_data in self.G.nodes(data=True):
            node_x = node_data.get('X')
            node_y = node_data.get('Y')
            node_id_val = node_data.get('id')
            vertex = (int(node_x), int(node_y), int(node_id))
            vertices.append(vertex)

        return vertices
    
    def get_depth_maps(self, matrix: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Generate binary depth maps for each depth level.
        
        Returns a dictionary where each key is a depth value and the
        corresponding value is a binary matrix indicating pixels at or
        below that depth.
        
        Args:
            matrix: Density matrix
            
        Returns:
            Dictionary mapping depth values to binary matrices
        """
        depth_maps = {}
        
        # Find max depth
        max_depth = int(np.max(matrix))
        
        # Create binary map for each depth level
        for depth in range(max_depth + 1):
            depth_map = np.zeros((IMG_REZ, IMG_REZ))
            
            for i in range(min(len(matrix), IMG_REZ)):
                for j in range(min(len(matrix[i]), IMG_REZ)):
                    if 0 < matrix[i][j] <= depth:
                        depth_map[i][j] = 1
            
            depth_maps[depth] = depth_map
        
        return depth_maps