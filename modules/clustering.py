import copy
import os
import networkx 
import numpy as np
from scipy import signal
from PIL import Image as Image
import pickle
import seaborn as sbn
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from frechetdist import frdist
import re
from matplotlib.backends.backend_pdf import PdfPages
from pdf2image import convert_from_path
import requests
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_violin, geom_boxplot, theme, element_text, labs, element_blank
import math
from modules.EPB.experiments import Experiment
from typing import List
import pylab
from networkx.drawing.nx_pydot import graphviz_layout
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from matplotlib.path import Path
from sklearn.metrics import silhouette_score

BIG_Threshold = 10   #Threshlods for the numbers of connected nodes when to consider a cluster big, medium or small
MEDIUM_Threshold = 5
SMALL_Threshold = 2

TIDE_MAX = 100  # Range for TIDE - Number that represents how much space is allowed between a
TIDE_MIN = 1    # node and an edge to be considered in that edges cluster

IMG_REZ = 1550  # Resolution of the image
EDGE_REZ = 100  # Resolution of the edge

CONVOLUTION = 1 # How many times the matrix is convoluted

class Clustering:

    def __init__(self, G): 
        self.G = G
        

    class Pixel:
        x: int
        y: int
        depth: int

    class node:
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

    def draw_heatMaps(self, matrix, verticies):
        
        max_depth = 0
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                matrix[i][j] = int(matrix[i][j])
                max_depth = max(max_depth, matrix[i][j])
        
        for depth in range(int(max_depth), -1, -1):
            checkMatrix = np.zeros((IMG_REZ,IMG_REZ))
            for i in range(0, len(matrix)):
                for j in range(0, len(matrix[i])):
                    if(matrix[i][j] >= depth):
                        searchedPos = []
                        searchedPos.append((i,j))
                        searchStack = []
                        searchStack.append((i,j))
                        while(len(searchStack) != 0):
                            currentPos = searchStack.pop()
                            I = currentPos[0]
                            J = currentPos[1]
                            for x in range (-1, 2):
                                for y in range (-1, 2):
                                    if(I + x >= 0 and I + x < len(matrix) and J + y >= 0 and J + y < len(matrix[i]) and checkMatrix[I + x][J + y] == 0):
                                        if(matrix[I + x][J + y] >= depth):
                                            searchedPos.append((I + x, J + y))
                                            searchStack.append((I + x, J + y))
                                            checkMatrix[I + x][J + y] = 1
                                    checkMatrix[I][J] = 1
                        for x in range(0, len(searchedPos)):
                            matrix[searchedPos[x][0]][searchedPos[x][1]] -= 1
            plt.imshow(checkMatrix, cmap='hot', interpolation='nearest')
            plt.savefig(f"heatMap_{depth}.png")
            

        return 
    def draw_clusters(self, clusters):
        """
        Draws all clusters as nodes in a directed tree, 
        using the last cluster in the list as the root.
        """
        G = networkx.DiGraph()

        cluster_map = {}
        for i, cluster in enumerate(clusters):
            cluster_map[cluster] = i

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

    def cluster_mst(self, polilines, matrix, vertices):
        """
        Cluster vertices using Minimum Spanning Tree approach.
        Keeps pairwise distance and depth calculations, then builds MST.
        """
        
        max_depth = 0
        print("computing max depth")
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                matrix[i][j] = int(matrix[i][j])
                max_depth = max(max_depth, matrix[i][j])

        # Normalize matrix values to be between 0 and 10
        if max_depth > 0:
            for i in range(0, len(matrix)):
                for j in range(0, len(matrix[i])):
                    if matrix[i][j] > 0:
                        matrix[i][j] = (matrix[i][j] / max_depth) * 10
                        matrix[i][j] = int(matrix[i][j])
        print("Max depth: ", max_depth, "but normalised to 10")

        # Create matrix to store pairwise distance and average depth between nodes
        num_nodes = len(vertices)
        pairwise_distance = np.zeros((num_nodes, num_nodes))
        pairwise_avg_depth = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Calculate Euclidean distance between nodes
                    pairwise_distance[i][j] = np.sqrt((vertices[i][0] - vertices[j][0])**2 + 
                                                      (vertices[i][1] - vertices[j][1])**2)
                    # Calculate average depth along the line between nodes
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
                        pairwise_avg_depth[i][j] = np.mean(depths) if depths else 0
                        pairwise_avg_depth[j][i] = np.mean(depths) if depths else 0
        
        normalized_depth = pairwise_avg_depth / 10.0
        normalized_depth = normalized_depth / np.max(normalized_depth)
        
        # Normalize distance to [0, 1]
        max_distance = np.max(pairwise_distance)
        normalized_distance = pairwise_distance / max_distance if max_distance > 0 else pairwise_distance
        
        c1 = 0.0  # Weight for depth (higher depth = nodes are more connected)
        c2 = 1.0  # Weight for distance (higher distance = nodes are farther apart)
        
        pairwise_distance_score = c1 * (1 - normalized_depth) + (1-c1) * normalized_distance

        # Create graph with all vertices and pairwise edges
        G = networkx.Graph()
        
        # Add all vertices as nodes
        for i in range(num_nodes):
            G.add_node(i, pos=vertices[i])
        
        # Add all pairwise edges with weights equal to their distance scores
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = pairwise_distance_score[i][j]
                G.add_edge(i, j, weight=weight)

        # Compute Minimum Spanning Tree
        mst = networkx.minimum_spanning_tree(G, weight='weight')
        
        print(f"\nMST computed with {mst.number_of_edges()} edges")
        
        # Detect inconsistent edges and remove them to find clusters
        k_sigma = 2.0  # Tunable parameter: edge is inconsistent if weight > mean + k*std
        neighborhood_depth = 5  # How many tree steps to consider for neighborhood
        
        inconsistent_edges = []
        
        for u, v, data in mst.edges(data=True):
            edge_weight = data['weight']
            
            # Collect neighboring edges (within neighborhood_depth steps in the tree)
            neighbor_weights = []
            
            # BFS to find neighborhood of u and v
            visited = set()
            queue = [(u, 0), (v, 0)]  # (node, distance_from_source)
            visited.add(u)
            visited.add(v)
            
            while queue:
                node, dist = queue.pop(0)
                if dist < neighborhood_depth:
                    for neighbor in mst.neighbors(node):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, dist + 1))
                            # Add the edge weight
                            edge_data = mst.get_edge_data(node, neighbor)
                            if edge_data and 'weight' in edge_data:
                                neighbor_weights.append(edge_data['weight'])
            
            # Calculate mean and standard deviation of neighboring edges
            if len(neighbor_weights) > 1:
                mean_weight = np.mean(neighbor_weights)
                std_weight = np.std(neighbor_weights, ddof=1)  # Sample standard deviation
                
                # Check if edge is inconsistent
                threshold = mean_weight + k_sigma * std_weight
                if edge_weight > threshold:
                    inconsistent_edges.append((u, v, edge_weight, mean_weight, std_weight))
        
        # Remove inconsistent edges to get clusters
        mst_clustered = mst.copy()
        for u, v, _, _, _ in inconsistent_edges:
            mst_clustered.remove_edge(u, v)
        
        clusters = list(networkx.connected_components(mst_clustered))
        
        vertex_clusters = np.full(num_nodes, -1)
        
        for cluster_id, cluster_nodes in enumerate(clusters):
            for node in cluster_nodes:
                vertex_clusters[node] = cluster_id
        
        print(f"\nMST Clustering: {len(clusters)} clusters found, {len(inconsistent_edges)} edges removed")
        
        return vertex_clusters

    def get_clusters(self, polilines, matrix, vertices):
        

        overall_clusters = []
        current_clusters = []
        matrix_with_clusters = np.zeros((IMG_REZ,IMG_REZ), dtype=object)
        max_depth = 0
        
        print("computing max depth")
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                matrix[i][j] = int(matrix[i][j])
                max_depth = max(max_depth, matrix[i][j])

        # Normalize matrix values to be between 0 and 10
        if max_depth > 0:
            for i in range(0, len(matrix)):
                for j in range(0, len(matrix[i])):
                    if matrix[i][j] > 0:
                        matrix[i][j] = (matrix[i][j] / max_depth) * 10
                        matrix[i][j] = int(matrix[i][j])
        print("Max depth: ", max_depth, "but normalised    to 10")

        # Create matrix to store pairwise distance and average depth between nodes
        num_nodes = len(vertices)
        pairwise_distance = np.zeros((num_nodes, num_nodes))
        pairwise_avg_depth = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    # Calculate Euclidean distance between nodes
                    pairwise_distance[i][j] = np.sqrt((vertices[i][0] - vertices[j][0])**2 + 
                                                      (vertices[i][1] - vertices[j][1])**2)
                    # Calculate average depth along the line between nodes
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
                        pairwise_avg_depth[i][j] = np.mean(depths)if depths else 0
                        pairwise_avg_depth[j][i] = np.mean(depths)if depths else 0
                        
        
        normalized_depth = pairwise_avg_depth / 10.0
        normalized_depth = normalized_depth / np.max(normalized_depth)

        # Normalize distance to [0, 1]
        max_distance = np.max(pairwise_distance)
        normalized_distance = pairwise_distance / max_distance if max_distance > 0 else pairwise_distance
        
        c1 = 0  # Weight for depth (higher depth = nodes are more connected)
        c2 = 0.1  # Weight for distance (higher distance = nodes are farther apart)
        
        from scipy.linalg import issymmetric
        print(issymmetric(normalized_distance))
        print("symmetric depth: ", issymmetric(normalized_depth))

        pairwise_distance_score = c1 * (1 - normalized_depth) + (1- c1) * normalized_distance

        # Cluster vertices using DBSCAN with the combined distance score
        from sklearn.cluster import DBSCAN
        
        # DBSCAN parameters (tunable)
        eps = 0.65  # Maximum distance for two nodes to be in same neighborhood
        min_samples = 4  # Minimum nodes to form a dense region
        from sklearn.cluster import HDBSCAN
        # Run DBSCAN with precomputed distance matrix
        dbscan = HDBSCAN(min_samples=min_samples, metric='precomputed')
        vertex_clusters = dbscan.fit_predict(pairwise_distance_score)
        
        num_clusters = len(np.unique(vertex_clusters[vertex_clusters != -1]))
        num_noise = np.sum(vertex_clusters == -1)
        
        print(f"DBSCAN Clustering: {num_clusters} clusters found, {num_noise} noise points")
        
        return vertex_clusters

        
        

        
        ########################## THIS IS OLD VARIANT FOR CLUSTERING ###########################
        # for i in range(len(matrix)):
        #     for j in range(len(matrix[i])):
        #         if matrix[i][j] > 0:
        #             matrix[i][j] = 10 - matrix[i][j]                             
        #         else:
        #             matrix[i][j] = 10


        # graphs_by_depth = []
        # whatCluster = np.zeros((IMG_REZ,IMG_REZ))
        
        # # Process each depth level (from low to high density)
        # for depth_threshold in range(0, 11):  # 0 to 10 (normalized depth values)
             
        #     matrix_graph = networkx.Graph()

        #     # Only add nodes that meet the depth threshold (i.e., density >= threshold)
        #     active_nodes = set()
        #     for x in range(0, len(matrix)):
        #         for y in range(0, len(matrix[x])):
        #             if matrix[x][y] <= depth_threshold:  # Lower values = higher density
        #                 matrix_graph.add_node((x,y))
        #                 active_nodes.add((x,y))
            
        #     # Connect adjacent pixels that both meet the threshold
        #     for x in range(0, len(matrix)):
        #         for y in range(0, len(matrix[x])):
        #             if (x, y) not in active_nodes:
        #                 continue
                    
        #             for dx in [-1, 0, 1]:
        #                 for dy in [-1, 0, 1]:
        #                     if dx == 0 and dy == 0:
        #                         continue
        #                     nx = x + dx
        #                     ny = y + dy
        #                     if 0 <= nx < len(matrix) and 0 <= ny < len(matrix[x]):
        #                         if (nx, ny) in active_nodes:
        #                             matrix_graph.add_edge((x,y), (nx,ny))
            
        #     # Detect connected components in the graph
        #     if len(active_nodes) > 0:
        #         connected_components = list(networkx.connected_components(matrix_graph))
        #     else:
        #         connected_components = []
            
        #     # Filter connected components to keep only those that contain vertices
        #     vertex_positions = set((v[0], v[1]) for v in vertices)
        #     filtered_components = []
        #     for component in connected_components:
        #         if any(pos in vertex_positions for pos in component):
        #             filtered_components.append(component)
        #     connected_components = filtered_components
            
        #     graphs_by_depth.append({
        #         'depth': depth_threshold,
        #         'graph': matrix_graph,
        #         'components': connected_components,
        #         'num_components': len(connected_components)
        #     })
        #     print(f"Depth {depth_threshold}: Found {len(connected_components)} clusters (active pixels: {len(active_nodes)})")
                                    
        # return graphs_by_depth
            #####################THIS IS VERY OLD CLUSTERING VARIANT ##########################
        # for i in range(0, len(vertices)):
        #     cluster = self.Cluster()
        #     cluster.id = i
        #     cluster.x = vertices[i][0]
        #     cluster.y = vertices[i][1]
        #     cluster.depth = matrix[vertices[i][0]][vertices[i][1]]
        #     cluster.children = []
        #     matrix_with_clusters[vertices[i][0]][vertices[i][1]] = cluster
        #     cluster.parent = []
        #     cluster.contains = 1
        #     current_clusters.append(cluster)
        #     overall_clusters.append(cluster)
        # print("Current clusters: ", len(current_clusters))
        # for depth in range(int(max_depth), -1, -1):
        #     print("Processing depth: ", depth)
        #     checkMatrix = np.zeros((IMG_REZ,IMG_REZ))
        #     for i in range(0, min(matrix.__len__() , IMG_REZ)):
        #         for j in range(0, min(matrix[i].__len__(), IMG_REZ)):
        #             if(matrix[i][j] == depth):
        #                 searchedPos = []
        #                 searchedPos.append((i,j))
        #                 searchStack = []
        #                 searchStack.append((i,j))
        #                 brother_clusters = []
        #                 if(matrix_with_clusters[i][j] != 0):
        #                     brother_clusters.append(matrix_with_clusters[i][j])
                        
        #                 while(len(searchStack) != 0):
        #                     currentPos = searchStack.pop()
        #                     I = currentPos[0]
        #                     J = currentPos[1]
                            
        #                     for x in range (-1, 2):
        #                         for y in range (-1, 2):
        #                             if(I + x >= 0 and I + x < min(matrix.__len__() , IMG_REZ) and J + y >= 0 and J + y < min(matrix[i].__len__(), IMG_REZ) and checkMatrix[I + x][J + y] == 0):
        #                                 if(matrix[I + x][J + y] >= depth):
        #                                     searchedPos.append((I + x, J + y))
        #                                     searchStack.append((I + x, J + y))
        #                                     checkMatrix[I + x][J + y] = 1
        #                                     if(matrix_with_clusters[I + x][J + y] != 0):
        #                                         brother_clusters.append(matrix_with_clusters[I + x][J + y])
        #                                         checkMatrix[I + x][J + y] = 1
        #                             checkMatrix[I][J] = 1
            
        #                 for x in range(0, len(searchedPos)):
        #                     matrix[searchedPos[x][0]][searchedPos[x][1]] -= 1
        #                 cluster = self.Cluster()
        #                 if(len(brother_clusters) != 0):
                            
        #                     cluster.id = len(overall_clusters)
        #                     cluster.x = brother_clusters[0].x
        #                     cluster.y = brother_clusters[0].y
        #                     cluster.depth = matrix[brother_clusters[0].x][brother_clusters[0].y]+1
        #                     cluster.children = []
        #                     cluster.parent = []
        #                     cluster.contains = 0
        #                     for x in range(0, len(brother_clusters)):
        #                         cluster.children.append(brother_clusters[x])
        #                         cluster.contains += brother_clusters[x].contains
        #                         for y in range(0, len(brother_clusters[x].children)):
        #                             brother_clusters[x].children[y].parent = cluster
        #                     overall_clusters.append(cluster)

        #                     for x in range(0, len(brother_clusters)):
        #                         matrix_with_clusters[brother_clusters[x].x][brother_clusters[x].y] = 0
        #                         if (x == 0):
        #                             matrix_with_clusters[brother_clusters[x].x][brother_clusters[x].y] = cluster
                                
        #                         current_clusters.remove(brother_clusters[x])
        #                     current_clusters.append(cluster)

        # clusters_by_level = {}
        # for depth in range(int(max_depth) + 1):
        #     clusters_by_level[depth] = {}
        #     for node in vertices:
        #         node_id = node[2]
        #         for cluster in overall_clusters:
        #             if cluster.depth == depth:
        #                 #check if node is anywhere in cluster, check all the children 
        #                 if cluster.contains > 1:
        #                     for child in cluster.children:
        #                         if child.x == node[0] and child.y == node[1]:
        #                             if node_id not in clusters_by_level[depth]:
        #                                 clusters_by_level[depth][node_id] = []
        #                             clusters_by_level[depth][node_id].append(cluster.id)

                
        #         if node_id not in clusters_by_level[depth]:     
        #             clusters_by_level[depth][node_id] = []
        
        # return clusters_by_level  

                    

    # def get_clusters(self, polilines, matrix, vertices):
        

    #     overall_clusters = []
    #     current_clusters = []
    #     matrix_with_clusters = np.zeros((IMG_REZ,IMG_REZ), dtype=object)
    #     max_depth = 0
    #     # matrix = [
    #     #     [1,1,1,1,1,1,1,0,0,0,0,0],
    #     #     [1,2,2,2,2,2,1,0,0,0,0,0],
    #     #     [1,2,3,3,3,2,1,0,0,0,0,0],
    #     #     [1,2,4,3,3,2,1,0,0,0,0,0],
    #     #     [1,2,4,4,3,2,1,0,0,0,0,0],
    #     #     [1,2,2,2,2,2,1,0,2,2,2,0],
    #     #     [1,1,1,1,1,1,1,0,2,3,2,0],
    #     #     [0,0,0,0,0,0,0,0,2,2,2,0],
    #     #     [0,0,0,0,0,0,0,0,0,0,0,0],
    #     #     [0,0,0,0,0,0,0,0,0,0,0,0]
    #     # ]
    #     print("computing max depth")
    #     for i in range(0, len(matrix)):
    #         for j in range(0, len(matrix[i])):
    #             matrix[i][j] = int(matrix[i][j])
    #             max_depth = max(max_depth, matrix[i][j])

    #     # Normalize matrix values to be between 0 and 10
    #     if max_depth > 0:
    #         for i in range(0, len(matrix)):
    #             for j in range(0, len(matrix[i])):
    #                 if matrix[i][j] > 0:
    #                     matrix[i][j] = (matrix[i][j] / max_depth) * 10
    #                     matrix[i][j] = int(matrix[i][j])
    #     print("Max depth: ", max_depth, "but normalised to 10")
    #     for i in range(0, len(vertices)):
    #         cluster = self.Cluster()
    #         cluster.id = i
    #         cluster.x = vertices[i][0]
    #         cluster.y = vertices[i][1]
    #         cluster.depth = matrix[vertices[i][0]][vertices[i][1]]
    #         cluster.children = []
    #         matrix_with_clusters[vertices[i][0]][vertices[i][1]] = cluster
    #         cluster.parent = []
    #         cluster.contains = 1
    #         current_clusters.append(cluster)
    #         overall_clusters.append(cluster)
    #     print("Current clusters: ", len(current_clusters))
    #     for depth in range(int(max_depth), -1, -1):
    #         print("Processing depth: ", depth)
    #         checkMatrix = np.zeros((IMG_REZ,IMG_REZ))
    #         for i in range(0, min(matrix.__len__() , IMG_REZ)):
    #             for j in range(0, min(matrix[i].__len__(), IMG_REZ)):
    #                 if(matrix[i][j] == depth):
    #                     searchedPos = []
    #                     searchedPos.append((i,j))
    #                     searchStack = []
    #                     searchStack.append((i,j))
    #                     brother_clusters = []
    #                     if(matrix_with_clusters[i][j] != 0):
    #                         brother_clusters.append(matrix_with_clusters[i][j])
                        
    #                     while(len(searchStack) != 0):
    #                         currentPos = searchStack.pop()
    #                         I = currentPos[0]
    #                         J = currentPos[1]
                            
    #                         for x in range (-1, 2):
    #                             for y in range (-1, 2):
    #                                 if(I + x >= 0 and I + x < min(matrix.__len__() , IMG_REZ) and J + y >= 0 and J + y < min(matrix[i].__len__(), IMG_REZ) and checkMatrix[I + x][J + y] == 0):
    #                                     if(matrix[I + x][J + y] >= depth):
    #                                         searchedPos.append((I + x, J + y))
    #                                         searchStack.append((I + x, J + y))
    #                                         checkMatrix[I + x][J + y] = 1
    #                                         if(matrix_with_clusters[I + x][J + y] != 0):
    #                                             brother_clusters.append(matrix_with_clusters[I + x][J + y])
    #                                             checkMatrix[I + x][J + y] = 1
    #                                 checkMatrix[I][J] = 1
            
    #                     for x in range(0, len(searchedPos)):
    #                         matrix[searchedPos[x][0]][searchedPos[x][1]] -= 1
    #                     cluster = self.Cluster()
    #                     if(len(brother_clusters) != 0):
                            
    #                         cluster.id = len(overall_clusters)
    #                         cluster.x = brother_clusters[0].x
    #                         cluster.y = brother_clusters[0].y
    #                         cluster.depth = matrix[brother_clusters[0].x][brother_clusters[0].y]+1
    #                         cluster.children = []
    #                         cluster.parent = []
    #                         cluster.contains = 0
    #                         for x in range(0, len(brother_clusters)):
    #                             cluster.children.append(brother_clusters[x])
    #                             cluster.contains += brother_clusters[x].contains
    #                             for y in range(0, len(brother_clusters[x].children)):
    #                                 brother_clusters[x].children[y].parent = cluster
    #                         overall_clusters.append(cluster)

    #                         for x in range(0, len(brother_clusters)):
    #                             matrix_with_clusters[brother_clusters[x].x][brother_clusters[x].y] = 0
    #                             if (x == 0):
    #                                 matrix_with_clusters[brother_clusters[x].x][brother_clusters[x].y] = cluster
                                
    #                             current_clusters.remove(brother_clusters[x])
    #                         current_clusters.append(cluster)

    #     clusters_by_level = {}
    #     for depth in range(int(max_depth) + 1):
    #         clusters_by_level[depth] = {}
    #         for node in vertices:
    #             node_id = node[2]
    #             for cluster in overall_clusters:
    #                 if cluster.depth == depth:
    #                     #check if node is anywhere in cluster, check all the children 
    #                     if cluster.contains > 1:
    #                         for child in cluster.children:
    #                             if child.x == node[0] and child.y == node[1]:
    #                                 if node_id not in clusters_by_level[depth]:
    #                                     clusters_by_level[depth][node_id] = []
    #                                 clusters_by_level[depth][node_id].append(cluster.id)

                
    #             if node_id not in clusters_by_level[depth]:     
    #                 clusters_by_level[depth][node_id] = []
                    


        
        """clusters_array = {}
        for depth in range(max_depth + 1):
            if clusters_by_level[depth]:
                
            for node_id in range(max_node_id + 1):
                if node_id in clusters_by_level[depth] and clusters_by_level[depth][node_id]:
                    clusters_array[depth][node_id] = clusters_by_level[depth][node_id][0]
                else:
                    clusters_array[depth][node_id] = -1

        print(clusters_array)"""
        
        
        return clusters_by_level  
                            
    """
    def get_clusters(self, polilines, matrix, vertices):
        
        #initialize the clusters
        overall_clusters = []
        current_clusters = []
        max_depth = 0

        #max depth in matrix
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                matrix[i][j] = int(matrix[i][j])
                max_depth = max(max_depth, matrix[i][j])
        
        #initialize the clusters
        for i in range(0, len(vertices)):
            cluster = self.Cluster()
            cluster.x = vertices[i][0]
            cluster.y = vertices[i][1]
            cluster.depth = matrix[vertices[i][0]][vertices[i][1]]
            cluster.children = []
            cluster.parent = []
            cluster.contains = 1
            current_clusters.append(cluster)

        current_clusters = sorted(current_clusters, key=lambda x: x.depth, reverse=True)
        overall_clusters = current_clusters
        
        for depth in range(int(max_depth), -1, -1):
            
            checkMatrix = np.zeros((IMG_REZ,IMG_REZ))
            if current_clusters.__len__() and current_clusters[0].depth == depth:
                brother_clusters = []
                brother_clusters.append((current_clusters[0].x, current_clusters[0].y))
                brother_clusters.append(self.getBrotherClusters(current_clusters[0].x, current_clusters[0].y, current_clusters[0].depth, matrix, checkMatrix))

                if(len(brother_clusters)!=1):
                    cluster = self.Cluster()
                    cluster.x = current_clusters[0].x
                    cluster.y = current_clusters[0].y
                    cluster.depth = current_clusters[0].depth
                    cluster.children = []
                    cluster.parent = []
                    cluster.contains = 0 
                    for i in range(0, len(brother_clusters)-1):
                        for j in range(0, len(current_clusters)-1):
                            if(i < len(overall_clusters)-1 and  j<(len(current_clusters)-1) and brother_clusters[i][0] == current_clusters[j].x and brother_clusters[i][1] == current_clusters[j].y):
                                cluster.children.append(current_clusters[j])
                                cluster.contains += current_clusters[j].contains
                                current_clusters.pop(j)
                    overall_clusters.append(cluster) 

        return overall_clusters

    

    def getBrotherClusters(self, x, y, depth, matrix, checkMatrix):
        brother_clusters = []
        checkMatrix[x][y] = 1
        for i in range(-1, 2):
            for j in range(-1, 2):
                if(x + i >= 0 and x + i < IMG_REZ and y + j >= 0 and y + j < IMG_REZ and checkMatrix[x + i][y + j] == 0):
                    if(matrix[x + i][y + j] >= depth):
                        brother_clusters.append((x + i, y + j))
                        checkMatrix[x + i][y + j] = 1
                        brother_clusters.append(self.getBrotherClusters(x + i, y + j, depth, matrix, checkMatrix))

        return brother_clusters
    """

    

    def all_edges(self, G):
        list_edges = list(self.G.edges(data = True))
        polylines = []
        for index, (u,v,data) in enumerate(list_edges):
                
            numbers_y = []
            numbers_x = []

            numbers_x = [float(num) for num in data.get('X')]
            
            numbers_y = [float(num) for num in data.get('Y')]
                
            polyline = [(numbers_x[i], numbers_y[i]) for i in range(0, len(numbers_x))]
            polylines.append(polyline)
        return polylines

    def init_matrix(self, polylines, n_clusters='auto', cluster_depth_boost=500):
        """
        Initialize matrix with polyline data and apply k-means clustering to vertices.
        
        Args:
            polylines: List of polylines (each polyline is a list of (x, y) coordinates)
            n_clusters: Number of clusters for k-means, or 'auto' to determine automatically (default: 'auto')
            cluster_depth_boost: Depth value to add to each cluster center (default: 500)
        """
        matrix = np.zeros((IMG_REZ, IMG_REZ))

        # Step 1: Collect all vertices (endpoints of polylines)
        vertices = []
        for polyline in polylines:
            if len(polyline) >= 2:
                # Start point
                vertices.append([polyline[0][0], polyline[0][1]])
                # End point
                vertices.append([polyline[-1][0], polyline[-1][1]])
        
        vertices = np.array(vertices)
        print(f"Collected {len(vertices)} vertices for clustering")
        
        # # Step 2: Determine optimal number of clusters if 'auto'
        # if n_clusters == 'auto':
        #     n_clusters = self._find_optimal_clusters(vertices)
        #     print(f"Automatically determined optimal number of clusters: {n_clusters}")
        
        # # Step 2: Perform k-means clustering on vertices
        # if len(vertices) > 0 and len(vertices) >= n_clusters:
        #     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        #     kmeans.fit(vertices)
        #     cluster_centers = kmeans.cluster_centers_
        #     cluster_labels = kmeans.labels_
            
        #     print(f"K-means clustering completed with {n_clusters} clusters")
        #     print(f"Cluster centers: {cluster_centers}")
            
        #     # Step 3: Add depth to the contour of each cluster
        #     for cluster_id in range(n_clusters):
        #         # Get all vertices belonging to this cluster
        #         cluster_vertices = vertices[cluster_labels == cluster_id]
                
        #         if len(cluster_vertices) < 3:
        #             # Not enough points for a convex hull, just add depth to the points
        #             for vertex in cluster_vertices:
        #                 x = int(np.clip(vertex[0], 0, IMG_REZ - 1))
        #                 y = int(np.clip(vertex[1], 0, IMG_REZ - 1))
        #                 matrix[x][y] += cluster_depth_boost
        #             print(f"Cluster {cluster_id}: Added depth to {len(cluster_vertices)} individual points")
        #         else:
        #             # Create convex hull to get the contour
        #             try:
        #                 hull = ConvexHull(cluster_vertices)
        #                 hull_vertices = cluster_vertices[hull.vertices]
                        
        #                 print(f"Cluster {cluster_id}: Created convex hull with {len(hull_vertices)} vertices")
                        
        #                 # Create a Path object for the hull to check if points are inside
        #                 hull_path = Path(hull_vertices)
                        
        #                 # Find bounding box of the hull
        #                 min_x = int(max(0, np.floor(hull_vertices[:, 0].min())))
        #                 max_x = int(min(IMG_REZ - 1, np.ceil(hull_vertices[:, 0].max())))
        #                 min_y = int(max(0, np.floor(hull_vertices[:, 1].min())))
        #                 max_y = int(min(IMG_REZ - 1, np.ceil(hull_vertices[:, 1].max())))
                        
        #                 # Fill interior with weaker depth (same for entire cluster including edges)
        #                 interior_boost = cluster_depth_boost * 0.3  # 30% of original strength
        #                 for x in range(min_x, max_x + 1):
        #                     for y in range(min_y, max_y + 1):
        #                         if hull_path.contains_point((x, y)):
        #                             matrix[x][y] += interior_boost
                        
        #                 print(f"Cluster {cluster_id}: Filled cluster area with {interior_boost:.1f} depth boost")
                        
        #             except Exception as e:
        #                 print(f"Cluster {cluster_id}: Could not create convex hull ({e}), adding depth to points")
        #                 for vertex in cluster_vertices:
        #                     x = int(np.clip(vertex[0], 0, IMG_REZ - 1))
        #                     y = int(np.clip(vertex[1], 0, IMG_REZ - 1))
        #                     matrix[x][y] += cluster_depth_boost
            
        #     print(f"Added depth boost of {cluster_depth_boost} to contours of {n_clusters} clusters")
        # else:
        #     print(f"Skipping k-means clustering: not enough vertices (need at least {n_clusters})")

        # Step 4: Process polylines as before
        for polyline in polylines:
            checkMatrix = np.zeros((IMG_REZ, IMG_REZ))
            matrix[int(polyline[0][0])][int(polyline[0][1])] += 90
            matrix[int(polyline[-1][0])][int(polyline[-1][1])] += 90
            
            # Interpolate EDGE_REZ points between the points of the polyline
            for i in range(0, len(polyline) - 1):
                x1 = int(polyline[i][0])
                y1 = int(polyline[i][1])
                x2 = int(polyline[i+1][0])
                y2 = int(polyline[i+1][1])
                
                # Interpolate EDGE_REZ points between the points of the polyline
                x = np.linspace(x1, x2, EDGE_REZ)
                y = np.linspace(y1, y2, EDGE_REZ)
                for j in range(0, EDGE_REZ):
                    if(checkMatrix[int(x[j])][int(y[j])] == 0):
                        matrix[int(x[j])][int(y[j])] += 10
                    else:
                        matrix[int(x[j])][int(y[j])] += 2
                    checkMatrix[int(x[j])][int(y[j])] = 1

        return matrix
    
    def _find_optimal_clusters(self, vertices, min_clusters=2, max_clusters=10):
        """
        Automatically determine the optimal number of clusters using silhouette score.
        
        Args:
            vertices: Array of vertex coordinates
            min_clusters: Minimum number of clusters to try (default: 2)
            max_clusters: Maximum number of clusters to try (default: 10)
            
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
                
                # Calculate silhouette score (higher is better, range -1 to 1)
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
    
    def calcMatrix(self, matrix):

        #apply a gaussian filter to the matrix make it way more smooth
        for i in range(0, CONVOLUTION):
            # matrix = signal.convolve2d(matrix, np.array([[1,2,1],[2,4,2],[1,2,1]])/16, mode='same')
            matrix = signal.convolve2d(matrix, np.ones((12,12)), mode='same')

        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.savefig("heatMap.png")
        return matrix
    
    def init_Points(self):
        vetices = []
            
        for node_id, node_data in self.G.nodes(data=True):
            
            nodeX = node_data.get('X')
            nodeY = node_data.get('Y')
            nodeId = node_data.get('id')
            vertex = (int(nodeX), int(nodeY), int(node_id))
            vetices.append(vertex)

        
        return vetices
        
    
    def get_depth_maps(self, matrix):
        """
        Returns a dictionary of depth maps where each key is a depth value
        and the corresponding value is a matrix where pixels of that depth or lower
        are included (marked as 1), and others are excluded (marked as 0).
        
        Args:
            matrix: A 2D numpy array representing depth values
            
        Returns:
            dict: A dictionary mapping depth values to binary matrices
        """
        depth_maps = {}
        
        max_depth = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = int(matrix[i][j])
                max_depth = max(max_depth, matrix[i][j])
        
        for depth in range(int(max_depth) + 1):
            depth_map = np.zeros((IMG_REZ, IMG_REZ))
            
            for i in range(min(len(matrix), IMG_REZ)):
                for j in range(min(len(matrix[i]), IMG_REZ)):
                    if matrix[i][j] <= depth and matrix[i][j] > 0:
                        depth_map[i][j] = 1
            
            depth_maps[depth] = depth_map
        
        return depth_maps