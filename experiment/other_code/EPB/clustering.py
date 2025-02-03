import copy
import os
import networkx as nx
import numpy as np
from scipy import signal
from PIL import Image as PILImage
import pickle
import seaborn as sbn
import matplotlib.pyplot as plt
import matplotlib 
from frechetdist import frdist
import re
from matplotlib.backends.backend_pdf import PdfPages
from pdf2image import convert_from_path
import requests
from wand.image import Image
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_violin, geom_boxplot, theme, element_text, labs, element_blank
import math
from other_code.EPB.experiments import Experiment
from typing import List
import pylab
from networkx.drawing.nx_pydot import graphviz_layout

matplotlib.use('qt5Agg')

BIG_Threshold = 10   #Threshlods for the numbers of connected nodes when to consider a cluster big, medium or small
MEDIUM_Threshold = 5
SMALL_Threshold = 2

TIDE_MAX = 100  # Range for TIDE - Number that represents how much space is allowed between a
TIDE_MIN = 1    # node and an edge to be considered in that edges cluster

IMG_REZ = 1600  # Resolution of the image
EDGE_REZ = 100  # Resolution of the edge

CONVOLUTION = 50 # How many times the matrix is convoluted

class Clustering:

    def __init__(self, G, Straight):
        self.Straight = Straight.G
        self.G = G.G
        self.name = G.name

    class Pixel:
        x: int
        y: int
        depth: int

    class node:
        x: int
        y: int
        depth: int
    class Cluster:
        depth: int
        x: int
        children: List['Clustering.Cluster']
        parent: List['Clustering.Cluster']
        parent: []
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
        # Build a directed graph
        G = nx.DiGraph()

        # Map each cluster object to an integer ID
        cluster_map = {}
        for i, cluster in enumerate(clusters):
            cluster_map[cluster] = i

        # Add each cluster as a node, storing the depth
        for cluster in clusters:
            G.add_node(cluster_map[cluster], depth=cluster.depth)

        # Add edges from parent -> child (or child -> parent, as needed)
        for cluster in clusters:
            for child in cluster.children:
                if child in cluster_map:
                    # Directed edge from this cluster -> child
                    G.add_edge(cluster_map[cluster], cluster_map[child])

        # The last cluster in 'clusters' will be our "root"
        root_idx = cluster_map[clusters[-1]]

        # Layout: use graphviz "dot" to create a hierarchical tree
        # root is the node we want at the top.
        pos = graphviz_layout(G, prog='dot', root=root_idx)

        # Extract depths (for coloring)
        depths = [c.depth for c in clusters]
        min_depth = min(depths)
        max_depth = max(depths)

        # Create colormap
        cmap = plt.cm.viridis
        norm = matplotlib.colors.Normalize(vmin=min_depth, vmax=max_depth)
        node_colors = [norm(d) for d in depths]

        # Prepare figure/axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=400,
            cmap=cmap,
            ax=ax
        )
        # Use arrows for the directed edges
        nx.draw_networkx_edges(
            G, pos,
            arrowstyle='-|>',
            arrowsize=12,
            ax=ax
        )
        nx.draw_networkx_labels(
            G, pos,
            labels={cluster_map[c]: f"d={c.depth}" for c in clusters},
            font_color='white',
            ax=ax
        )

        # Colorbar
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Depth')

        ax.set_title("Clusters as a Tree (Root = Last Cluster)")
        ax.axis('off')
        plt.tight_layout()
        
        plt.savefig("clusters_by_depth.png", dpi=300)

    def get_clusters(self, polilines, matrix, vertices):
        

        overall_clusters = []
        current_clusters = []
        matrix_with_clusters = np.zeros((IMG_REZ,IMG_REZ), dtype=object)
        max_depth = 0
        matrix = [
            [1,1,1,1,1,1,1,0,0,0,0,0],
            [1,2,2,2,2,2,1,0,0,0,0,0],
            [1,2,3,3,3,2,1,0,0,0,0,0],
            [1,2,4,3,3,2,1,0,0,0,0,0],
            [1,2,4,4,3,2,1,0,0,0,0,0],
            [1,2,2,2,2,2,1,0,2,2,2,0],
            [1,1,1,1,1,1,1,0,2,3,2,0],
            [0,0,0,0,0,0,0,0,2,2,2,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0]
        ]
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                matrix[i][j] = int(matrix[i][j])
                max_depth = max(max_depth, matrix[i][j])
        
        vertices = [(0,0), (1,1), (0,8), (2,4), (3,4), (4,2), (6,9), (7,9), (9,10)]
        for i in range(0, len(vertices)):
            cluster = self.Cluster()
            cluster.x = vertices[i][0]
            cluster.y = vertices[i][1]
            cluster.depth = matrix[vertices[i][0]][vertices[i][1]]
            cluster.children = []
            matrix_with_clusters[vertices[i][0]][vertices[i][1]] = cluster
            cluster.parent = []
            cluster.contains = 1
            current_clusters.append(cluster)
            overall_clusters.append(cluster)

        for depth in range(int(max_depth), -1, -1):
            
            checkMatrix = np.zeros((IMG_REZ,IMG_REZ))
            for i in range(0, min(matrix.__len__() , IMG_REZ)):
                for j in range(0, min(matrix[i].__len__(), IMG_REZ)):
                    if(matrix[i][j] == depth):
                        searchedPos = []
                        searchedPos.append((i,j))
                        searchStack = []
                        searchStack.append((i,j))
                        brother_clusters = []
                        if(matrix_with_clusters[i][j] != 0):
                            brother_clusters.append(matrix_with_clusters[i][j])
                        
                        while(len(searchStack) != 0):
                            currentPos = searchStack.pop()
                            I = currentPos[0]
                            J = currentPos[1]
                            
                            for x in range (-1, 2):
                                for y in range (-1, 2):
                                    if(I + x >= 0 and I + x < min(matrix.__len__() , IMG_REZ) and J + y >= 0 and J + y < min(matrix[i].__len__(), IMG_REZ) and checkMatrix[I + x][J + y] == 0):
                                        if(matrix[I + x][J + y] >= depth):
                                            searchedPos.append((I + x, J + y))
                                            searchStack.append((I + x, J + y))
                                            checkMatrix[I + x][J + y] = 1
                                            if(matrix_with_clusters[I + x][J + y] != 0):
                                                brother_clusters.append(matrix_with_clusters[I + x][J + y])
                                                checkMatrix[I + x][J + y] = 1
                                    checkMatrix[I][J] = 1
            
                        for x in range(0, len(searchedPos)):
                            matrix[searchedPos[x][0]][searchedPos[x][1]] -= 1
                        cluster = self.Cluster()
                        if(len(brother_clusters) != 0):
                            
                            cluster.x = brother_clusters[0].x
                            cluster.y = brother_clusters[0].y
                            cluster.depth = matrix[brother_clusters[0].x][brother_clusters[0].y]+1
                            cluster.children = []
                            cluster.parent = []
                            cluster.contains = 0
                            for x in range(0, len(brother_clusters)):
                                cluster.children.append(brother_clusters[x])
                                cluster.contains += brother_clusters[x].contains
                                for y in range(0, len(brother_clusters[x].children)):
                                    brother_clusters[x].children[y].parent = cluster
                            overall_clusters.append(cluster)

                            for x in range(0, len(brother_clusters)):
                                matrix_with_clusters[brother_clusters[x].x][brother_clusters[x].y] = 0
                                if (x == 0):
                                    matrix_with_clusters[brother_clusters[x].x][brother_clusters[x].y] = cluster
                                
                                current_clusters.remove(brother_clusters[x])
                            current_clusters.append(cluster)

        return overall_clusters
                            
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

    

    def all_edges(self):
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

    def init_matrix(self, polylines):
        #i want a matrix 1600 x 1600
        matrix = np.zeros((IMG_REZ,IMG_REZ))

        for polyline in polylines:

            checkMatrix = np.zeros((IMG_REZ,IMG_REZ))
            #interpolate EDGE_REZ points between the points of the polyline and increment all the matrix points that are touched by the point, but only once
            for i in range(0, len(polyline) - 1):
                x1 = int(polyline[i][0])
                y1 = int(polyline[i][1])
                x2 = int(polyline[i+1][0])
                y2 = int(polyline[i+1][1])
                #interpolate EDGE_REZ points between the points of the polyline
                x = np.linspace(x1, x2, EDGE_REZ)
                y = np.linspace(y1, y2, EDGE_REZ)
                for j in range(0, EDGE_REZ):
                    if(checkMatrix[int(x[j])][int(y[j])] == 0):
                        matrix[int(x[j])][int(y[j])] += 10
                    checkMatrix[int(x[j])][int(y[j])] = 1


        return matrix
    
    def calcMatrix(self, matrix):

        #apply a gaussian filter to the matrix make it way more smooth
        for i in range(0, CONVOLUTION):
            matrix = signal.convolve2d(matrix, np.array([[1,2,1],[2,4,2],[1,2,1]])/16, mode='same')


        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.savefig("heatMap.png")
        return matrix
    
    def init_Points(self):
        list_edges = list(self.G.edges(data = True))
        vetices = []
        for index, (u,v,data) in enumerate(list_edges):
            numbers_y = []
            numbers_x = []
            numbers_x = [float(num) for num in data.get('X')]
            numbers_y = [float(num) for num in data.get('Y')]
            for i in range(0, len(numbers_x)):
                #append the integere value of the coordonates
                vetices.append((int(numbers_x[i]), int(numbers_y[i])))
        return vetices
        
    
