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
        
        

    def get_clusters(self, polilines, matrix, vertices):
        
        #initialize the clusters
        clusters = []
        max_depth = 0
        ver_matrix = np.zeros((IMG_REZ,IMG_REZ))

        #max depth in matrix
        for i in range(0, len(matrix)):
            for j in range(0, len(matrix[i])):
                max_depth = max(max_depth, matrix[i][j])
        
        Cluster = nx.Graph()

        for node in vertices:
            Cluster.add_node(node)

            ver_matrix[node[0]][node[1]] = 1

        
        
        for depth in range (max_depth, 0):
            
            for i in range (0, len(matrix)):
                for j in range (0, len(matrix[i])):
                    checked_matrix = np.zeros((IMG_REZ,IMG_REZ))
                    if(matrix[i][j] == depth):
                        

            

                    

        return clusters

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
                        matrix[int(x[j])][int(y[j])] += 1
                    checkMatrix[int(x[j])][int(y[j])] = 1


        return matrix
    
    def calcMatrix(self, matrix):

        #apply a gaussian filter to the matrix make it way more smooth
        for i in range(0, CONVOLUTION):
            matrix = signal.convolve2d(matrix, np.array([[1,2,1],[2,4,2],[1,2,1]])/16, mode='same')


        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.show()
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
        
    
