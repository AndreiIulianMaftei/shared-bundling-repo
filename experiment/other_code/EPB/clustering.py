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

matplotlib.use('qt5Agg')

BIG_Threshold = 10   #Threshlods for the numbers of connected nodes when to consider a cluster big, medium or small
MEDIUM_Threshold = 5
SMALL_Threshold = 2

TIDE_MAX = 100  # Range for TIDE - Number that represents how much space is allowed between a
TIDE_MIN = 1    # node and an edge to be considered in that edges cluster


class Clustering:

    def __init__(self, G, Straight):
        self.Straight = Straight.G
        self.G = G.G
        self.name = G.name

    def get_clusters(self, algorithm):
        
        for TIDE in range(TIDE_MIN, TIDE_MAX):
            
            polylines = []
            list_edges = list(self.G.edges(data = True))

            for index, (u,v,data) in enumerate(list_edges):
                
                Y = []
                X = []
                polyline = []
                X = [float(num) for num in data.get('X')]
                Y = [float(num) for num in data.get('Y')]

                x0 = self.G.nodes[u]['X']
                y0 = self.G.nodes[u]['Y']

                x1 = self.G.nodes[v]['X']
                y1 = self.G.nodes[v]['Y'] 
                for i in range(0, len(X)):
                    polyline.append((X[i], Y[i]))



    
        
    
