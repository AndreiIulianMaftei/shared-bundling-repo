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

GREY_THRESH = 255                       #Threshold when a pixel is considered as 'occupied'
ANGLE_THRESHOLD = 2 * np.pi * 7.5 / 360  #Angle threshold when two edges crossing are considered ambiguous
SCALEFACTOR = 0.25                      #Scale down input image for more efficient processing
AMBIGUITY_SP_CUTOFF = 5                 #Compute ambiguity up to a hop distance

class Experiment:

    def __init__(self, G, Straight):
        self.Straight = Straight.G
        self.G = G.G
        self.name = G.name
        
    

    def run(self, path):
        '''
        Calculate the metrics from the paper.
        '''
        ink = self.calcInkRatio(path)
        dist = self.calcDistortion()
        amb = self.calcAmbiguity(path)

        return ink, dist, amb

    def calcInk(self, path):
        imPath = path + self.name + '.png'
        imGray = np.array(PILImage.open(imPath).convert('L'))

        allPixels = imGray.size

        greyscalepixel = (imGray < GREY_THRESH).sum()

        return greyscalepixel, allPixels

    def calcInkRatio(self, path):
        '''
        Calculate the ink ratio. Assumes that there is a .png with the investigated algorithm and a straight line drawing.
        '''

        imPath = path + 'straight' + '.png'
        imGray = np.array(PILImage.open(imPath).convert('L'))

        inkratioG = (imGray < GREY_THRESH).sum()

        imPath = path + self.name + '.png'
        imGray = np.array(PILImage.open(imPath).convert('L'))

        greyscalepixel = (imGray < GREY_THRESH).sum()
        inkratio = greyscalepixel / inkratioG

        return inkratio
    def calcMonotonicity(self, algorithm):

        monotonicitys = []
        polylines = []
        x = 0
       
        
        list_edges = list(self.G.edges(data = True))

        for index, (u,v,data) in enumerate(list_edges):
            
            
            if index == 1 and algorithm == "wr":
                yes = 1
            Y = []
            X = []
            polyline = []
            X = [float(num) for num in data.get('X')]
            Y = [float(num) for num in data.get('Y')]

            x0 = self.G.nodes[u]['X']
            y0 = self.G.nodes[u]['Y']

            x1 = self.G.nodes[v]['X']
            y1 = self.G.nodes[v]['Y']
            
            
            monotonicity = 0
            previous_sign = None

            for i in range(0, len(X)):
                polyline.append((X[i], Y[i]))
            index2 = 0
            A = Experiment.Point(x0,y0)
            B = Experiment.Point(polyline[1][0],polyline[1][1])
            C = Experiment.Point(x1,y1)
            orientation = Experiment.orientation(A, B, C)
            for point in polyline:
                
                    
                point1x = point[0]
                point1y = point[1]
                A = Experiment.Point(x0,y0)
                B = Experiment.Point(point1x, point1y)
                C = Experiment.Point(x1, y1)
                orientation2 = Experiment.orientation(A, B, C)

                projection = Experiment.project_point_onto_line((point1x, point1y), (x0, y0), (x1, y1))
                
                distance = np.linalg.norm(np.array((point1x, point1y)) - np.array(projection))
                if math.isclose(distance, 0, abs_tol=1e-5):
                    distance = 0
                if orientation!= orientation2 and distance != 0:
                    distance = distance * -1
                distanceFromSource = np.linalg.norm(projection - np.array((x0, y0)))
                newPoint = (distanceFromSource, distance)
                polyline[index2] = newPoint
                index2 += 1
            #print(index, len(polyline))
            xDirection = None
            yDirection = None
            index2 = 0
            for point in polyline:
                #compute the monocity by checking if the xOrientation or y orientation changes
                if index2 == 1:
                    if point[0] >= polyline[index2 - 1][0] and xDirection == None:
                        xDirection = 1                    
                    if point[0] < polyline[index2 - 1][0] and xDirection == None:
                        xDirection = -1
                    if point[1] >= polyline[index2 - 1][1] and yDirection == None:
                        yDirection = 1
                    if point[1] < polyline[index2 - 1][1] and yDirection == None:
                        yDirection = -1
                checked = 0 
                if index2 > 1:
                    if point[0]> polyline[index2 - 1][0] and xDirection == -1:
                        checked += 1
                        xDirection = 1
                        
                    if point[0] < polyline[index2 - 1][0] and xDirection == 1:
                        checked += 1
                        xDirection = -1
                        
                    if point[1] > polyline[index2 - 1][1] and yDirection == -1:
                        checked += 1
                        yDirection = 1

                    if point[1] < polyline[index2 - 1][1] and yDirection == 1:
                        checked += 1
                        yDirection = -1
                if checked != 0:
                    monotonicity += 1
                index2 += 1
            monotonicitys.append(monotonicity)

        return monotonicitys
    def calcNumberOfSegments(self, algorithm):
        segments = []
        list_edges = list(self.G.edges(data = True))
        for index, (u,v,data) in enumerate(list_edges):
            numbers_y = []
            numbers_x = []
            numbers_x = [float(num) for num in data.get('X')]
            numbers_y = [float(num) for num in data.get('Y')]
            polyline = [(numbers_x[i], numbers_y[i]) for i in range(0, len(numbers_x))]
            segments = segments + [len(polyline)]
        return segments
    def calcAngle(self, algorithm):
        angles = []
        list_edges = list(self.G.edges(data = True))
        for index, (u,v,data) in enumerate(list_edges):
            

            if index == 19:
                yes = 1
            Y = []
            X = []
            polyline = []
            X = [num for num in data.get('X')]
            Y = [num for num in data.get('Y')]

            x0 = self.G.nodes[u]['X']
            y0 = self.G.nodes[u]['Y']

            x1 = self.G.nodes[v]['X']
            y1 = self.G.nodes[v]['Y']
            
            
            monotonicity = 0
            previous_sign = None

            
            for i in range(0, len(X)):
                polyline.append((X[i], Y[i]))
            
            index = 0
            for point in polyline:
                if index > 0 and index < len(polyline) - 1:
                    A = Experiment.Point(polyline[index - 1][0], polyline[index - 1][1])
                    B = Experiment.Point(polyline[index][0], polyline[index][1])
                    C = Experiment.Point(polyline[index + 1][0], polyline[index + 1][1])
                    
                    BA = (B.x - A.x, B.y - A.y)
                    BC = (C.x - B.x, C.y - B.y)
                    
                    dot_product = BA[0]*BC[0] + BA[1]*BC[1]

                    cross = BA[0]*BC[1] - BA[1]*BC[0]

                    angle_radians = abs(math.atan2(cross, dot_product))

                    angle= math.degrees(angle_radians)
                    if angle > 90:
                        test = 1
                    '''                    
                    mag_BA = math.sqrt(BA[0]**2 + BA[1]**2)
                    mag_BC = math.sqrt(BC[0]**2 + BC[1]**2)
                    
                    if mag_BA == 0 or mag_BC == 0:
                        angles.append(0)
                        index += 1
                        continue           
                             
                    cos_theta = dot_product / (mag_BA * mag_BC)
                    
                    cos_theta = max(min(cos_theta, 1.0), -1.0)
                    
                    theta_radians = math.acos(cos_theta)
                    
                    theta_degrees = math.degrees(theta_radians)
                    '''
                    angles.append(angle)
                index += 1

        return angles
                
    
    def calcFrechet(self, algorithm):

        ma = -999999999999
        mi = 999999999999
        frechet = []
        polylines = []
        x= 0
        
        #print(G.edges(data = True))
        
        if(algorithm != "wt"):
            
            list_edges = list(self.G.edges(data = True))

            for index, (u,v,data) in enumerate(list_edges):
                
               

                numbers_y = []
                numbers_x = []

                numbers_x = [float(num) for num in data.get('X')]
            
                numbers_y = [float(num) for num in data.get('Y')]
                
                polyline = [(numbers_x[i], numbers_y[i]) for i in range(0, len(numbers_x))]
                polylines.append(polyline)

                x0 = self.G.nodes[u]['X']
                y0 = self.G.nodes[u]['Y']

                x1 = self.G.nodes[v]['X']
                y1 = self.G.nodes[v]['Y']
                
                t_values = np.linspace(0, 1, num=len(polyline))

                x_values = x0 + t_values * (x1 - x0)
                y_values = y0 + t_values * (y1 - y0)

                interpolated_points = [(float(x), float(y)) for x, y in zip(x_values, y_values)]
                
                if(len(polyline) == 0):
                    x = 0
                else:
                    x = frdist(interpolated_points, polyline)
                
                
                frechet.append(x)
                ma = max(ma, x)
                mi = min(mi, x)

        
        return frechet
    def all_edges(self):
        
        polylines = []
        x= 0
        list_edges = list(self.G.edges(data = True))
        '''for index, (u,v,data) in enumerate(list_edges):
            numbers_y = []
            numbers_x = []

            numbers_x = [float(num) for num in data.get('X')]
            
            numbers_y = [float(num) for num in data.get('Y')]
                
            polyline = [(numbers_x[i], numbers_y[i]) for i in range(0, len(numbers_x))]
            polylines.append(polyline)'''
        return list_edges
    class Point: 
        def __init__(self, x, y): 
            self.x = x 
            self.y = y 
    def all_intersection(self, list_edges):
        intersections = 0
        
        for index, (u,v,data) in enumerate(list_edges):
            numbers_y = []
            numbers_x = []
            numbers_x = [float(num) for num in data.get('X')]
            numbers_y = [float(num) for num in data.get('Y')]
            polyline1 = [(numbers_x[i], numbers_y[i]) for i in range(0, len(numbers_x))]
            for index2, (u2,v2,data2) in enumerate(list_edges):
                if index != index2:
                    numbers_y2 = []
                    numbers_x2 = []
                    numbers_x2 = [float(num) for num in data2.get('X')]
                    numbers_y2 = [float(num) for num in data2.get('Y')]
                    polyline2 = [(numbers_x2[i], numbers_y2[i]) for i in range(0, len(numbers_x2))]
                    for point1 in polyline1:
                        for point2 in polyline2:
                            
                            if point1 != polyline1[-1] and point2 != polyline2[-1]:
                                A = Experiment.Point(point1[0], point1[1])
                                B = Experiment.Point(polyline1[polyline1.index(point1) + 1][0], polyline1[polyline1.index(point1) + 1][1])
                                C = Experiment.Point(point2[0], point2[1])
                                D = Experiment.Point(polyline2[polyline2.index(point2) + 1][0], polyline2[polyline2.index(point2) + 1][1])
                            if Experiment.doIntersect(A, B, C, D):
                                intersections += 1

        return intersections

            
                
                
    def project_point_onto_line(point, line_start, line_end):
        
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        line_len_squared = np.dot(line_vec, line_vec)
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_squared))
        projection = np.array(line_start) + t * line_vec
        return projection
    
    def fastFrechet(self, algorithm):
        frechet = []
        polylines = []
        x= 0
        if(algorithm != "zz"):
            list_edges = list(self.G.edges(data = True))

            for index, (u,v,data) in enumerate(list_edges):
                
                

                numbers_y = []
                numbers_x = []

                numbers_x = [float(num) for num in data.get('X')]
            
                numbers_y = [float(num) for num in data.get('Y')]
                
                polyline = [(numbers_x[i], numbers_y[i]) for i in range(0, len(numbers_x))]
                polylines.append(polyline)


                x0 = self.G.nodes[u]['X']
                y0 = self.G.nodes[u]['Y']

                x1 = self.G.nodes[v]['X']
                y1 = self.G.nodes[v]['Y']

                distances = []
                for i in range(0, len(polyline)):
                    point = polyline[i]
                    projection  = Experiment.project_point_onto_line(point, (x0, y0), (x1, y1))
                    #if the projection is outside the line segment, calculate the distance to the closest endpoint
                    if np.dot(np.array(projection) - np.array((x0, y0)), np.array((x1, y1)) - np.array((x0, y0))) < 0:
                        distance = np.linalg.norm(np.array(point) - np.array((x0, y0)))
                    elif np.dot(np.array(projection) - np.array((x1, y1)), np.array((x0, y0)) - np.array((x1, y1))) < 0:
                        distance = np.linalg.norm(np.array(point) - np.array((x1, y1)))
                    
                    distance = np.linalg.norm(np.array(point) - np.array(projection))

                    distances.append(distance)
                    #print(projection)

                x = max(distances)
                frechet.append(x)


        
        return frechet
    
    def projection_Monotonicty(self, algorithm):
        monotonicities = []

        list_edges = list(self.G.edges(data=True))

        for index, (u, v, data) in enumerate(list_edges):

            numbers_x = [float(num) for num in data.get('X')]
            numbers_y = [float(num) for num in data.get('Y')]
            polyline = [(numbers_x[i], numbers_y[i]) for i in range(len(numbers_x))]
            projections = []
            mono = 0
            orientation = 0
            x0 = self.G.nodes[u]['X']
            y0 = self.G.nodes[u]['Y']
            x1 = self.G.nodes[v]['X']
            y1 = self.G.nodes[v]['Y']
            
            index = 0 
            for point in polyline:
                
                projection = Experiment.project_point_onto_line(point, (x0, y0), (x1, y1))
                projections.append(projection)

                index += 1
            index = 0
            order = 0
            for point in projections:
                if index > 0:
                    A = Experiment.Point(projections[index - 1][0], projections[index - 1][1])
                    B = Experiment.Point(point[0], point[1])
                    
                    if index == 1:
                        if A.x < B.x:
                            order = 1
                        else:
                            order = -1
                    else:
                        if A.x < B.x and order != 1:
                            mono += 1
                            order = 1
                        if A.x > B.x and order != -1:
                            mono += 1
                            order = -1
            
                index += 1
            if mono == 5:
                test = 1
            monotonicities.append(mono)
        return monotonicities

    def onSegment(p, q, r): 
        if ( (q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and 
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))): 
            return True
        return False
    def orientation(p, q, r): 
        
        
        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
        if (val > 0): 
            
            # Clockwise orientation 
            return 1
        elif (val < 0): 
            
            # Counterclockwise orientation 
            return 2
        else: 
            
            # Collinear orientation 
            return 0
    def doIntersect(p1,q1,p2,q2): 
        
        
        o1 = Experiment.orientation(p1, q1, p2) 
        o2 = Experiment.orientation(p1, q1, q2) 
        o3 = Experiment.orientation(p2, q2, p1) 
        o4 = Experiment.orientation(p2, q2, q1) 
    
        # General case 
        if ((o1 != o2) and (o3 != o4)): 
            return True
    
        # Special Cases 
    
        if ((o1 == 0) and Experiment.onSegment(p1, p2, q1)): 
            return True
    
        if ((o2 == 0) and Experiment.onSegment(p1, q2, q1)): 
            return True
    
        if ((o3 == 0) and Experiment.onSegment(p2, p1, q2)): 
            return True
    
        if ((o4 == 0) and Experiment.onSegment(p2, q1, q2)): 
            return True
    
        return False
    def line_intersection(A, B, C, D):
        
        a1 = B[1] - A[1]
        b1 = A[0] - B[0]
        c1 = a1 * A[0] + b1 * A[1]

        a2 = D[1] - C[1]
        b2 = C[0] - D[0]
        c2 = a2 * C[0] + b2 * C[1]

        determinant = a1 * b2 - a2 * b1

        if determinant == 0:
            return None   
        else:
            x = (b2 * c1 - b1 * c2) / determinant
            y = (a1 * c2 - a2 * c1) / determinant

            if (
                min(A[0], B[0]) <= x <= max(A[0], B[0]) and
                min(A[1], B[1]) <= y <= max(A[1], B[1]) and
                min(C[0], D[0]) <= x <= max(C[0], D[0]) and
                min(C[1], D[1]) <= y <= max(C[1], D[1])
            ):
                return (x, y)
            else:
                return None

    def count_self_intersections(self, algorithm):
        """Count the number of self-intersections in a polyline defined by a list of points."""
        all_intersections = []
        monotonicities = []
        number_of_intersections = 0

        list_edges = list(self.G.edges(data=True))
        self_intersected = []
        for index, (u, v, data) in enumerate(list_edges):
            if index >100:
                break
            numbers_y = []
            numbers_x = []

            numbers_x = data['X']

            numbers_y = data['Y']

            intersections = 0
            ok = 0
            polyline = [(numbers_x[i], numbers_y[i]) for i in range(0, len(numbers_x))]
            for i in range(len(polyline) - 1):
                for j in range(i + 2, len(polyline) - 1):
                    if i == 0 and j == len(polyline) - 1:
                        continue
                    
                    A = Experiment.Point(polyline[i][0], polyline[i][1])
                    B = Experiment.Point(polyline[i + 1][0], polyline[i + 1][1])
                    C = Experiment.Point(polyline[j][0], polyline[j][1])
                    D = Experiment.Point(polyline[j + 1][0], polyline[j + 1][1])
                    if(Experiment.doIntersect(A, B, C, D)):
                        intersections += 1
                        ok = 1
            if ok == 1:
                self_intersected.append(polyline)
            if intersections == 3:
                test = 1
            number_of_intersections += intersections
            all_intersections.append(intersections)
        return all_intersections, number_of_intersections


    def plotMegaGraph(self, algorithms, metric, histogram_image_paths, ink_ratios):


        def embed_image(ax, image_path, title, subtitle=None):
            try:
                img = plt.imread(image_path)
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(title, fontsize=10)
                if subtitle:
                    ax.text(0.5, -0.1, subtitle, fontsize=9, ha='center', va='top', transform=ax.transAxes)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                ax.text(0.5, 0.5, "Error loading image", ha="center", va="center", fontsize=12)
                ax.axis('off')

        pdf_path = f"output_comparison_{metric}.pdf"
        with PdfPages(pdf_path) as pdf:
            fig = plt.figure(figsize=(12, 12))
 
            for i, graph_path in enumerate(algorithms):
                ax = plt.subplot2grid((3, 3), (0, i))
                ink_ratio = ink_ratios[i]
                subtitle = f"Ink Ratio: {ink_ratio}"
                embed_image(ax, f'output/airlines/images/{graph_path}.png', f"Graph {algorithms[i]}", subtitle=subtitle)

            for i, hist_path in enumerate(histogram_image_paths):
                ax = plt.subplot2grid((3, 3), (1, i))
                embed_image(ax, hist_path, f"{metric} Histogram {i+1}")

            violin_plot_path = f"violin_plot_{metric}.png"
            ax = plt.subplot2grid((3, 3), (2, 0), colspan=3)
            embed_image(ax, violin_plot_path, f"{metric} Violin Plot")

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

        print(f"PDF created at: {pdf_path}")

    def plotAll(self, algorithms, metrics):
        for index, rez in enumerate(algorithms):

            main_graph_image_path = f'output/airlines/images/{rez}.png'  
            histogram_pdf_paths = []
            for metric in metrics:
                histogram_pdf_paths.append(f"{metric}_normalisation_{rez}.png")

            def create_histogram(ax, data, title):
                ax.hist(data, bins=10, alpha=0.7, color='blue')
                ax.set_title(title)
                ax.grid(True)

            
            def convert_pdf_to_image(pdf_path, dpi=300):
                with Image(filename=pdf_path, resolution=dpi) as img:
                    img.format = 'png'
                    img.alpha_channel = 'remove'  
                    return img.clone()

            def embed_image(ax, image_path, title):
                img = plt.imread(image_path) 
                ax.imshow(img)  
                ax.axis('off')  
                ax.set_title(title)
            
            pdf_path = f'output_layout_with_image_{rez}.pdf'
            

            with PdfPages(pdf_path) as pdf:
                fig = plt.figure(figsize=(8, 10))

                main_graph_ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                main_graph_img = plt.imread(main_graph_image_path)
                main_graph_ax.imshow(main_graph_img)
                main_graph_ax.axis('off')  
                main_graph_ax.set_title("Main Graph (Image)")

                for i, hist_pdf_path in enumerate(histogram_pdf_paths):
                    ax = plt.subplot2grid((3, 3), (2, i))
                    try:
                        embed_image(ax, hist_pdf_path, f"Histogram {metrics[i]}")
                    except Exception as e:
                        print(f"Error processing {hist_pdf_path}: {e}")
                        ax.text(0.5, 0.5, "Error loading PDF", ha="center", va="center", fontsize=12)
                        ax.axis('off')

                plt.tight_layout()
                pdf.savefig(fig)
                plt.close()

            print(f"PDF created at: {pdf_path}")

        
  


        
        

        
       


        
    def plotHistogram(self, values):
        sbn.displot(x=values)
        plt.show()

    def calcDistortion(self):
        '''
        Calculate the distortion by summing up the polyline segments of the Bezier approximation.
        '''
        lPoly = 0
        lStraight = 0

        distortions = []

        for source, target, data in self.G.edges(data=True):
            X = data['X']
            Y = data['Y']

            x0 = self.G.nodes[source]['X']
            x1 = self.G.nodes[target]['X']
            y0 = self.G.nodes[source]['Y']
            y1 = self.G.nodes[target]['Y']

            lPoly = 0
            lStraight = np.sqrt(np.square(x0-x1) + np.square(y0-y1))    

            if len(X) <= 2:
                distortion = 1
                lPoly = lStraight
            else:
                for i in range(len(X) - 1):
                    x0 = X[i]
                    x1 = X[i + 1]
                    y0 = Y[i]
                    y1 = Y[i + 1]

                    lPoly += np.sqrt(np.square(x0-x1) + np.square(y0-y1))        

                distortion =  lPoly / lStraight

            distortions.append(distortion)

        return (np.min(distortions), np.mean(distortions), np.std(distortions), np.var(distortions), np.median(distortions), distortions)
    
                    

    def calcAmbiguity(self, path):
        '''
        Calculate the ambiguity. Store intermediate results in the folder /pickle and reuse. Important if parameters for the ambiguity are changed then delete the files.
        '''
        try:
            edgeEdgeSet = pickle.load(open(f'{path}/pickle/{self.name}.ees', 'rb'))
            edgeSet = pickle.load(open(f'{path}/pickle/{self.name}.es', 'rb'))
            print(f'{self.name} pickle found, processing now')

            return self.ambiguityCalculation(edgeSet, edgeEdgeSet)
        except (OSError, IOError) as e:
            print('No pickle found, processing now')

            edgeSet, edgeEdgeSet = self.preprocessAmbiguity(path)

            return self.ambiguityCalculation(edgeSet, edgeEdgeSet)

    def ambiguityCalculation(self, edgeSetOrg, edgeEdgeSetOrg):
        '''
        Calculate the ambiguity by processing the ambiguous edges.
        '''
        edges = list(self.G.edges())
        trueEdges = dict()
        falseEdges = dict()
        reachable1 = dict()
        unreachable1 = dict()

        vDictUnreachable = dict()
        vDictReachable = dict()

        apsp = dict(nx.all_pairs_shortest_path_length(self.G))
        
        Ambiguities = []
        
        for cutoff in range(1, AMBIGUITY_SP_CUTOFF + 1):
            edgeSet = edgeSetOrg.copy()
            edgeEdgeSet = edgeEdgeSetOrg.copy()
        
            for n in self.G.nodes():
                vDictReachable[n] = set()
                vDictUnreachable[n] = set()

            '''
            Remove edges that are not ambiguous as the edge exists 
            '''
            for s1,t1,data in self.G.edges(data=True):
                toRemove = []

                trueEdges[(s1,t1)] = set()
                falseEdges[(s1,t1)] = set()
                for e in edgeEdgeSet[(s1,t1)]:
                    s2,t2 = edges[e]

                    if self.G.has_edge(s1,t2) and self.G.has_edge(t1,s2):
                        toRemove.append(e)
                        trueEdges[(s1,t1)].add(e)
                    elif self.G.has_edge(t1,t2) and self.G.has_edge(s1,s2):
                        toRemove.append(e)
                        trueEdges[(s1,t1)].add(e)
                    else:
                        falseEdges[(s1,t1)].add(e)

                trueEdges[(s1,t1)].add(-1)
                edgeEdgeSet[(s1,t1)].difference_update(toRemove)

                reachable1[(s1,t1)] = set()
                unreachable1[(s1,t1)] = set()
                for v in edgeSet[(s1,t1)]:
                    if v is s1:
                        continue
                    elif v in apsp[s1] and apsp[s1][v] <= cutoff:
                        reachable1[(s1,t1)].add(v)
                        vDictReachable[s1].add(v)
                    else:
                        unreachable1[(s1,t1)].add(v)
                        vDictUnreachable[s1].add(v)

                reachable1[(t1,s1)] = set()
                unreachable1[(t1,s1)] = set()
                for v in edgeSet[(t1,s1)]:
                    if v is t1:
                        continue
                    elif v in apsp[t1] and apsp[t1][v] <= cutoff:
                        reachable1[(t1,s1)].add(v)
                        vDictReachable[t1].add(v)
                    else:
                        unreachable1[(t1,s1)].add(v)
                        vDictUnreachable[t1].add(v)

                vDictReachable[t1].add(s1)
                vDictReachable[s1].add(t1)
                reachable1[(s1,t1)].add(t1)
                reachable1[(t1,s1)].add(s1)

            oben = 0
            unten = 0

            ambVAmbiguity = []
            for n in self.G.nodes():
                oben += len(vDictUnreachable[n])
                unten += len(vDictUnreachable[n]) + len(vDictReachable[n])

                lie1 = len(vDictUnreachable[n]) / (len(vDictUnreachable[n]) + len(vDictReachable[n]))

                ambVAmbiguity.append(lie1)

            Ambiguities.append(np.mean(ambVAmbiguity))
        
        return Ambiguities

    def preprocessAmbiguity(self, outPath):
        '''
        Compute which edges in the drawing are close and parallel or have a shallow crossing.
        '''

        G = self.copyAndScale(SCALEFACTOR)

        w1 = int(np.ceil(G.graph['xmax']))
        h1 = int(np.ceil(G.graph['ymax']))

        occup = np.zeros((w1 + 1,h1 + 1,len(G.edges())), dtype=np.int32)
        angle = np.zeros((w1 + 1,h1 + 1,len(G.edges())), dtype=np.float64)
        pixelpathLength = np.zeros(len(G.edges()))
        #print(occup.shape)

        z = 0
        for source, target, data in G.edges(data=True):
            X = data['X']
            Y = data['Y']

            path = []
        
            for i in range(len(X) - 1):
                x0 = X[i]
                x1 = X[i + 1]
                y0 = Y[i]
                y1 = Y[i + 1]

                l = int(np.ceil(np.sqrt(np.square(x0-x1) + np.square(y0-y1)) / 0.4))

                a = np.arctan2(y0 - y1, x0 - x1)
                
                if a < 0:
                    a = 2 * np.pi + a                   
    
                x = np.linspace(x0, x1, num=l)
                y = np.linspace(y0, y1, num=l)

                pX = np.empty_like(x)
                pY = np.empty_like(y)

                np.floor(x, pX)
                np.floor(y, pY)
                pX = pX.astype(int)
                pY = pY.astype(int)

                xOld = -1
                yOld = -1

                for j in range(l):
                    try:
                        if not (pX[j] == xOld and pY[j] == yOld ):
                            path.append((pX[j],pY[j]))
                            xOld = pX[j]
                            yOld = pY[j]

                            angle[pX[j]][pY[j]][z] = a
                            occup[pX[j]][pY[j]][z] = 1
                    except IndexError:
                        continue

            #print(len(path),path)
            data['Path'] = path
            pixelpathLength[z] = len(path) - 2
            z = z + 1

        kernel = np.ones((3,3))

        conv = occup
        aangle = angle

        pixelDict = dict()

        for z in range(len(occup[0][0])):
            arr = occup[:,:,z]
            arr = signal.convolve2d(arr, kernel, boundary='fill', mode='same')
            
            arr2 = angle[:,:,z]
            arr2 = signal.convolve2d(arr2, kernel,boundary='fill', mode='same')

            arr2[arr > 1] = arr2[arr > 1] / arr[arr > 1]        
            arr[arr > 1] = 1

            conv[:, :, z] = arr
            aangle[:,:,z] = arr2

        print('done Convolution')

        for id, data in G.nodes(data=True):
            x = int(data['X'])
            y = int(data['Y'])

            for xx in [-1,0,1]:
                for yy in [-1,0,1]:
                    try:
                        conv[x + xx, y + yy,:] = 0
                        aangle[x + xx, y + yy,:] = 0.0
                    except IndexError:
                        continue

        print('done removing vertices')

        for x in range(w1):
            for y in range(h1):
                l = []

                for z in range(len(occup[0][0])):
                    if conv[x][y][z]:
                        l.append(z)

                pixelDict[(x,y)] = l

        #amb = np.zeros((w1,h1))
        amb = np.zeros((w1,h1, len(G.edges())))

        print('done building structure')
        edgeAmbSet = dict()
        edgeEdgeAmbSet = dict()
        edges = list(G.edges())
        
        for s,t in G.edges():
            edgeEdgeAmbSet[(s,t)] = set()
            edgeAmbSet[(s,t)] = set()
            edgeAmbSet[(t,s)] = set()
            edgeAmbSet[(s,t)].add(t)
            edgeAmbSet[(t,s)].add(s)

        ambVal = np.zeros((len(G.edges()), len(G.edges())))
        ambEdge = np.zeros(len(G.edges()))

        for x in range(w1):
            for y in range(h1):
                lll = pixelDict[(x,y)]

                for i in range(len(lll)):
                    for j in range(i + 1, len(lll)):
                        e1 = lll[i]
                        e2 = lll[j]

                        a1 = aangle[x,y,e1]
                        a2 = aangle[x,y,e2]

                        a = np.mod(a1 - a2, 2 * np.pi)
                        aa = np.mod(a2 - a1, 2 * np.pi)
                        a = np.min([a, aa])
                        aTest = np.min([a, np.pi - a])

                        if aTest < ANGLE_THRESHOLD:
                            amb[x,y,z] += 1


                            ambEdge[e1] += 1
                            ambEdge[e2] += 1

                            ambVal[e1,e2] += 1
                            ambVal[e2,e1] += 1

                            s1,t1 = edges[e1]
                            s2,t2 = edges[e2]

                            edgeEdgeAmbSet[(s1,t1)].add(e2)
                            edgeEdgeAmbSet[(s2,t2)].add(e1)

                            if a < np.pi - a:
                                edgeAmbSet[(s1, t1)].add(t2)
                                edgeAmbSet[(t1, s1)].add(s2)
                                edgeAmbSet[(t2, s2)].add(s1)
                                edgeAmbSet[(s2, t2)].add(t1)
                            else:
                                edgeAmbSet[(s1, t1)].add(s2)
                                edgeAmbSet[(t1, s1)].add(t2)
                                edgeAmbSet[(t2, s2)].add(t1)
                                edgeAmbSet[(s2, t2)].add(s1)


        print('done calculating')
    
        pickle.dump(edgeEdgeAmbSet, open(f'{outPath}/pickle/{self.name}.ees', 'wb'))
        pickle.dump(edgeAmbSet, open(f'{outPath}/pickle/{self.name}.es', 'wb'))

        return edgeAmbSet, edgeEdgeAmbSet

    def copyAndScale(self, factor):
        '''
        Copy the graph and scale the position by factor. requires a graph with positive positions.

        G - Graph
        factor - scaling factor

        return
        G - Scaled and copied graph
        '''
        G = copy.deepcopy(self.G)

        for n, data in G.nodes(data=True):
            data['X'] = data['X'] * factor
            data['Y'] = data['Y'] * factor

        for t,s,data in G.edges(data=True):
            X = []
            Y = []

            for x in data['X']:
                X.append(x * factor)

            for y in data['Y']:
                Y.append(y * factor)

            data['X'] = X
            data['Y'] = Y
    
        G.graph['xmax'] = G.graph['xmax'] * factor
        G.graph['ymax'] = G.graph['ymax'] * factor

        return G




class Metrics:
    
    @staticmethod
    def calcInk(path):
        imPath = path + '.png'
        imGray = np.array(Image.open(imPath).convert('L'))

        allPixels = imGray.size

        greyscalepixel = (imGray < GREY_THRESH).sum()

        return greyscalepixel, allPixels

    @staticmethod
    def aspect_ratio(path):
        imPath = path + '.png'
        imGray = np.array(Image.open(imPath).convert('L'))

        return imGray.shape

    @staticmethod
    def calcDistortion(G):
        '''
        Calculate the distortion by summing up the polyline segments of the Bezier approximation.
        '''
        lPoly = 0
        lStraight = 0

        distortions = []

        for source, target, data in G.edges(data=True):
            X = data['X']
            Y = data['Y']

            x0 = G.nodes[source]['X']
            x1 = G.nodes[target]['X']
            y0 = G.nodes[source]['Y']
            y1 = G.nodes[target]['Y']

            lPoly = 0
            lStraight = np.sqrt(np.square(x0-x1) + np.square(y0-y1))    

            if len(X) <= 2:
                distortion = 1
                lPoly = lStraight
            else:
                for i in range(len(X) - 1):
                    x0 = X[i]
                    x1 = X[i + 1]
                    y0 = Y[i]
                    y1 = Y[i + 1]

                    lPoly += np.sqrt(np.square(x0-x1) + np.square(y0-y1))        

                distortion =  lPoly / lStraight

            distortions.append(distortion)

        return (np.min(distortions), np.mean(distortions), np.std(distortions), np.var(distortions), np.median(distortions))

    @staticmethod
    def calcAmbiguity(G, path, name):
        '''
        Calculate the ambiguity. Store intermediate results in the folder /pickle and reuse. Important if parameters for the ambiguity are changed then delete the files.
        '''
        try:
            edgeEdgeSet = pickle.load(open(f'{path}/pickle/{name}.ees', 'rb'))
            edgeSet = pickle.load(open(f'{path}/pickle/{name}.es', 'rb'))
            print(f'{name} pickle found, processing now')

            return Metrics.ambiguityCalculation(G, edgeSet, edgeEdgeSet)
        except (OSError, IOError) as e:
            print('No pickle found, processing now')

            edgeSet, edgeEdgeSet = Metrics.preprocessAmbiguity(G, path, name)

            return Metrics.ambiguityCalculation(G, edgeSet, edgeEdgeSet)

    @staticmethod
    def ambiguityCalculation(G, edgeSetOrg, edgeEdgeSetOrg):
        '''
        Calculate the ambiguity by processing the ambiguous edges.
        '''
        edges = list(G.edges())
        trueEdges = dict()
        falseEdges = dict()
        reachable1 = dict()
        unreachable1 = dict()

        vDictUnreachable = dict()
        vDictReachable = dict()

        apsp = dict(nx.all_pairs_shortest_path_length(G))
        
        Ambiguities = []
        
        for cutoff in range(1, AMBIGUITY_SP_CUTOFF + 1):
            edgeSet = edgeSetOrg.copy()
            edgeEdgeSet = edgeEdgeSetOrg.copy()
        
            for n in G.nodes():
                vDictReachable[n] = set()
                vDictUnreachable[n] = set()

            '''
            Remove edges that are not ambiguous as the edge exists 
            '''
            for s1,t1,data in G.edges(data=True):
                toRemove = []

                trueEdges[(s1,t1)] = set()
                falseEdges[(s1,t1)] = set()
                for e in edgeEdgeSet[(s1,t1)]:
                    s2,t2 = edges[e]

                    if G.has_edge(s1,t2) and G.has_edge(t1,s2):
                        toRemove.append(e)
                        trueEdges[(s1,t1)].add(e)
                    elif G.has_edge(t1,t2) and G.has_edge(s1,s2):
                        toRemove.append(e)
                        trueEdges[(s1,t1)].add(e)
                    else:
                        falseEdges[(s1,t1)].add(e)

                trueEdges[(s1,t1)].add(-1)
                edgeEdgeSet[(s1,t1)].difference_update(toRemove)

                reachable1[(s1,t1)] = set()
                unreachable1[(s1,t1)] = set()
                for v in edgeSet[(s1,t1)]:
                    if v is s1:
                        continue
                    elif v in apsp[s1] and apsp[s1][v] <= cutoff:
                        reachable1[(s1,t1)].add(v)
                        vDictReachable[s1].add(v)
                    else:
                        unreachable1[(s1,t1)].add(v)
                        vDictUnreachable[s1].add(v)

                reachable1[(t1,s1)] = set()
                unreachable1[(t1,s1)] = set()
                for v in edgeSet[(t1,s1)]:
                    if v is t1:
                        continue
                    elif v in apsp[t1] and apsp[t1][v] <= cutoff:
                        reachable1[(t1,s1)].add(v)
                        vDictReachable[t1].add(v)
                    else:
                        unreachable1[(t1,s1)].add(v)
                        vDictUnreachable[t1].add(v)

                vDictReachable[t1].add(s1)
                vDictReachable[s1].add(t1)
                reachable1[(s1,t1)].add(t1)
                reachable1[(t1,s1)].add(s1)

            oben = 0
            unten = 0

            ambVAmbiguity = []
            for n in G.nodes():
                oben += len(vDictUnreachable[n])
                unten += len(vDictUnreachable[n]) + len(vDictReachable[n])

                lie1 = len(vDictUnreachable[n]) / (len(vDictUnreachable[n]) + len(vDictReachable[n]))

                ambVAmbiguity.append(lie1)

            Ambiguities.append(np.mean(ambVAmbiguity))
        
        return Ambiguities

    @staticmethod
    def preprocessAmbiguity(G, outPath, name):
        '''
        Compute which edges in the drawing are close and parallel or have a shallow crossing.
        '''

        G = Metrics.copyAndScale(G, SCALEFACTOR)

        w1 = int(np.ceil(G.graph['xmax']))
        h1 = int(np.ceil(G.graph['ymax']))

        occup = np.zeros((w1 + 1,h1 + 1,len(G.edges())), dtype=np.int32)
        angle = np.zeros((w1 + 1,h1 + 1,len(G.edges())), dtype=np.float64)
        pixelpathLength = np.zeros(len(G.edges()))
        #print(occup.shape)

        z = 0
        for source, target, data in G.edges(data=True):
            X = data['X']
            Y = data['Y']

            path = []
        
            for i in range(len(X) - 1):
                x0 = X[i]
                x1 = X[i + 1]
                y0 = Y[i]
                y1 = Y[i + 1]

                l = int(np.ceil(np.sqrt(np.square(x0-x1) + np.square(y0-y1)) / 0.4))

                a = np.arctan2(y0 - y1, x0 - x1)
                
                if a < 0:
                    a = 2 * np.pi + a                   
    
                x = np.linspace(x0, x1, num=l)
                y = np.linspace(y0, y1, num=l)

                pX = np.empty_like(x)
                pY = np.empty_like(y)

                np.floor(x, pX)
                np.floor(y, pY)
                pX = pX.astype(int)
                pY = pY.astype(int)

                xOld = -1
                yOld = -1

                for j in range(l):
                    try:
                        if not (pX[j] == xOld and pY[j] == yOld ):
                            path.append((pX[j],pY[j]))
                            xOld = pX[j]
                            yOld = pY[j]

                            angle[pX[j]][pY[j]][z] = a
                            occup[pX[j]][pY[j]][z] = 1
                    except IndexError:
                        continue

            #print(len(path),path)
            data['Path'] = path
            pixelpathLength[z] = len(path) - 2
            z = z + 1

        kernel = np.ones((3,3))

        conv = occup
        aangle = angle

        pixelDict = dict()

        for z in range(len(occup[0][0])):
            arr = occup[:,:,z]
            arr = signal.convolve2d(arr, kernel, boundary='fill', mode='same')
            
            arr2 = angle[:,:,z]
            arr2 = signal.convolve2d(arr2, kernel,boundary='fill', mode='same')

            arr2[arr > 1] = arr2[arr > 1] / arr[arr > 1]        
            arr[arr > 1] = 1

            conv[:, :, z] = arr
            aangle[:,:,z] = arr2

        print('done Convolution')

        for id, data in G.nodes(data=True):
            x = int(data['X'])
            y = int(data['Y'])

            for xx in [-1,0,1]:
                for yy in [-1,0,1]:
                    try:
                        conv[x + xx, y + yy,:] = 0
                        aangle[x + xx, y + yy,:] = 0.0
                    except IndexError:
                        continue

        print('done removing vertices')

        for x in range(w1):
            for y in range(h1):
                l = []

                for z in range(len(occup[0][0])):
                    if conv[x][y][z]:
                        l.append(z)

                pixelDict[(x,y)] = l

        #amb = np.zeros((w1,h1))
        amb = np.zeros((w1,h1, len(G.edges())))

        print('done building structure')
        edgeAmbSet = dict()
        edgeEdgeAmbSet = dict()
        edges = list(G.edges())
        
        for s,t in G.edges():
            edgeEdgeAmbSet[(s,t)] = set()
            edgeAmbSet[(s,t)] = set()
            edgeAmbSet[(t,s)] = set()
            edgeAmbSet[(s,t)].add(t)
            edgeAmbSet[(t,s)].add(s)

        ambVal = np.zeros((len(G.edges()), len(G.edges())))
        ambEdge = np.zeros(len(G.edges()))

        for x in range(w1):
            for y in range(h1):
                lll = pixelDict[(x,y)]

                for i in range(len(lll)):
                    for j in range(i + 1, len(lll)):
                        e1 = lll[i]
                        e2 = lll[j]

                        a1 = aangle[x,y,e1]
                        a2 = aangle[x,y,e2]

                        a = np.mod(a1 - a2, 2 * np.pi)
                        aa = np.mod(a2 - a1, 2 * np.pi)
                        a = np.min([a, aa])
                        aTest = np.min([a, np.pi - a])

                        if aTest < ANGLE_THRESHOLD:
                            amb[x,y,z] += 1


                            ambEdge[e1] += 1
                            ambEdge[e2] += 1

                            ambVal[e1,e2] += 1
                            ambVal[e2,e1] += 1

                            s1,t1 = edges[e1]
                            s2,t2 = edges[e2]

                            edgeEdgeAmbSet[(s1,t1)].add(e2)
                            edgeEdgeAmbSet[(s2,t2)].add(e1)

                            if a < np.pi - a:
                                edgeAmbSet[(s1, t1)].add(t2)
                                edgeAmbSet[(t1, s1)].add(s2)
                                edgeAmbSet[(t2, s2)].add(s1)
                                edgeAmbSet[(s2, t2)].add(t1)
                            else:
                                edgeAmbSet[(s1, t1)].add(s2)
                                edgeAmbSet[(t1, s1)].add(t2)
                                edgeAmbSet[(t2, s2)].add(t1)
                                edgeAmbSet[(s2, t2)].add(s1)


        print('done calculating')
    
        out = f'{outPath}/pickle/'
        if not os.path.exists(out):
            os.makedirs(out)

        pickle.dump(edgeEdgeAmbSet, open(f'{out}{name}.ees', 'wb'))
        pickle.dump(edgeAmbSet, open(f'{out}{name}.es', 'wb'))

        return edgeAmbSet, edgeEdgeAmbSet

    @staticmethod
    def copyAndScale(G, factor):
        '''
        Copy the graph and scale the position by factor. requires a graph with positive positions.

        G - Graph
        factor - scaling factor

        return
        G - Scaled and copied graph
        '''
        G = copy.deepcopy(G)

        for n, data in G.nodes(data=True):
            data['X'] = data['X'] * factor
            data['Y'] = data['Y'] * factor

        for t,s,data in G.edges(data=True):
            X = []
            Y = []

            for x in data['X']:
                X.append(x * factor)

            for y in data['Y']:
                Y.append(y * factor)

            data['X'] = X
            data['Y'] = Y
    
        G.graph['xmax'] = G.graph['xmax'] * factor
        G.graph['ymax'] = G.graph['ymax'] * factor

        return G
