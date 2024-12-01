import copy
import os
import networkx as nx
import numpy as np
from scipy import signal
from PIL import Image
import pickle
import seaborn as sbn
import matplotlib.pyplot as plt
import matplotlib
from frechetdist import frdist
import re
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from pdf2image import convert_from_path
import requests
from wand.image import Image

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
        imGray = np.array(Image.open(imPath).convert('L'))

        allPixels = imGray.size

        greyscalepixel = (imGray < GREY_THRESH).sum()

        return greyscalepixel, allPixels

    def calcInkRatio(self, path):
        '''
        Calculate the ink ratio. Assumes that there is a .png with the investigated algorithm and a straight line drawing.
        '''

        imPath = path + 'straight' + '.png'
        imGray = np.array(Image.open(imPath).convert('L'))

        inkratioG = (imGray < GREY_THRESH).sum()

        imPath = path + self.name + '.png'
        imGray = np.array(Image.open(imPath).convert('L'))

        greyscalepixel = (imGray < GREY_THRESH).sum()
        inkratio = greyscalepixel / inkratioG

        return inkratio
    def calcMonotonicity(self, algorithm):

        monotonicitys = []
        polylines = []
        x = 0
        if(algorithm != "wr"):
            
            list_edges = list(self.G.edges(data = True))

            for index, (u,v,data) in enumerate(list_edges):
                
                if(index == 500):
                    break

                Y = []
                X = []

                X = [float(num) for num in data.get('X')]
            
                Y = [float(num) for num in data.get('Y')]
                if len(X) < 3:
                    # With less than 3 points, there are no turns
                    return 0, []

                direction_changes = []
                monotonicity = 0
                previous_sign = None

                for i in range(len(X) - 2):
                    # Points
                    x0, y0 = X[i], Y[i]
                    x1, y1 = X[i+1], Y[i+1]
                    x2, y2 = X[i+2], Y[i+2]

                    # Vectors
                    v1_x = x1 - x0
                    v1_y = y1 - y0
                    v2_x = x2 - x1
                    v2_y = y2 - y1

                    # Cross product
                    cross = v1_x * v2_y - v1_y * v2_x

                    # Determine the sign
                    if cross > 0:
                        current_sign = 1  # Left turn
                    elif cross < 0:
                        current_sign = -1  # Right turn
                    else:
                        current_sign = 0  # Straight line or colinear

                    # Check if the direction has changed (excluding zero crossings)
                    if previous_sign is not None and current_sign != 0:
                        if current_sign != previous_sign:
                            monotonicity += 1
                            direction_changes.append(i+1)  # Index where the change occurs

                    # Update previous_sign if current_sign is non-zero
                    if current_sign != 0:
                        previous_sign = current_sign
                    
                monotonicitys.append(monotonicity)
                x0 = self.G.nodes[u]['X']
                y0 = self.G.nodes[u]['Y']

                x1 = self.G.nodes[v]['X']
                y1 = self.G.nodes[v]['Y']

            return monotonicitys
                
        else:
            list_edges = list(self.G.edges(data = True))
            G = nx.Graph()
            G = self.G

            for index, (u,v,data) in enumerate(list_edges):
                
                if(index == 500):
                    break
                
                #x_spline_value = G[data.get('X')][data.get('Y')]['X_Spline']
                
                Y = []
                X = []

                X = re.findall(r'-?\d+\.\d+', G[list_edges[index][0]][list_edges[index][1]]["X_Spline"])
                X = [float(num) for num in X]
                Y = re.findall(r'-?\d+\.\d+', G[list_edges[index][0]][list_edges[index][1]]["Y_Spline"])
                Y = [float(num) for num in Y]
                if len(X) < 3:
                    # With less than 3 points, there are no turns
                    return 0, []

                direction_changes = []
                monotonicity = 0
                previous_sign = None

                for i in range(len(X) - 2):
                    # Points
                    x0, y0 = X[i], Y[i]
                    x1, y1 = X[i+1], Y[i+1]
                    x2, y2 = X[i+2], Y[i+2]

                    # Vectors
                    v1_x = x1 - x0
                    v1_y = y1 - y0
                    v2_x = x2 - x1
                    v2_y = y2 - y1

                    # Cross product
                    cross = v1_x * v2_y - v1_y * v2_x

                    # Determine the sign
                    if cross > 0:
                        current_sign = 1  # Left turn
                    elif cross < 0:
                        current_sign = -1  # Right turn
                    else:
                        current_sign = 0  # Straight line or colinear

                    # Check if the direction has changed (excluding zero crossings)
                    if previous_sign is not None and current_sign != 0:
                        if current_sign != previous_sign:
                            monotonicity += 1
                            direction_changes.append(i+1)  # Index where the change occurs

                    # Update previous_sign if current_sign is non-zero
                    if current_sign != 0:
                        previous_sign = current_sign
                    
                monotonicitys.append(monotonicity)
                x0 = self.G.nodes[u]['X']
                y0 = self.G.nodes[u]['Y']

                x1 = self.G.nodes[v]['X']
                y1 = self.G.nodes[v]['Y']

            return monotonicitys
    def plotMonotonicity(self, rezults):
        Max = -999999999999
        Min = 999999999999
        mean = 0
        number = 0
        print(mean)
        for rez in rezults: 
            if(rez[1] == "epb"):
                Max = Max
                Min = Min
                continue
            Max = max(np.max(rez[0]), Max)
            Min = min(np.min(rez[0]), Min)
            mean += np.sum(rez[0])
            number = number+ rez[0].__len__()
        mean = mean/number
            
        print (rezults.__len__())
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        list_normalised_min_max = [None]*rezults.__len__()
        list_normalised_z_score = [None]*rezults.__len__()
        list = [None]*rezults.__len__()
        for index, rez in enumerate(rezults):
            if(rez[1] == "epb"):
                list_normalised_min_max[index] = []
                continue
            #normalise rez[0] (a list ) using min/max
            list[index] = rez[0]
            #print(rez[0])
            list_normalised_min_max[index] = [(x - Min) / (Max - Min) for x in [list[index]]]
            #list_normalised_z_score[index] = [(x - mean) / np.std(list[index]) for x in list[index]]
        #print single histogram
        png_file_min_max = []
        png_file_z_score = []
        for index, rez in enumerate(rezults):
            plt.figure()
            plt.hist(list_normalised_min_max[index], bins=100, alpha=0.5, label=f'Algorithm {rez[1]}')
            plt.legend(loc='upper right')
            plt.xlabel('Frechet Distance')
            plt.ylabel('Number of Edges')
            plt.savefig(f'monotonicity_normalisation_{rez[1]}.png')
            png_file_min_max.append(f'min-max normalisation {rez[1]}.png')
            plt.figure()
        return
                
                
        
    def calcFrechet(self, algorithm):

        ma = -999999999999
        mi = 999999999999
        frechet = []
        polylines = []
        x= 0
        
        #print(G.edges(data = True))
        
        if(algorithm != "wr"):
            
            list_edges = list(self.G.edges(data = True))

            for index, (u,v,data) in enumerate(list_edges):
                
                if(index == 500):
                    break

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
        else:
            list_edges = list(self.G.edges(data = True))
            G = nx.Graph()
            G = self.G

            for index, (u,v,data) in enumerate(list_edges):
                
                if(index == 500):
                    break
                
                #x_spline_value = G[data.get('X')][data.get('Y')]['X_Spline']
                
                numbers_y = []
                numbers_x = []

                numbers_x = re.findall(r'-?\d+\.\d+', G[list_edges[index][0]][list_edges[index][1]]["X_Spline"])
                numbers_x = [float(num) for num in numbers_x]
                numbers_y = re.findall(r'-?\d+\.\d+', G[list_edges[index][0]][list_edges[index][1]]["Y_Spline"])
                numbers_y = [float(num) for num in numbers_y]
                
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
                    continue
                
                
                x = frdist(interpolated_points, polyline) 
                
                frechet.append(x)
                ma = max(ma, x)
                mi = min(mi, x)

                #polylines.append(polyline)

        
        return frechet
    
    def plotFrechet(self, rezults):
        '''
        Plot the normalised Frechet distance of two algorithms. a histogram, withn x axis as the frechet distance and y axis as the number of edges .
        '''
        Max = -999999999999
        Min = 999999999999
        mean = 0
        number = 0
        print(mean)
        for rez in rezults: 
            Max = max(np.max(rez[0]), Max)
            Min = min(np.min(rez[0]), Min)
            mean += np.sum(rez[0])
            number = number+ rez[0].__len__()
        mean = mean/number
            
        print (rezults.__len__())
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        list_normalised_min_max = [None]*rezults.__len__()
        list_normalised_z_score = [None]*rezults.__len__()
        list = [None]*rezults.__len__()
        for index, rez in enumerate(rezults):
            list[index] = rez[0]
            
            list_normalised_min_max[index] = [(x - Min) / (Max - Min) for x in [list[index]]]
            
        png_file_min_max = []
        png_file_z_score = []
        for index, rez in enumerate(rezults):
            plt.figure()
            plt.hist(list_normalised_min_max[index], bins=100, alpha=0.5, label=f'Algorithm {rez[1]}')
            plt.legend(loc='upper right')
            plt.xlabel('Frechet Distance')
            plt.ylabel('Number of Edges')
            plt.savefig(f'frechet_normalisation_{rez[1]}.png')
            png_file_min_max.append(f'frechet_normalisation_{rez[1]}.png')
            plt.figure()
            
        #print cumulative min-max histogram
        #base_image = Image.open(png_file_min_max[0]).convert("RGB")

        # Open the rest of the images and convert them to RGB
        #other_images = [Image.open(png).convert("RGB") for png in png_file_min_max[1:]]

        # Save all images as a PDF
        #output_pdf_path = "output_min_max.pdf"
        #base_image.save(output_pdf_path, save_all=True, append_images=other_images)

        #print(f"PDF created successfully: {output_pdf_path}")
        plt.figure()
        for index, rez in enumerate(rezults):
            plt.hist(list_normalised_min_max[index], bins=100, alpha=0.5, label=f'Algorithm {rez[1]}')
            plt.legend(loc='upper right')
            plt.xlabel('Frechet Distance')
            plt.ylabel('Number of Edges')    
        
        plt.savefig(f'min-max normalisation cumulative.png')

    def plotMegaGraph(self,algorithms ,metric, histogram_image_paths):
        def embed_image(ax, image_path, title):
            try:
                img = plt.imread(image_path)
                ax.imshow(img)
                ax.axis('off')  
                ax.set_title(title, fontsize=10)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                ax.text(0.5, 0.5, "Error loading image", ha="center", va="center", fontsize=12)
                ax.axis('off')

        pdf_path = f"output_comparison_{metric}.pdf"
        with PdfPages(pdf_path) as pdf:
            # Create a figure with the desired layout
            fig = plt.figure(figsize=(12, 8))

            # Top row: Graphs
            for i, graph_path in enumerate(algorithms):
                ax = plt.subplot2grid((2, 3), (0, i))  # Top row, 3 columns
                embed_image(ax, f'output/airlines/images/{graph_path}.png', f"Graph {algorithms[i]}")

            # Bottom row: Histograms
            for i, hist_path in enumerate(histogram_image_paths):
                ax = plt.subplot2grid((2, 3), (1, i))  # Bottom row, 3 columns
                embed_image(ax, hist_path, f" {metric}")

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
                # Create a figure with the desired layout
                fig = plt.figure(figsize=(8, 10))

                main_graph_ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
                main_graph_img = plt.imread(main_graph_image_path)
                main_graph_ax.imshow(main_graph_img)
                main_graph_ax.axis('off')  # Turn off the axis for the image
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
    def plotDistortionHistogram(self, rezults):
        
        Max = -999999999999
        Min = 999999999999
        mean = 0
        number = 0
        print(mean)
        for rez in rezults: 
            Max = max(np.max(rez[0]), Max)
            Min = min(np.min(rez[0]), Min)
            mean += np.sum(rez[0])
            number = number+ rez[0].__len__()
        mean = mean/number
            
        print (rezults.__len__())
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        list_normalised_min_max = [None]*rezults.__len__()
        list_normalised_z_score = [None]*rezults.__len__()
        list = [None]*rezults.__len__()
        for index, rez in enumerate(rezults):
            #normalise rez[0] (a list ) using min/max
            list[index] = rez[0]
            #print(rez[0])
            list_normalised_min_max[index] = [(x - Min) / (Max - Min) for x in [list[index]]]
            #list_normalised_z_score[index] = [(x - mean) / np.std(list[index]) for x in list[index]]
        #print single histogram
        png_file_min_max = []
        png_file_z_score = []
        for index, rez in enumerate(rezults):
            plt.figure()
            plt.hist(list_normalised_min_max[index], bins=100, alpha=0.5, label=f'Algorithm {rez[1]}')
            plt.legend(loc='upper right')
            plt.xlabel('Frechet Distance')
            plt.ylabel('Number of Edges')
            plt.savefig(f'distortion_normalisation_{rez[1]}.png')
            png_file_min_max.append(f'min-max normalisation {rez[1]}.png')
            plt.figure()
            

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
