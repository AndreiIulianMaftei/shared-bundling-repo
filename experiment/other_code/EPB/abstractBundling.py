import numpy as np
from nx2ipe.nx2ipe import IpeConverter, SplineC
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import scipy.interpolate as si
import cmcrameri as cmc
import networkx as nx
from abc import ABC, abstractmethod
import sys

### Plotting Parameters for printing ###
# LINEWIDTH = 2.0
# LINE_COLOR = 'darkgrey'
# ALPHA = 0.4
# CIRCLE = 6
# CIRCLE_COLOR = 'firebrick'
# CIRCLE_SMALL = 2.0
# CIRCLE_COLOR_LIGHT = 'darkslategrey'
# BACKGROUND_COLOR = 'white'
# DPI = 600
# NUM_POINTS_BEZIER = 100
# GWIDTH = 19200

### Plotting Parameters for ambiguity tests
LINEWIDTH = 0.5
LINE_COLOR = '#393433'
ALPHA = 0.5
CIRCLE = 10.0
CIRCLE_COLOR = 'firebrick'
CIRCLE_SMALL = 1.0
CIRCLE_COLOR_LIGHT = '#181716'
BACKGROUND_COLOR = 'white'
DPI = 192
NUM_POINTS_BEZIER = 50
GWIDTH = 1600
#GWIDTH = 6921

class AbstractBundling:
    '''
    Base class for implemented bundling algorithms. Handles drawing.
    '''
    def __init__(self, G : nx.Graph):
        self.G = G.copy()
        self.name = 'abstract'
        for (u, v, data) in G.edges(data=True):
            data['cp'] = []



    @abstractmethod
    def bundle(self):
        raise NotImplemented

    def getPosDict(self):
        pos = {}

        for id, data in self.G.nodes().data():
            pos[id] = (data['X'], data['Y'])

        return pos

    def store(self, path):
        GG = nx.Graph()
        GG.add_edges_from(self.G.edges())
        GG.add_nodes_from(self.G.nodes())

        self.approximateCurve()

        for u,v, data in self.G.edges(data=True):
            if 'X_CP' in data:
                GG[u][v]['Spline_X'] = ' '.join(map(str, data['X_CP']))
                GG[u][v]['Spline_Y'] = ' '.join(map(str, data['Y_CP']))
                
        for u, data in self.G.nodes(data=True):
            GG.nodes[u]['X'] = data['X']
            GG.nodes[u]['Y'] = data['Y']

        nx.write_graphml(GG, path)


    def setPosDict(self, pos):
        for id, data in self.G.nodes().data():
            data['X'], data['Y'] = pos[id]
            data['x'], data['y'] = pos[id]

    def scaleToWidth(self, G_width):
        xmin = sys.maxsize
        xmax = -sys.maxsize - 1
        ymin = sys.maxsize
        ymax = -sys.maxsize - 1

        for n, data in self.G.nodes(data = True):
            if not ('X' in data and 'Y' in data):
                print(f'Coordinate error in dataset {input}')
                return None

            else:
                x = data['X']
                y = data['Y']

                if x < xmin: xmin = x
                if y < ymin: ymin = y
                if x > xmax: xmax = x
                if y > ymax: ymax = y
            
        factor = G_width / (xmax - xmin)
        # width = G_width
        # height = (ymax - ymin) * factor

        for node, data in self.G.nodes().data():
            self.G.nodes()[node]['X'] = (data['X'] - xmin) * factor
            data['Y'] = (data['Y'] - ymin) * factor

        for (u,v,data) in self.G.edges(data = True):
            d1 = self.G.nodes[u]
            d2 = self.G.nodes[v]

            dx = d1['X'] - d2['X']
            dy = d1['Y'] - d2['Y']

            l = np.sqrt((dx)**2 + (dy)**2)

            data['dist'] = l

        self.G.graph['xmin'] = 0
        self.G.graph['xmax'] = G_width
        self.G.graph['ymin'] = 0
        self.G.graph['ymax'] = (ymax - ymin) * factor

    def approximateCurve(self):
        for source, target, data in self.G.edges(data=True):
            if 'Spline' in data:
                points = [(self.G.nodes[source]['X'], self.G.nodes[source]['Y'])] + data['Spline'].points + [(self.G.nodes[target]['X'], self.G.nodes[target]['Y'])]

                # if abs(self.G.nodes[source]['X'] - points[1][0]) > 50 and abs(self.G.nodes[source]['Y'] - points[1][1]) > 50:
                #     print()

                X, Y = self.approxBezier(points, NUM_POINTS_BEZIER)
            else:
                X = [self.G.nodes[source]['X'], self.G.nodes[target]['X']]
                Y = [self.G.nodes[source]['Y'], self.G.nodes[target]['Y']]         

            data['X_CP'] = X
            data['Y_CP'] = Y

    def colorEdges(self):
        '''
        Calculate the edge angle and color edges accordingly.
        '''
        for source, target, data in self.G.edges(data=True):
            
            x0 = self.G.nodes[source]['X']
            y0 = self.G.nodes[source]['Y']
            x1 = self.G.nodes[target]['X']
            y1 = self.G.nodes[target]['Y']

            x = x1 - x0
            y = y1 - y0

            angle = np.arctan2(y, x) * 180 / np.pi

            
            if angle < 0:
                angle = 360 + angle

            if self.G.is_directed():
                a = angle / 360
            else:            
                if angle > 180:
                    angle = (angle + 180) % 360
                a = angle / 180
            
            a += 0.75
            if a > 1:
                a = a - 1

            cmap = cmc.cm.romaO
            color = cmap(a)
            data['Angle'] = a
            data['Stroke'] = f"{color[0]} {color[1]} {color[2]}"

        print('done')

    def drawTrl(self, path):
        '''
        Create a file with trails of edges.
        '''
        with open(f'{path}{self.name}.trl', 'w') as f:
        
            i = 1
            for source, target, data in self.G.edges().data():
                if 'Spline' in data:
                    points = [(self.G.nodes[source]['X'], self.G.nodes[source]['Y'])] + data['Spline'].points + [(self.G.nodes[target]['X'], self.G.nodes[target]['Y'])]

                    X, Y = self.approxBezier(points, NUM_POINTS_BEZIER)
                else:
                    X = [self.G.nodes[source]['X'], self.G.nodes[target]['X']]
                    Y = [self.G.nodes[source]['Y'], self.G.nodes[target]['Y']]        
                    
                f.write(f"{i}: ")
                
                for i, x in enumerate(X):
                    y = Y[i]
                    f.write(f"{x:.2f} {y:.2f} ")
                f.write("\n")

    def draw(self, path, color=True, plotIpe=False, plotSpanner=False, plotSubgraph=None, fileAddition="", color_vertices=None):
        '''
        Draw the bundling. Either using the assign color function or the coloring given by the bundling. if plotIpe is true, it will create an IPE drawing as well.
        '''
        nx.set_edge_attributes(self.G, '50%', name='Opacity')

        if color:
            self.colorEdges()

        print(len(self.G.edges()))

        if plotIpe:
            ipe = IpeConverter()
            ipe._options._DRAWING_UNBOUND = False
            ipe.createDrawing(self.G, f'{path}{self.name}.xml')

        fig, ax = plt.subplots(figsize=(self.G.graph['xmax'] / DPI, self.G.graph['ymax'] / DPI), dpi=DPI)
        ax.axis('off')
        #fig.canvas.set_window_title(self.name)

        cmap = cmc.cm.romaO

        if plotSubgraph:
            for source, target, data in plotSubgraph.edges().data():
                X = [self.G.nodes[source]['X'], self.G.nodes[target]['X']]
                Y = [self.G.nodes[source]['Y'], self.G.nodes[target]['Y']]

                ax.plot(X, Y, color='red', alpha=ALPHA, lw = LINEWIDTH)

        for source, target, data in self.G.edges(data = True):
            

            if 'Xapprox' in data and 'Yapprox' in data:
                X = data['Xapprox']
                Y = data['Yapprox']
            elif 'Spline' in data:
                points = [(self.G.nodes[source]['X'], self.G.nodes[source]['Y'])] + data['Spline'].points + [(self.G.nodes[target]['X'], self.G.nodes[target]['Y'])]
                X, Y = self.approxBezier(points, 50)
            else:
                X = [self.G.nodes[source]['X'], self.G.nodes[target]['X']]
                Y = [self.G.nodes[source]['Y'], self.G.nodes[target]['Y']]

            if color:
                ax.plot(X, Y, color=cmap(data['Angle']), alpha=ALPHA, lw = LINEWIDTH)
            else:
                ax.plot(X, Y, color=LINE_COLOR, alpha=ALPHA, lw = LINEWIDTH)



        X = []
        Y = []
        C = []
        for id, data in self.G.nodes().data():
            
            X.append(data['X'])
            Y.append(data['Y'])
            
            if color_vertices:
                cmap = matplotlib.cm.get_cmap('tab10')
                
                if data[color_vertices] == 100:
                    C.append("red")
                else:
                    C.append(cmap(data[color_vertices]))
            else:
                C.append(CIRCLE_COLOR_LIGHT)

        ax.scatter(X, Y, color=C, marker='.', s = CIRCLE, zorder=2)


        plt.savefig(f'{path}{self.name}{fileAddition}.png')
        plt.close(fig)
        
        if plotSpanner:
            
            fig, ax = plt.subplots(figsize=(self.G.graph['xmax'] / DPI, self.G.graph['ymax'] / DPI), dpi=DPI)
            ax.axis('off')
            fig.canvas.set_window_title(self.name)
            
            for source, target, data in self.G.edges().data():
                if 'Layer' in data and data['Layer'] == 'Spanning':
                    X = [self.G.nodes[source]['X'], self.G.nodes[target]['X']]
                    Y = [self.G.nodes[source]['Y'], self.G.nodes[target]['Y']]

                    data['X'] = X
                    data['Y'] = Y
                    ax.plot(X, Y, color='#853D36', alpha=1, lw = 2*LINEWIDTH)

            X = []
            Y = []
            for id, data in self.G.nodes().data():
                X.append(data['X'])
                Y.append(data['Y'])

            ax.plot(X, Y, linestyle="none", color=CIRCLE_COLOR_LIGHT, marker='.', markersize = CIRCLE)

            plt.savefig(f'{path}{self.name}_{fileAddition}_spanner.png')
            plt.close(fig)

    def approxBezier(self, points, n):
        X = []
        Y = []
        binom = {}

        for i, p in enumerate(points):
            binom[i] = self.binomial(len(points) - 1, i)
            i += 1

        for t in np.linspace(0, 1, n):
            pX = 0
            pY = 0


            for i, p in enumerate(points):
                tpi = np.power(1 - t, len(points) - 1 - i)
                coeff = tpi * np.power(t, i)

                pX += binom[i] * coeff * p[0]
                pY += binom[i] * coeff * p[1]

            X.append(pX)
            Y.append(pY)
        
        return X, Y


    def binomial(self, n, k):
        coeff = 1
        x = n - k + 1
        while x <= n:
            coeff *=x
            x += 1
        
        for x in range(1, k + 1): coeff /= x
        return coeff

class GraphLoader(AbstractBundling):
    def __init__(self, G):
        self.G = G
        self.reverse = False
        self.invertY = False
        self.invertX = False
        self.is_graphml = False
            
    @property
    def name(self):
        return f'{self.filename}'
    
    def bundle(self):
        if self.is_graphml:
            self.G = self.readData_graphml(self.filepath, self.filename, self.invertY, self.invertX, self.reverse)
        else:
            self.G = self.readData(self.filepath, self.filename, self.invertY, self.invertX, self.reverse)
        return 0
    
    def readData_graphml(self, path, file, invertY, invertX, reverse):
        '''
        Read the graph in our custom file format from path/file. Expects .nodes and .edges file.

        path - Path of the file
        file - Name of the file. .edges or .nodes suffix is automatically added
        invertY - Invert the y-axis
        invertX - Invert the x-axis

        return
        G - The loaded input graph
        '''

        to_read = path + "/" + file + '.graphml'

        G = nx.read_graphml(to_read)      

        xmin = sys.maxsize
        xmax = -sys.maxsize - 1
        ymin = sys.maxsize
        ymax = -sys.maxsize - 1

        for u, data in G.nodes(data=True):
            if 'X' in data and 'Y' in data:
                data['X'] = float(data['X'])
                data['Y'] = float(data['Y'])
            else:
                continue

            xmin = min(xmin, data['X'])
            xmax = max(xmax, data['X'])
            ymin = min(ymin, data['Y'])
            ymax = max(ymax, data['Y'])

        ## TODO fix the factor
        factor = GWIDTH / (xmax - xmin)
        width = GWIDTH
        height = (ymax - ymin) * factor

        G.graph['xmin'] = 0
        G.graph['xmax'] = GWIDTH
        G.graph['ymin'] = 0
        G.graph['ymax'] = height

        for node, data in G.nodes().data():
            
            G.nodes()[node]['X'] = (data['X'] - xmin) * factor
            data['Y'] = (data['Y'] - ymin) * factor

            if invertX: G.nodes()[node]['X'] = width - data['X']
            if invertY: data['Y'] = height - data['Y']

        for u,v,data in G.edges(data=True):
            

            if "Spline_X" in data:
                X = [(float(x) - xmin) * factor for x in data['Spline_X'].split(" ")]
                Y = [(float(x) - ymin) * factor for x in data['Spline_Y'].split(" ")]

                if abs(G.nodes[u]['X'] - X[0]) > 0.001:
                    X.reverse()

                if abs(G.nodes[u]['Y'] - Y[0]) > 0.001:
                    Y.reverse()

                data["Xapprox"] = X
                data["Yapprox"] = Y
                data['X'] = data['Xapprox']
                data['Y'] = data['Yapprox']
            else:
                data['X'] = []
                data['Y'] = []


            x1 = G.nodes[u]['X']
            y1 = G.nodes[u]['Y']

            x2 = G.nodes[v]['X']
            y2 = G.nodes[v]['Y']

            data['weight'] = np.sqrt((x1-x2)**2 + (y1 - y2)**2)

        return G

    def readData(self, path, file, invertY, invertX, reverse):
        '''
        Read the graph in our custom file format from path/file. Expects .nodes and .edges file.

        path - Path of the file
        file - Name of the file. .edges or .nodes suffix is automatically added
        invertY - Invert the y-axis
        invertX - Invert the x-axis

        return
        G - The loaded input graph
        '''

        nFile = path + file + '.nodes'
        eFile = path + file + '.edges'

        if self.G.is_directed():
            G = nx.DiGraph()
        else:       
            G = nx.Graph()
        

        G.graph['title'] = file

        xmin = sys.maxsize
        xmax = -sys.maxsize - 1
        ymin = sys.maxsize
        ymax = -sys.maxsize - 1

        with open(nFile, "r") as f:

            lines = f.readlines()

            for line in lines:
                l = line.split(' ')

                id = int(l[0])
                x = float(l[1])
                y = float(l[2])

                if x < xmin: xmin = x
                if y < ymin: ymin = y
                if x > xmax: xmax = x
                if y > ymax: ymax = y

                G.add_node(id, X = x, Y = y)    

        factor = GWIDTH / (xmax - xmin)
        width = GWIDTH
        height = (ymax - ymin) * factor

        for node, data in G.nodes().data():
            G.nodes()[node]['X'] = (data['X'] - xmin) * factor
            data['Y'] = (data['Y'] - ymin) * factor

            if invertX: G.nodes()[node]['X'] = width - data['X']
            if invertY: data['Y'] = height - data['Y']

        G.graph['xmin'] = 0
        G.graph['xmax'] = GWIDTH
        G.graph['ymin'] = 0
        G.graph['ymax'] = (ymax - ymin) * factor

        with open(eFile, 'r') as f:
            
            lines = f.readlines()

            for line in lines:
                l = line.split(' ')

                source = int(l[0])
                target = int(l[1])

                X = []
                Y = []
                

                for c, val in enumerate(l[2:]):
                    if c % 2 == 0:
                        pos = (float(val) - xmin) * factor
                        if invertX: pos = width - pos
                        X.append(pos)
                    else:
                        pos = (float(val) - ymin) * factor
                        if invertY: pos = height- pos
                        Y.append( pos)

                #print(X[0], G.nodes[source]['x'], Y[0], G.nodes[source]['y'])
                if reverse == True:
                    X.reverse()
                    Y.reverse()
                    G.add_edge(target, source, Xapprox = X, Yapprox = Y, weight=1)
                else:
                    G.add_edge(source, target, Xapprox = X, Yapprox = Y, weight=1)

        for u,v,data in G.edges(data=True):
            x1 = G.nodes[u]['X']
            y1 = G.nodes[u]['Y']
            x2 = G.nodes[v]['X']
            y2 = G.nodes[v]['Y']

            data['weight'] = np.sqrt((x1-x2)**2 + (y1 - y2)**2)

        return G