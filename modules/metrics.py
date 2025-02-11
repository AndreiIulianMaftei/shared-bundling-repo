import os
import networkx as nx
import numpy as np
from modules.abstractBundling import RealizedBundling
from modules.EPB.straight import StraightLine

class Metrics():
    implemented_metrics = ["distortion", "inkratio"]
    def __init__(self,bundle:RealizedBundling):
        """
        G should be a RealizedBundling instance. 
        All nodes should have 'X' and 'Y' coordinates defined. 
        All edges should have "Spline_X" and "Spline_Y" attributes defined. 
        """
        self.Bundle = bundle
        self.G      = bundle.G

        #Setup things we need for metrics (but we'll compute them on demand)
        self.bundleDrawing   = None
        self.straightGraph   = None
        self.straightDrawing = None

        #Let's label all the edges 1,..,m in case we need this
        for i, (u,v) in enumerate(self.G.edges()):
            self.G[u][v]['id'] = i

        self.metricvalues = dict()

    def compute_metric(self,metric, **args):
        match metric:
            case "distortion":
                return self.calcDistortion(**args)
            case "inkratio":
                return self.calcInkRatio(**args)
            case _:
                print("not yet implemented")
                return 
            
    def store_metric(self,metricname,metricvalue):
        if isinstance(metricvalue,np.ndarray): metricvalue = [float(x) for x in metricvalue]
        self.metricvalues[metricname] = metricvalue

    def write_to_file(self,path):
        import json 
        with open(path, 'w') as fdata:
            json.dump(self.metricvalues, fdata,indent=4)

    def getDrawing(self):
        """
        Get drawing for computing metrics. Not intended for visualization, instead use the draw methods from AbstractBundling.
        """
        if self.bundleDrawing is None: 
            self.bundleDrawing = self._metricDraw()
        return self.bundleDrawing
    
    def getStraightDrawing(self):
        """
        Get drawing for computing metrics. Not intended for visualization, instead use the draw methods from AbstractBundling.
        """        
        if self.straightDrawing is None:
            self.straightDrawing = self._metricDraw(straightline=True)    
        return self.straightDrawing
    
    def getStraightLineGraph(self):
        """
        Get drawing for computing metrics. Not intended for visualization, instead use the draw methods from AbstractBundling.
        """        
        if self.straightGraph is None:
            self.straightGraph = self._genStraightGraph()
        return self.straightGraph 
    
    def _genStraightGraph(self):
        SL = StraightLine(self.G)
        SL.bundle()
        return SL.G

    def _metricDraw(self,straightline=False, DPI=192,CIRCLE=10.0,LINEWIDTH=0.5,return_fig=False):
        '''
        Draw the bundling. Either using the assign color function or the coloring given by the bundling. 
        Not intended for visualization, but for use in metrics.
        '''
        import pylab as plt

        H = self.G if not straightline else self.getStraightLineGraph()

        nx.set_edge_attributes(H, '50%', name='Opacity')

        fig, ax = plt.subplots(figsize=(H.graph['xmax'] / DPI, H.graph['ymax'] / DPI), dpi=DPI)
        ax.axis('off')

        for source, target, data in H.edges(data = True):
            if 'Xapprox' in data and 'Yapprox' in data:
                X = data['Xapprox']
                Y = data['Yapprox']
            elif 'Spline' in data:
                points = [(H.nodes[source]['X'], H.nodes[source]['Y'])] + data['Spline'].points + [(H.nodes[target]['X'], H.nodes[target]['Y'])]

                X, Y = self.Bundle.approxBezier(points, 50)
            else:
                X = [H.nodes[source]['X'], H.nodes[target]['X']]
                Y = [H.nodes[source]['Y'], H.nodes[target]['Y']]
            
            ax.plot(X, Y, color='black',linewidth=LINEWIDTH)

        pos = [(data['X'], data['Y']) for v, data in H.nodes(data=True)]
        X, Y = zip(*pos)

        ax.scatter(X, Y, marker='.', color='black', s = CIRCLE, zorder=2)

        if return_fig: 
            return fig, ax
        buffer, (width, height) = fig.canvas.print_to_buffer()
        imgBundle = np.frombuffer(buffer,dtype=np.uint8).reshape((height,width,4))
        plt.close(fig)
        return imgBundle

###########################################################################
# Metric calculations                                                     #
###########################################################################

    def calcDistortion(self,return_mean=True):
        '''
        Calculate the distortion by summing up the polyline segments of the Bezier approximation.
        if return_mean == True (default) will only return the mean. Otherwise, returns the full 
        array of distortions.
        '''

        distortions = np.zeros((self.G.number_of_edges()),dtype=np.float32)

        for index, (source, target, data) in enumerate(self.G.edges(data=True)):
            X = np.array(data['X'])
            Y = np.array(data['Y'])

            dx = np.diff(X)
            dy = np.diff(Y)

            #L2 distance between first and last point on curve
            lStraight = np.sqrt((X[-1] - X[0]) * (X[-1] - X[0]) + (Y[-1] - Y[0]) * (Y[-1] - Y[0]))
            lPoly     = np.sum(np.sqrt(dx * dx + dy * dy))

            distortions[index] = lPoly / lStraight

        if return_mean: return np.mean(distortions)
        else: return distortions


    def calcInkRatio(self,return_mean=True):
        '''
        Calculate the ink ratio.
        '''
        from PIL import Image as PILImage
        GREY_THRESH = 255            

        imgBundle = self.getDrawing()
        imGrayBundle  = np.array(PILImage.fromarray(imgBundle).convert("L"))

        imStraight = self.getStraightDrawing()
        imGrayStraight = np.array(PILImage.fromarray(imStraight).convert("L"))

        inkratio = (imGrayBundle < GREY_THRESH).sum() / (imGrayStraight < GREY_THRESH).sum()

        return inkratio
    
