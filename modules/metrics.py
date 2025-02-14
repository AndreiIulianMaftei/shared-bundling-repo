import os
import networkx as nx
import numpy as np
from gdMetriX import number_of_crossings

from modules.abstractBundling import RealizedBundling
from modules.EPB.straight import StraightLine
from modules.EPB.experiments import Metrics as EPBMetrics

PATH_TO_PICKLE_FOLDER = "pickle/"
if not os.path.isdir(PATH_TO_PICKLE_FOLDER): os.mkdir(PATH_TO_PICKLE_FOLDER)

class Metrics():
    implemented_metrics = ["distortion", "inkratio", "self_intersections", "all_intersections"]
    def __init__(self,bundle:RealizedBundling, verbose=True):
        """
        G should be a RealizedBundling instance. 
        All nodes should have 'X' and 'Y' coordinates defined. 
        All edges should have "Spline_X" and "Spline_Y" attributes defined. 
        """
        self.Bundle  = bundle
        self.G       = bundle.G
        self.verbose = verbose

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
            case "self_intersections":
                return self.calcSelfIntersections(**args)
            case "all_intersections":
                return self.calcAllIntersections(**args)
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
    
    def calcAmbiguity(self, return_mean=True):
        '''
        Calculate the ambiguity. Store intermediate results in the folder /pickle and reuse. 
        Important if parameters for the ambiguity are changed then delete the files.
        '''
        ambiguities = EPBMetrics.calcAmbiguity(self.G, PATH_TO_PICKLE_FOLDER, self.G.graph['name'])

        return [float(x) for x in ambiguities]

    def calcSelfIntersections(self, return_mean=True):
        """Count the number of self-intersections in a polyline defined by a list of points."""
        intersections = np.zeros(self.G.number_of_edges())
        for index, (u,v, data) in enumerate(self.G.edges(data=True)):
            ncontrol_points = len(data['X'])
            H = nx.Graph()
            H.add_edges_from([(i,i+1) for i in range(ncontrol_points-1)])
            pos = {i: (data['X'][i], data['Y'][i]) for i in range(ncontrol_points)}

            intersections[index] = number_of_crossings(H,pos)
        
        if return_mean: return np.mean(intersections)
        return intersections
    
    def calcAllIntersections(self,return_mean=True):
        """Counts all intersections in the bundling"""
        H = nx.Graph()
        H.add_nodes_from(self.G.nodes())
        pos = {v: (self.G.nodes[v]['X'], self.G.nodes[v]['Y']) for v in self.G.nodes()}
        for u,v,data in self.G.edges(data=True):
            ncontrol_points = len(data["X"])
            
            eid = (u,v)
            #Add first and last edge 
            H.add_edge(u,( eid ,1))
            H.add_edge((eid, ncontrol_points-1), v)

            #Fill in edges from first to last control point
            H.add_edges_from([( (eid, i), (eid,i+1) ) for i in range(1,ncontrol_points-1)])

            #add positions 
            pos |= {(eid,i): (data["X"][i], data["Y"][i]) for i in range(1,ncontrol_points)}

        if self.verbose: print("computing crossings, will take a while")
        ncrossings = number_of_crossings(H,pos)
        return ncrossings

    def calcFrechet(self,return_mean=True):
        frechet = np.zeros(self.G.number_of_edges())

        for index, (u,v,data) in enumerate(self.G.edges(data=True)):

            points = np.array([[x,y] for x,y in zip(data['X'], data['Y'])])

            x0 = self.G.nodes[u]['X']
            y0 = self.G.nodes[u]['Y']
            x1 = self.G.nodes[v]['X']
            y1 = self.G.nodes[v]['Y']
            line = np.array([[x0,y0], [x1,y1]])

            projected_points = project_points_to_line(points, line)

            minx,maxx = min(x0, x1), max(x0,x1)
            miny,maxy = min(y0, y1), max(y0,y1)
            inside_segment_mask = (
                (minx <= projected_points[:,0]) & (projected_points[:,0] <= maxx) & 
                (miny <= projected_points[:,1]) & (projected_points[:,1] <= maxy)
            )

            inside_segment  = points[ inside_segment_mask]
            outside_segment = points[~inside_segment_mask]

            #Projected distance from (x2,y2)  to line ((x0,y0), (x1,y1)) is given by 
            #d = \frac{\left| (x_2 - x_0)(y_1 - y_0) - (y_2 - y_0)(x_1 - x_0) \right|} {||(x_1,y_1) - (x_0, y_0)||}
            ab = line[1] - line[0]
            ab_norm = np.sqrt(np.sum(np.square(ab)))
            inside_distances = np.abs(np.cross(inside_segment - line[0], ab)) / ab_norm 

            if outside_segment.size > 0: 
                #For outside the segment, get the minimum distance to either endpoint
                outside_distances = np.min(np.linalg.norm(outside_segment[:, np.newaxis, :] - line, axis=2), axis=1)
            else: outside_distances = np.zeros(1)

            frechet[index] = max(np.max(inside_distances), np.max(outside_distances))
        
        if return_mean: return np.mean(frechet)
        return frechet

    


##################################################
# Helper functions                               #
##################################################

def project_points_to_line(points:np.array, line: np.array):
    a,b = line 

    ab = b - a 
    ab_norm_sq = np.dot(ab, ab)

    ap = points - a 

    projection = np.dot(ap, ab) / ab_norm_sq 
    projected_points = a + np.outer(projection, ab)
    return projected_points