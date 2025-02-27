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
    @staticmethod
    def getImplementedMetrics():
        return Metrics.getGlobalMetrics() + Metrics.getLocalMetrics()

    @staticmethod
    def getGlobalMetrics():
        return ['inkratio', 'all_intersections', 'ambiguity']
    
    @staticmethod
    def getLocalMetrics():
        return ['distortion', 'frechet', 'directionality', 'monotonicity', 'projected_monotonicity', 'SL_angle', 'self_intersections']

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
            case "SL_angle":
                return self.calcSLAngle(**args)
            case "ambiguity": 
                return self.calcAmbiguity(**args)
            case "frechet":
                return self.calcFrechet(**args)
            case "directionality":
                return self.calcDirectionalityChange(**args)
            case "monotonicity":
                return self.calcMonotonicity(**args)
            case "projected_monotonicity": 
                return self.calcProjectedMonotonicity(**args)
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
    
    def calcMeanOccupationArea(self):
        """
        Calculate the Mean Occupation Area (MOA) using 8×8 pixel blocks,
        but consider a block occupied only if at least 4 pixels are drawn.
        MOA = (1 / number_of_edges) * (count of occupied 8×8 blocks).
        """
        from PIL import Image as PILImage
        GREY_THRESH = 255
        BLOCK_SIZE = 8
        MIN_OCCUPIED_PIXELS = 4  # Only count the block if >= 4 pixels are occupied

        imgBundle = self.getDrawing()  
        imGrayBundle = np.array(PILImage.fromarray(imgBundle).convert("L"))

        height, width = imGrayBundle.shape

        occupied_blocks = 0
        vblocks = int(np.ceil(height / BLOCK_SIZE))
        hblocks = int(np.ceil(width / BLOCK_SIZE))

        for by in range(vblocks):
            row_start = by * BLOCK_SIZE
            row_end   = min(row_start + BLOCK_SIZE, height)

            for bx in range(hblocks):
                col_start = bx * BLOCK_SIZE
                col_end   = min(col_start + BLOCK_SIZE, width)

                # Extract
                sub_block = imGrayBundle[row_start:row_end, col_start:col_end]

                # below GREY_THRESH
                num_occupied_pixels = np.count_nonzero(sub_block < GREY_THRESH)
                if num_occupied_pixels >= MIN_OCCUPIED_PIXELS:
                    occupied_blocks += 1

        n_edges = self.G.number_of_edges()
        moa = occupied_blocks / n_edges if n_edges > 0 else 0.0

        return moa

    def calcEdgeDensityDistribution(self, block_size=8):
        """
        Calculates Edge Density Distribution (EDD) by partitioning the
        canvas into blocks of size 'block_size' x 'block_size'.
        
        For each block a, we define p(a) = fraction of pixels in that
        block that are drawn (non-white). Then,
        
            EDD = (1 / #blocks) * sum_{a in A} | p(a) - mean(p(a)) |.
        """
        from PIL import Image as PILImage
        GREY_THRESH = 255
        
        imgBundle = self.getDrawing()  
        imGrayBundle = np.array(PILImage.fromarray(imgBundle).convert("L"))
        
        height, width = imGrayBundle.shape

        p_values = []
        
        vblocks = int(np.ceil(height / block_size))
        hblocks = int(np.ceil(width  / block_size))
        
        for by in range(vblocks):
            row_start = by * block_size
            row_end   = min(row_start + block_size, height)
            
            for bx in range(hblocks):
                col_start = bx * block_size
                col_end   = min(col_start + block_size, width)
                
                # Extract sub-block
                sub_block = imGrayBundle[row_start:row_end, col_start:col_end]
                
                # Count how many pixels are drawn (non-white)
                num_occupied = np.count_nonzero(sub_block < GREY_THRESH)
                
                block_area = (row_end - row_start) * (col_end - col_start)
                p_block = num_occupied / float(block_area)
                
                p_values.append(p_block)
        

        
        p_values = np.array(p_values, dtype=np.float32)
        p_mean = p_values.mean()
        
        edd = np.mean(np.abs(p_values - p_mean))
        return float(edd)
   
    
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
        """
        Implements 'fastFrechet'. Since we always compare to a straight line 
        it suffices to compute distances of the curve from outside and inside 
        the line segment, both of which can be simplified.
        """
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

    def calcSLAngle(self, return_mean=True): 
        angles = np.zeros((self.G.number_of_edges()),dtype=np.float32)
        for index, (u,v,data) in enumerate(self.G.edges(data=True)):
            x0 = self.G.nodes[u]['X']
            y0 = self.G.nodes[u]['Y']
            x1 = self.G.nodes[v]['X']
            y1 = self.G.nodes[v]['Y']
            
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
                
            angles[index] = a
        
        if return_mean: return np.mean(angles)
        return angles
    
    def calcProjectedMonotonicity(self, return_mean = True,normalize=True):
        monotonicities = np.zeros(self.G.number_of_edges(), dtype=np.float32)

        for index, (u, v, data) in enumerate(self.G.edges(data=True)):
            points = np.array([[x,y] for x,y in zip(data['X'], data['Y'])])

            if np.shape(points)[0] <= 2:
                continue

            x0 = self.G.nodes[u]['X']
            y0 = self.G.nodes[u]['Y']
            x1 = self.G.nodes[v]['X']
            y1 = self.G.nodes[v]['Y']
            assert x0 != x1, "Vertical line"

            line = np.array([[x0,y0], [x1,y1]])
            projection = project_points_to_line(points, line)

            #Consecutive difference in x axis
            diffX = np.diff(projection[:,0])
            signs = np.sign(diffX)

            #Consecutive difference in signs. e.g. consecutive positive means zero difference
            directionChanges = np.diff(signs)

            monotonicity = np.count_nonzero(directionChanges)

            #We can normalize by number of control points
            if normalize: monotonicity /= points.shape[0]

            monotonicities[index] = monotonicity

        if return_mean: return np.mean(monotonicities)
        return monotonicities
    
    def calcMonotonicity(self, return_mean = True,normalize=True):
        """
        Computes a 'monotonicity' measure for each edge, normalized by the
        number of polyline control points. Returns a list of normalized
        monotonicity values (one per edge).
        """
        monotonicities = np.zeros(self.G.number_of_edges(), dtype=np.float32)

        for index, (u, v, data) in enumerate(self.G.edges(data=True)):
            points = np.array([[x,y] for x,y in zip(data['X'], data['Y'])])

            if np.shape(points)[0] <= 2:
                continue

            x0 = self.G.nodes[u]['X']
            y0 = self.G.nodes[u]['Y']
            x1 = self.G.nodes[v]['X']
            y1 = self.G.nodes[v]['Y']

            line = np.array([[x0,y0], [x1,y1]])
            projection = project_points_to_line(points,line)

            s = np.linalg.norm(projection - line[0],axis=1)
    
            d = orientationVecTriples(points)

            sdiff = np.diff(s)
            ddiff = np.diff(d)

            sprod = sdiff[1:] * sdiff[:-1] # (s_i - s_{i-1}) * (s_{i-1} - s_{i-2})
            dprod = ddiff[1:] * ddiff[:-1]

            s_changes = np.sum(sprod < 0)
            d_changes = np.sum(dprod < 0 )
            monotonicity = s_changes + d_changes
           
            if normalize: monotonicity /= points.shape[0]

            monotonicities[index] = monotonicity

        if return_mean: return np.mean(monotonicities)
        return monotonicities
    
    def calcDirectionalityChange(self,return_mean=True,normalize=True):
        dchange = np.zeros(self.G.number_of_edges(),np.float32)

        for index, (u,v,data) in enumerate(self.G.edges(data=True)):
            points = np.array([[x,y] for x,y in zip(data['X'], data['Y'])])

            signs = orientationVecTriples(points)

            edgechanges = np.sum(signs[1:] * signs[:-1] < 0)

            if normalize: edgechanges /= (points.shape[0] - 1)

            dchange[index] = edgechanges
        
        if return_mean: return np.mean(dchange)
        return dchange



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

def orientationVec(points: np.ndarray, a: np.ndarray, b:np.ndarray):
    ab = b - a 
    ap = points - a 
    return np.sign(np.cross(ab,ap))

def orientationVecTriples(points:np.ndarray):
    """
    Computes the orientation of all consecutive pairs of triples in array points.
    """
    if points.shape[0] < 3: return np.array([])

    p1 = points[:-2]
    p2 = points[1:-1]
    p3 = points[2:]

    v1 = p2 - p1 
    v2 = p3 - p1

    return np.sign(np.cross(v1,v2))