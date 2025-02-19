import os
import networkx as nx
import numpy as np
from gdMetriX import number_of_crossings

from modules.abstractBundling import RealizedBundling
from modules.EPB.straight import StraightLine
from modules.EPB.experiments import Metrics as EPBMetrics
import math

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
            case "SL_angle":
                return self.calcSLAngle(**args)
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
                
        return angles

    def calcFrechetDistance(self, return_mean=True):
        """Calculate the Frechet distance between the straight line and the bundled line."""
        from frechetdist import frdist

        ma = -999999999999
        mi = 999999999999
        frechet = []
        polylines = []
        x= 0
        
        list_edges = list(self.G.edges(data=True))

        for index, (u, v, data) in enumerate(list_edges):
            numbers_x = [float(num) for num in data.get('X')]
            numbers_y = [float(num) for num in data.get('Y')]

            polyline = [(numbers_x[i], numbers_y[i]) for i in range(len(numbers_x))]
            polylines.append(polyline)

            x0 = self.G.nodes[u]['X']
            y0 = self.G.nodes[u]['Y']
            x1 = self.G.nodes[v]['X']
            y1 = self.G.nodes[v]['Y']

            t_values = np.linspace(0, 1, num=len(polyline))
            x_values = x0 + t_values * (x1 - x0)
            y_values = y0 + t_values * (y1 - y0)

            interpolated_points = [(float(x), float(y)) for x, y in zip(x_values, y_values)]

            if len(polyline) == 0:
                x = 0
            else:
                x = frdist(interpolated_points, polyline)

            frechet.append(x)
            ma = max(ma, x)
            mi = min(mi, x)

        return frechet
    
    def calcProjectedMonotonicity(self, return_mean = True):
        monotonicities = []
        list_edges = list(self.G.edges(data=True))

        for index, (u, v, data) in enumerate(list_edges):
            X = [float(num) for num in data.get('X')]
            Y = [float(num) for num in data.get('Y')]
            polyline = [(X[i], Y[i]) for i in range(len(X))]

            if len(polyline) < 2:
                monotonicities.append(0.0)
                continue

            x0 = self.G.nodes[u]['X']
            y0 = self.G.nodes[u]['Y']
            x1 = self.G.nodes[v]['X']
            y1 = self.G.nodes[v]['Y']

            projections = []
            for (px, py) in polyline:
                proj = Metrics.project_point_onto_line((px, py), (x0, y0), (x1, y1))
                projections.append(proj)

            sign_change_count = 0
            direction = 0 

            for i in range(1, len(projections)):
                prev_x = projections[i - 1][0]
                curr_x = projections[i][0]

                if i == 1:
                    direction = 1 if curr_x >= prev_x else -1
                else:
                    if curr_x > prev_x and direction == -1:
                        sign_change_count += 1
                        direction = 1
                    elif curr_x < prev_x and direction == 1:
                        sign_change_count += 1
                        direction = -1

            denom = max(1, len(projections) - 1)
            normalized_mono = sign_change_count / denom

            monotonicities.append(normalized_mono)

        return monotonicities
    
    def calcMonotonicity(self, return_mean = True):
        """
        Computes a 'monotonicity' measure for each edge, normalized by the
        number of polyline control points. Returns a list of normalized
        monotonicity values (one per edge).
        """
        monotonicities = []
        
        list_edges = list(self.G.edges(data=True))

        for index, (u, v, data) in enumerate(list_edges):
            X = [float(num) for num in data.get('X')]
            Y = [float(num) for num in data.get('Y')]
            
            x0 = self.G.nodes[u]['X']
            y0 = self.G.nodes[u]['Y']
            x1 = self.G.nodes[v]['X']
            y1 = self.G.nodes[v]['Y']

            polyline = [(X[i], Y[i]) for i in range(len(X))]
            
            if len(polyline) < 2:
                monotonicities.append(0.0)
                continue

    
            A = self.Point(x0, y0)
            B = self.Point(polyline[1][0], polyline[1][1])
            C = self.Point(x1, y1)
            main_orientation = Metrics.orientation(A, B, C)

            for i in range(len(polyline)):
                px, py = polyline[i]
                proj = Metrics.project_point_onto_line((px, py), (x0, y0), (x1, y1))
                dist_from_source = np.linalg.norm(np.array(proj) - np.array((x0, y0)))
                
            
                point_orientation = Metrics.orientation(A, self.Point(px, py), C)
                distance = np.linalg.norm(np.array((px, py)) - proj)
                
                if math.isclose(distance, 0, abs_tol=1e-5):
                    distance = 0.0
                
                if point_orientation != main_orientation and not math.isclose(distance, 0.0, abs_tol=1e-5):
                    distance = -distance
                
                polyline[i] = (dist_from_source, distance)

            monotonicity = 0
            xDirection = None
            yDirection = None

            for i in range(1, len(polyline)):
                curr_x, curr_y = polyline[i]
                prev_x, prev_y = polyline[i - 1]

                if i == 1:
                    xDirection = 1 if curr_x >= prev_x else -1
                    yDirection = 1 if curr_y >= prev_y else -1
                    continue

                if curr_x > prev_x and xDirection == -1:
                    monotonicity += 1
                    xDirection = 1
                elif curr_x < prev_x and xDirection == 1:
                    monotonicity += 1
                    xDirection = -1

                if curr_y > prev_y and yDirection == -1:
                    monotonicity += 1
                    yDirection = 1
                elif curr_y < prev_y and yDirection == 1:
                    monotonicity += 1
                    yDirection = -1

           
            denom = max(1, len(polyline) - 1)
            norm_monotonicity = monotonicity / denom

            monotonicities.append(norm_monotonicity)

        return monotonicities

    


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

def project_point_onto_line( point, line_start, line_end):
        
        line_vec = np.array(line_end) - np.array(line_start)
        point_vec = np.array(point) - np.array(line_start)
        line_len_squared = np.dot(line_vec, line_vec)
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_squared))
        projection = np.array(line_start) + t * line_vec
        return projection

def orientation( p, q, r): 
        
        
        val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y)) 
        if (val > 0): 
            
            return 1
        elif (val < 0): 
            
            return 2
        else: 
            
            return 0

def doIntersect(p1,q1,p2,q2): 
        
        
        o1 = Metrics.orientation(p1, q1, p2) 
        o2 = Metrics.orientation(p1, q1, q2) 
        o3 = Metrics.orientation(p2, q2, p1) 
        o4 = Metrics.orientation(p2, q2, q1) 
    
        if ((o1 != o2) and (o3 != o4)): 
            return True
    
    
        if ((o1 == 0) and Metrics.onSegment(p1, p2, q1)): 
            return True
    
        if ((o2 == 0) and Metrics.onSegment(p1, q2, q1)): 
            return True
    
        if ((o3 == 0) and Metrics.onSegment(p2, p1, q2)): 
            return True
    
        if ((o4 == 0) and Metrics.onSegment(p2, q1, q2)): 
            return True
    
        return False
