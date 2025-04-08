import csv
import json
import networkx as nx
import numpy as np
from datetime import datetime
from dateutil import parser

from abstractBundling import GWIDTH, GraphLoader, AbstractBundling
from experiments import Experiment
from ogdfWrapper import OGDF_Wrapper
from sepb import SpannerBundling, SpannerBundlingAStar, SpannerBundlingFG, SpannerBundlingFGSSP, SpannerBundlingNoSP, SpannerBundlingNoSPWithWF
from straight import StraightLine
from reader import Reader
from spanner import Spanner

from pyproj import Transformer
import gdMetriX
from sklearn.cluster import OPTICS, cluster_optics_dbscan

def scaleToWidth(G, G_width):
        xmin = sys.maxsize
        xmax = -sys.maxsize - 1
        ymin = sys.maxsize
        ymax = -sys.maxsize - 1

        for n, data in G.nodes(data = True):
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

        for node, data in G.nodes().data():
            G.nodes()[node]['X'] = (data['X'] - xmin) * factor
            data['Y'] = (data['Y'] - ymin) * factor
            G.nodes()[node]['x'] = G.nodes()[node]['X']
            G.nodes()[node]['y'] = G.nodes()[node]['Y']

        for (u,v,data) in G.edges(data = True):
            d1 = G.nodes[u]
            d2 = G.nodes[v]

            dx = d1['X'] - d2['X']
            dy = d1['Y'] - d2['Y']

            l = np.sqrt((dx)**2 + (dy)**2)

            data['dist'] = l

        G.graph['xmin'] = 0
        G.graph['xmax'] = G_width
        G.graph['ymin'] = 0
        G.graph['ymax'] = (ymax - ymin) * factor

def remove_overlap(G : nx.Graph, r):
    """
    This implements the GTREE algorithm. 
    """
    X = np.zeros((G.number_of_nodes(),2),dtype=np.float64) 
    for i, (v, data) in enumerate(G.nodes(data=True)):
        X[i][0] = data["X"] 
        X[i][1] = data["Y"]

    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if (X[i][1] - X[j][1]) < 0.0000001 and (X[i][0] - X[j][0]) < 0.0000001:
                X[j][0] += 0.0000001

    hasOverlap = True
    while(hasOverlap):
        tri = Delaunay(X)
        G_tri = nx.Graph()

        indptr = tri.vertex_neighbor_vertices[0]
        indices = tri.vertex_neighbor_vertices[1]

        for i in range(len(X)):
            for n in indices[indptr[i]:indptr[i+1]]:
                if n < 0:
                    continue
                G_tri.add_edge(i, n)

        hasOverlap = False
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                dist = np.sqrt((X[i][0] - X[j][0])**2 + (X[i][1] - X[j][1])**2)

                # if dist < 0.000000000000001:
                #     dist = r

                if dist < r:
                    G_tri.add_edge(i, j)

                if G_tri.has_edge(i,j):
                    G_tri[i][j]['dist'] = dist
                    G_tri[j][i]['dist'] = dist
                    
                    if dist > 2 * r:
                        G_tri[i][j]['c_ij'] = 1
                        G_tri[j][i]['c_ij'] = 1               
                    else:
                        hasOverlap = True
                        G_tri[i][j]['c_ij'] = 1 + ((2 * r) / (dist))
                        G_tri[j][i]['c_ij'] = 1 + ((2 * r) / (dist))    
                        # G_tri[i][j]['c_ij'] = 1 + ((2 * r) - dist)
                        # G_tri[j][i]['c_ij'] = 1 + ((2 * r) - dist)   


        T = nx.minimum_spanning_tree(G_tri, weight="dist") 
        nx.set_node_attributes(T, False, "processed")

        # for u,v,data in G_tri.edges(data=True):
        #     if data['dist'] > 1:
        #         data['dist'] = 1

        def growAtNode(r):
            for c in T.neighbors(r):
                if T.nodes[c]['processed'] == False:
                    X_new[c][0] = X_new[r][0] + G_tri[r][c]['c_ij'] * (X[c][0] - X[r][0])
                    X_new[c][1] = X_new[r][1] + G_tri[r][c]['c_ij'] * (X[c][1] - X[r][1])
                    T.nodes[c]['processed'] = True

                    growAtNode(c)
            return

        X_new = np.zeros((G.number_of_nodes(),2),dtype=np.float64)

        root = np.random.randint(0, len(G.nodes()))
        X_new[root][0] = X[root][0]
        X_new[root][1] = X[root][1]
        T.nodes[root]['processed'] = True
        growAtNode(root)
        
        X = X_new
        print('yes')

    for i, (v, data) in enumerate(G.nodes(data = True)):
        data['X'] = X[i][0]
        data['Y'] = X[i][1]
        data['x'] = X[i][0]
        data['y'] = X[i][1]
   
    return G

def read_edge_file(path):
    G = nx.Graph()
    with open(path, 'r' ) as theFile:
        
        for i, line in enumerate(theFile):
            st = line.split()
            src = st[0]
            tgt = st[1]

            G.add_edge(src, tgt)

    return G

def extract_by_location(long, lat, max_diff, city, G):
    """
    Extracts and processes location-based data from a Gowalla dataset, filters nodes 
    based on proximity to a specified longitude and latitude, and generates subgraphs 
    for further analysis and visualization.
    Args:
        long (float): Longitude of the target location.
        lat (float): Latitude of the target location.
        max_diff (float): Maximum allowable difference in longitude and latitude 
                          for filtering nodes.
        city (str): Name of the city for output file naming.
        G (networkx.Graph): Input graph containing nodes and edges.
    Workflow:
        1. Reads the Gowalla dataset from a file.
        2. Filters nodes based on proximity to the specified longitude and latitude.
        3. Groups nodes by month and updates their coordinates in the graph.
        4. Creates subgraphs for each month and processes them:
            - Ensures subgraphs meet size criteria (nodes or edges > 100).
            - Extracts the largest connected component if applicable.
        5. Adjusts overlapping node coordinates to ensure uniqueness.
        6. Scales and removes overlap in the graph for visualization purposes.
        7. Saves the processed subgraphs to GraphML files.
    Outputs:
        - GraphML files for each processed subgraph, saved in the 
          "output_userstudy/gowalla" directory with filenames based on the city 
          and month.
    Notes:
        - The function uses randomization to adjust overlapping node coordinates.
        - The graph is scaled and adjusted multiple times for visualization purposes.
    """

    cur_date = datetime.utcnow().date()
    
    month_set = dict()
    
    for i in range(1,13):
        month_set[i] = dict()
    
    with open("datasets/gowalla/Gowalla_totalCheckins.txt", 'r' ) as theFile:
    
        for i, line in enumerate(theFile):
            line = line.split()
            date = parser.parse(line[1]).date()
            

            #if cur_date != date:

                # G_sub = G.subgraph(node_set).copy()

                # if len(G_sub.nodes()) > 100 or len(G_sub.edges()) > 100:
                #     outpath = f"output_userstudy/gowalla/gowalla_{str(date)}_"

                #     straight = StraightLine(G_sub)
                #     straight.scaleToWidth(GWIDTH)
                #     straight.bundle()
                #     straight.draw(outpath, fileAddition=f'_spatial')

                #     sepb = SpannerBundlingFG(G_sub)
                #     sepb.scaleToWidth(GWIDTH)
                #     sepb.bundle()
                #     sepb.draw(outpath, fileAddition=f'_spatial')

                # node_set = set()
                # cur_date = date

            lo = float(line[2])
            la = float(line[3])

            if abs(lo - long) < max_diff and abs(la - lat) < max_diff: 
                n = line[0]
                month_set[date.month][n] = (lo, la)

        for i in range(1, 13):
            node_set = month_set[i]
            nodes = []
            
            for n, (la, lo) in node_set.items():
                G.nodes[n]['x'] = lo
                G.nodes[n]['y'] = la
                G.nodes[n]['X'] = lo
                G.nodes[n]['Y'] = la
                
                nodes.append(n)

            G_sub = G.subgraph(nodes).copy()
            print(len(G_sub.nodes()), len(G_sub.edges()))

            if len(G_sub.nodes()) > 100 or len(G_sub.edges()) > 100:
            
                Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
                G = G.subgraph(Gcc[0]).copy()

        np.random.seed(0)
        for u,d1 in G.nodes(data=True):
            for v, d2 in G.nodes(data=True):
                if u == v:
                    continue
                if abs(d1['X'] - d2['X']) < 0.000001 and abs(d1['Y'] - d2['Y']) < 0.000001:
                    d2['X'] = d2['X'] + (np.random.rand() - 0.5) * 0.0001
                    d2['Y'] = d2['Y'] + (np.random.rand() - 0.5) * 0.0001

        scaleToWidth(G, GWIDTH)
        remove_overlap(G, 25)
        scaleToWidth(G, GWIDTH)

        outpath = f"output_userstudy/gowalla/gowalla_{city}_{i}"

        nx.write_graphml(G_sub, outpath + '.graphml')
                
if __name__ == "__main__":
    print('starting computing layouts')

    max_diff = 0.1

    locations = [(34, -118.2, "LA"), (40.7, -74, 'NY'), (33.75, -84.4, 'ATLANTA'), (25.9, -80.2, 'MIAMI'), (29.7, -95.4, 'HOUSTON'), (29.4, -98.4, 'SANANTONIO'), (32.8, 96.8, 'DALLAS'), (37.75, -122.4, 'SF'), (37.4, -121.9, 'SANJOSE'), (36.2, -115.2, 'LASVEGAS'), ]
    
    G = read_edge_file("datasets/gowalla/Gowalla_edges.txt")

    for long, lat, city in locations:
        extract_by_location(long, lat, max_diff, city, G)