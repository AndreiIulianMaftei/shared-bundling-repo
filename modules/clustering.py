from collections import defaultdict
import numpy as np
import networkx as nx
import cv2
from sklearn.cluster import DBSCAN, HDBSCAN
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

from metrics_pipeline import read_bundling

CUTOFF = 0.65

def delaunay_graph(G_in, x_attr="X", y_attr="Y"):
    """
    Create a NetworkX graph representing the Delaunay triangulation of nodes
    in G_in. Node coordinates must be stored in node attributes X and Y.

    Parameters
    ----------
    G_in : networkx.Graph
        Input graph containing node coordinates in attributes x_attr, y_attr.
    x_attr, y_attr : str
        Attribute names holding coordinates.

    Returns
    -------
    G_del : networkx.Graph
        Graph whose nodes correspond to original nodes, and edges correspond
        to the Delaunay triangulation edges.
    tri : scipy.spatial.Delaunay
        The computed triangulation.
    pts : np.ndarray
        Coordinates array in triangulation order.
    index_to_node : list
        Mapping from triangulation index → original node ID.
    """

    # ----------------------------------------
    # 1. Collect points
    # ----------------------------------------
    pts = []
    index_to_node = []

    for node, data in G_in.nodes(data=True):
        if x_attr not in data or y_attr not in data:
            raise ValueError(f"Node {node} missing '{x_attr}' or '{y_attr}'.")

        pts.append([float(data[x_attr]), float(data[y_attr])])
        index_to_node.append(node)

    pts = np.array(pts, dtype=float)

    # ----------------------------------------
    # 2. Compute triangulation
    # ----------------------------------------
    tri = Delaunay(pts)

    # ----------------------------------------
    # 3. Build Delaunay graph
    # ----------------------------------------
    G_del = nx.Graph()

    # Add all nodes with their positions
    for i, node in enumerate(index_to_node):
        x, y = pts[i]
        G_del.add_node(node, X=x, Y=y)

    # Add edges for each triangle's sides
    for simplex in tri.simplices:   # each simplex has 3 vertices: i,j,k
        i, j, k = simplex
        ni, nj, nk = index_to_node[i], index_to_node[j], index_to_node[k]

        d1 = np.linalg.norm(pts[i] - pts[j])
        d2 = np.linalg.norm(pts[j] - pts[k])
        d3 = np.linalg.norm(pts[k] - pts[i])

        G_del.add_edge(ni, nj, weight=d1)
        G_del.add_edge(nj, nk, weight=d2)
        G_del.add_edge(nk, ni, weight=d3)
        
    return G_del, tri, pts, index_to_node


def average_in_segment_rect_xy(img, x1, y1, x2, y2, half_width):
    """
    img: HxW or HxWx3 numpy array
    (x1,y1), (x2,y2): centers of opposite sides, in *x=left-right, y=top-bottom* coords
    half_width: half rectangle thickness (pixels)
    """
    # direction in x,y space
    vx, vy = x2 - x1, y2 - y1
    length = np.hypot(vx, vy)
    if length == 0:
        return None

    ux, uy = vx / length, vy / length
    nx, ny = -uy, ux  # perpendicular

    H, W = img.shape[:2]

    # bounding box in x,y (then convert to col,row)
    min_x = int(np.floor(min(x1, x2) - half_width - 2))
    max_x = int(np.ceil (max(x1, x2) + half_width + 2))
    min_y = int(np.floor(min(y1, y2) - half_width - 2))
    max_y = int(np.ceil (max(y1, y2) + half_width + 2))

    # clamp to image bounds (remember: cols = x, rows = y)
    min_col = max(min_x, 0)
    max_col = min(max_x, W - 1)
    min_row = max(min_y, 0)
    max_row = min(max_y, H - 1)

    cols = np.arange(min_col, max_col + 1)
    rows = np.arange(min_row, max_row + 1)
    C, R = np.meshgrid(cols, rows)  # C = x, R = y

    # vector from (x1,y1) to each pixel center
    RX = C - x1
    RY = R - y1

    # projection along segment and perpendicular distance (in x,y space)
    t = RX * ux + RY * uy
    d = RX * nx + RY * ny

    mask = (t >= 0) & (t <= length) & (np.abs(d) <= half_width)
    if not np.any(mask):
        return None

    # rows -> first index, cols -> second index
    if img.ndim == 2:
        vals = img[min_row:max_row+1, min_col:max_col+1][mask]
        return float(vals.mean())
    else:
        vals = img[min_row:max_row+1, min_col:max_col+1][mask]
        return vals.mean(axis=0)


def clustering_gestalt(G, img_path, scale=.25):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.resize(gray, (0,0), fx=scale, fy=scale)
    blur = cv2.GaussianBlur(blur, (7, 7), 0)
    blur = blur.astype(np.float32)
    
    nodes = list(G.nodes())
  
    G_Del, tri, pts, index_to_node = delaunay_graph(G)
    alpha = 0.85
    
    to_remove = []
    for u,v,data in G_Del.edges(data=True):
        if data['weight'] < 5:
            to_remove.append((u,v))
    
    for u,v in to_remove:
        G_Del.remove_edge(u,v)
           
    nodes = list(G_Del.nodes(data=True))
    
    ink_lookup = {}
    for u,v in G_Del.edges():

        x1 = int(G_Del.nodes[u]['X'] * scale)
        y1 = blur.shape[0] -int(G_Del.nodes[u]['Y'] * scale)
        x2 = int(G_Del.nodes[v]['X'] * scale)
        y2 = blur.shape[0] -int(G_Del.nodes[v]['Y'] * scale)

        if x1 == x2 and y1 == y2:
            val = 255.0
        else:
            val = average_in_segment_rect_xy(blur, x1, y1, x2, y2, half_width=3)
            
        avg_intensity_normalized = val / 255.0    
        ink_lookup[(u, v)] = avg_intensity_normalized
        ink_lookup[(v, u)] = avg_intensity_normalized
    
    
    
    normalize = 0
    for u,v,data in G_Del.edges(data=True):
        normalize = max(normalize, data['weight'])

    for u,v,data in G_Del.edges(data=True):
        data['weight'] = data['weight'] / normalize
        
        data['weight'] = alpha * data['weight'] + (1.0 - alpha) * ink_lookup[(u,v)]

    Q0_P = []
    lookup_Q = defaultdict(dict)
    
    for i in range(len(nodes)):
        n = nodes[i]
        
        RX = list(nx.bfs_tree(G_Del, source=n[0], depth_limit=2).edges())
        
        if len(RX) < 1:
            continue
        
        mx = 1000000
        for u,v in RX: 
            mx = min(G_Del[u][v]['weight'], mx)
            
        RXX = []
        for u,v in RX:
            RXX.append((u,v, G_Del[u][v]['weight'] / mx))
            
            if u == n[0]:
                lookup_Q[n[0]][v] = G_Del[u][v]['weight'] / mx
            elif v == n[0]:
                lookup_Q[n[0]][u] = G_Del[u][v]['weight'] / mx

            
        for neigh in list(G_Del.neighbors(n[0])):
            Q = lookup_Q[n[0]][neigh]
            Q0_P.append((n[0], neigh, Q))
            
    Q1_S = sorted(Q0_P, key=lambda x: x[2])
    
    mean_x = 0
    for u,v,w in Q1_S:
        mean_x += w
    mean_x = mean_x / len(Q1_S)
    
    
    W_X = []
    for u,v,w in Q1_S:
        W_X.append(w / mean_x)
        
    Q1_P = [0]
    
    for j in range(1, len(Q1_S) - 1):
        Q1_P.append((Q1_S[j+1][2] - Q1_S[j][2]) / 2)
    Q1_P.append(Q1_P[-1])
        
        
    Q2_PP = [0]
    for j in range(1, len(Q1_P)-1):
        Q2_PP.append((Q1_S[j+1][2] + Q1_S[j-1][2] - 2 * Q1_S[j][2]) / 4)
    Q2_PP.append(Q2_PP[-1])
        
    Q_ID2 = []
    for i in range(len(Q1_S)):
        Q_ID2.append((Q2_PP[i] * Q1_S[i][2]**2 / mean_x))
        
    Q_ID = []
    for i in range(len(Q1_P)):
        Q_ID.append((Q1_P[i] * W_X[i]))
        
    for i in range(len(Q_ID)):
        if Q_ID[i] > 0.12:
            break
    w1 = Q1_S[i][2]
    

    for i in range(len(Q_ID2)):
        if Q_ID2[i] > 0.025 * 50 /len(nodes):
            break
    w2 = Q1_S[i][2]
    w2 = Q1_S[int(len(Q1_S) * 0.65)][2]
  
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            n1 = nodes[i][0]
            n2 = nodes[j][0]
            
            if G_Del.has_edge(n1, n2):
                if lookup_Q[n1][n2] > w2 or lookup_Q[n2][n1] > w2:
                    G_Del.remove_edge(n1, n2)
               
    CC = list(nx.connected_components(G_Del))
        
    areas = []
    perimeters = []
    
    j = 1
    for i, cc in enumerate(CC):
        
        if len(cc) > 5:
            j += 1
        
        pts = []
        for n in cc:
            G.nodes[n]['cluster'] = i

            pts.append([G.nodes[n]['X'], G.nodes[n]['Y']])
    
        if len(cc) <= 3:
            continue
            
        pts = np.array(pts)
        
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        
        # perimeter (closed polygon)
        diffs = np.diff(np.vstack([hull_pts, hull_pts[0]]), axis=0)
        edge_lengths = np.linalg.norm(diffs, axis=1)
        perimeter = float(edge_lengths.sum())

        # area via shoelace formula
        x = hull_pts[:, 0]
        y = hull_pts[:, 1]
        area = 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) -
                                  np.dot(y, np.roll(x, -1))))
        areas.append(area)
        perimeters.append(perimeter)

    print(f"Detected {j} clusters.")
    print("Cluster areas:", np.mean(areas))
    print("Cluster perimeters:", np.mean(perimeters))


    return CC, j, areas, perimeters

def process(Bundling):
    G = Bundling.G
    Bundling.draw('clustering.png', draw_nodes=False, color=False) 
    
    CC, j, areas, perimeters  = clustering_gestalt(G, 'clustering.png')   
    
    return CC, j, areas, perimeters


