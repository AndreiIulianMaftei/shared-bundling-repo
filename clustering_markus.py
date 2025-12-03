from collections import defaultdict
import numpy as np
import networkx as nx
import cv2
from sklearn.cluster import DBSCAN, HDBSCAN
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

from metrics_pipeline import read_bundling

MST_NEIGHBORHOOD_DEPTH = 5
MST_CUTOFF = 1.5


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

def bresenham_line(x0, y0, x1, y1):
    """
    Classic Bresenham line algorithm.
    Returns list of integer (x, y) pixel coordinates from (x0, y0) to (x1, y1).
    """
    points = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

    return points

def debug_segment_rectangle_xy(img, x1, y1, x2, y2, half_width,
                               out_path=None, show=True, vis = None):
    """
    Draw rectangle + endpoints on the SAME image using x,y coords.

    img: same array you used for averaging (no extra scaling)
    x1,y1,x2,y2 in x=left-right, y=top-bottom coords.
    """
    vis = vis.copy()

    vx, vy = x2 - x1, y2 - y1
    L = np.hypot(vx, vy)
    if L == 0:
        print("Zero-length segment.")
        return vis, None

    ux, uy = vx / L, vy / L
    nx, ny = -uy, ux

    c1 = np.array([x1, y1], float)  # (x,y)
    c2 = np.array([x2, y2], float)
    perp = half_width * np.array([nx, ny], float)

    # corners in x,y
    A = c1 + perp
    B = c2 + perp
    C = c2 - perp
    D = c1 - perp

    # convert to (col,row) = (x,y) integers for cv2
    poly = np.array(
        [[A[0], A[1]],
         [B[0], B[1]],
         [C[0], C[1]],
         [D[0], D[1]]],
        dtype=np.int32
    ).reshape((-1, 1, 2))

    # draw rectangle + endpoints
    cv2.polylines(vis, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.circle(vis, (int(x1), int(y1)), 4, (0, 255, 0), -1)
    cv2.circle(vis, (int(x2), int(y2)), 4, (0, 255, 0), -1)

    avg = average_in_segment_rect_xy(img, x1, y1, x2, y2, half_width)
    if avg is not None and np.isscalar(avg):
        txt = f"avg={avg:.1f}"
    else:
        txt = f"avg={avg}"  # e.g. 3-channel

    cv2.putText(vis, txt,
                (int(x1) + 10, int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if out_path is not None:
        cv2.imwrite(out_path, vis)
        print(f"Saved debug image to {out_path}")

    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()
        plt.savefig("debug_rectangle.png", dpi=150)

    return avg

def corridor_pixels(x0, y0, x1, y1, half_width, img_shape):
    """
    Use Bresenham to get the center line, then thicken it into a (2*half_width+1)-pixel corridor.

    half_width = 1  -> 3-pixel-wide corridor
    img_shape: (height, width) of the image, for bounds checking.

    Returns a list of (x, y) pixels in the corridor.
    """
    h, w = img_shape
    line = bresenham_line(x0, y0, x1, y1)
    pts = set()

    # If line is more horizontal, expand vertically; if more vertical, expand horizontally
    if abs(x1 - x0) >= abs(y1 - y0):
        # horizontal-ish: vary y
        for x, y in line:
            for off in range(-half_width, half_width + 1):
                yy = y + off
                if 0 <= x < w and 0 <= yy < h:
                    pts.add((x, yy))
    else:
        # vertical-ish: vary x
        for x, y in line:
            for off in range(-half_width, half_width + 1):
                xx = x + off
                if 0 <= xx < w and 0 <= y < h:
                    pts.add((xx, y))

    return list(pts)


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


def average_in_segment_rect(img, x1, y1, x2, y2, half_width):
    """
    Compute the average pixel value in a rectangle whose LONG sides lie along
    the segment (x1,y1)-(x2,y2), and whose short side half-width is `half_width`.

    img: HxW or HxWxC NumPy array (grayscale or color)
    (x1,y1), (x2,y2): centers of the two opposite sides
    half_width: half the rectangle thickness (in pixels)
    """
    # Work in float for geometry
    x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))

    # Direction along the segment
    vx, vy = x2 - x1, y2 - y1
    length = (vx**2 + vy**2)**0.5
    if length == 0:
        return None  # or np.nan

    ux, uy = vx / length, vy / length  # unit vector along segment

    # Perpendicular unit vector
    nx, ny = -uy, ux

    H, W = img.shape[:2]

    # Bounding box around the rectangle
    min_x = int(np.floor(min(x1, x2) - half_width)) - 1
    max_x = int(np.ceil (max(x1, x2) + half_width)) + 1
    min_y = int(np.floor(min(y1, y2) - half_width)) - 1
    max_y = int(np.ceil (max(y1, y2) + half_width)) + 1

    min_x = max(min_x, 0)
    min_y = max(min_y, 0)
    max_x = min(max_x, W - 1)
    max_y = min(max_y, H - 1)

    # Grid of pixel centers in the bounding box
    xs = np.arange(min_x, max_x + 1)
    ys = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(xs, ys)

    # Vector from (x1,y1) to each point
    RX = X - x1
    RY = Y - y1

    # Projection along the segment (0 .. length)
    t = RX * ux + RY * uy

    # Signed distance to the segment (perpendicular)
    d = RX * nx + RY * ny

    # Mask of points inside the rectangle
    mask = (t >= 0) & (t <= length) & (np.abs(d) <= half_width)

    if not np.any(mask):
        return None  # or np.nan

    vals = img[min_y:max_y+1, min_x:max_x+1][mask]
    return float(vals.mean())

def process(G, img_path, scale=.25):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.resize(gray, (0,0), fx=scale, fy=scale)
    blur = cv2.GaussianBlur(blur, (7, 7), 0)
    
    # cv2.imshow("blur", blur)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(blur.shape)
    
    vis = blur
    blur = blur.astype(np.float32)
    
    

    nodes = list(G.nodes())

    euclidean_distances = np.zeros((len(nodes), len(nodes)))
    ink_distances = np.zeros((len(nodes), len(nodes)))

    # for i1 in range(len(nodes)):
    #     for i2 in range(i1+1, len(nodes)):
    #         n1 = nodes[i1]
    #         n2 = nodes[i2]

    #         dist = np.sqrt((G.nodes[n1]['X'] - G.nodes[n2]['X'])**2 + (G.nodes[n1]['Y'] - G.nodes[n2]['Y'])**2)
    #         euclidean_distances[i1, i2] = dist
    #         euclidean_distances[i2, i1] = dist

    #         x1 = int(G.nodes[n1]['X'] * scale)
    #         y1 = blur.shape[0] -int(G.nodes[n1]['Y'] * scale)
    #         x2 = int(G.nodes[n2]['X'] * scale)
    #         y2 = blur.shape[0] -int(G.nodes[n2]['Y'] * scale)

    #         # pts = corridor_pixels(x1, y1, x2, y2, half_width=1, img_shape=blur.shape)
    #         # pts = bresenham_line(x1, y1, x2, y2)
    #         # intensity_sum = 0
    #         # for (x, y) in pts:
    #         #     if 0 <= x < blur.shape[1] and 0 <= y < blur.shape[0]:
    #         #         if blur[y,x] < 240:
    #         #             intensity_sum += 1
    #                 #intensity_sum += blur[y, x] 
                    
    #         test = average_in_segment_rect_xy(blur, x1, y1, x2, y2, half_width=3)
            
    #         # _, avg, rect = debug_segment_rectangle_xy(
    #         #     blur,
    #         #     x1=x1, y1=y1,
    #         #     x2=x2, y2=y2,
    #         #     half_width=3, show=True, vis = vis
    #         # )

    #         # print("Average pixel value:", avg)
    #         # print("Rectangle corners:", rect)

    #         # avg_intensity = intensity_sum / len(pts) if len(pts) > 0 else 1
    #         #avg_intensity_normalized = avg_intensity / 255.0
    #         avg_intensity_normalized = test / 255.0
    #         ink_distances[i1, i2] = avg_intensity_normalized
    #         ink_distances[i2, i1] = avg_intensity_normalized

    #ink_distances = ink_distances / np.max(ink_distances)
    # ink_distances = (ink_distances - np.min(ink_distances)) / (np.max(ink_distances) - np.min(ink_distances))
    # #ink_distances = 1.0 - ink_distances  # higher ink -> lower distance
    # max_e = np.max(euclidean_distances)
    # #geo_norm = euclidean_distances / np.sqrt(2 * (image.shape[0]**2 + image.shape[1]**2))
    # # geo_norm = geo_norm / max_e
    # geo_norm = (euclidean_distances - np.min(euclidean_distances)) / (np.max(euclidean_distances) - np.min(euclidean_distances))
    # alpha = 1 # weight between geometry and ink, tune as you like
    # combined_distances = alpha * geo_norm + (1.0 - alpha) * ink_distances

    # eps = 0.05
    # min_samples = 3

    # db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    # labels = db.fit_predict(combined_distances)
    
    G_Del, tri, pts, index_to_node = delaunay_graph(G)
    alpha = 0.75
    
    weights = []
    for u,v,data in G_Del.edges(data=True):
        weights.append(data['weight'])
    mean_w = np.mean(weights)
    std_w = np.std(weights)
    
    print(f"Delaunay edges before length-based removal: {G_Del.number_of_edges()}")
    to_remove = []
    for u,v,data in G_Del.edges(data=True):
        if data['weight'] < 5:
            to_remove.append((u,v))
    
    for u,v in to_remove:
        G_Del.remove_edge(u,v)
        
    print(f"Delaunay edges after length-based removal: {G_Del.number_of_edges()}")
    
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
            # val = debug_segment_rectangle_xy(blur, x1, y1, x2, y2, half_width=3, vis=vis)
        avg_intensity_normalized = val / 255.0    
        ink_lookup[(u, v)] = avg_intensity_normalized
        ink_lookup[(v, u)] = avg_intensity_normalized
    
    
    
    normalize = 0
    min_ink = 1000
    max_ink = -1000
    for u,v,data in G_Del.edges(data=True):
        normalize = max(normalize, data['weight'])
        min_ink = min(min_ink, ink_lookup[(u,v)])
        max_ink = max(max_ink, ink_lookup[(u,v)])
            
    for u,v,data in G_Del.edges(data=True):
        data['weight'] = data['weight'] / normalize
        #ink_lookup[(u,v)] = (ink_lookup[(u,v)] - min_ink) / (max_ink - min_ink)
        
        data['weight'] = alpha * data['weight'] + (1.0 - alpha) * ink_lookup[(u,v)]

    Q0_P = []
    Q0_PP = []
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
    w2 = Q1_S[int(len(Q1_S) * 0.60)][2]

    print(f"Edges before removal: {G_Del.number_of_edges()}")
    
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            n1 = nodes[i][0]
            n2 = nodes[j][0]
            
            if G_Del.has_edge(n1, n2):
                if lookup_Q[n1][n2] > w2 or lookup_Q[n2][n1] > w2:
                    G_Del.remove_edge(n1, n2)
                    
    print(f"Edges after removal: {G_Del.number_of_edges()}")
                
                
        
    # Q1 = sorted(Q0, key=lambda x: x[2])
    
    # for u,v,w in Q1:
    #     if w > 1.65:
    #         if G_Del.has_edge(u,v):
    #             G_Del.remove_edge(u,v)
                
    CC = list(nx.connected_components(G_Del))
        
    j = 1
    for i, cc in enumerate(CC):
        
        if len(cc) > 5:
            j += 1
        
        for n in cc:
            if len(cc) > 5:
                G.nodes[n]['cluster'] = i
            else:
                G.nodes[n]['cluster'] = 100

    # G_Compl = nx.Graph()
    # for i in range(len(nodes)):
    #     for j in range(i + 1, len(nodes)):
    #         G_Compl.add_edge(i, j, weight=combined_distances[i,j])
            
    # MST = nx.minimum_spanning_tree(G_Compl)
    
    # to_remove = []
    
    # edges = list(MST.edges(data=True))
    # edges = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)
    # for u,v, data in edges:
        
    #     weights = []
    #     visited = {u, v}
    #     queue = [(u, 0), (v, 0)]
        
    #     while queue:
    #         node, dist = queue.pop(0)
    #         if dist < MST_NEIGHBORHOOD_DEPTH:
    #             for neighbor in MST.neighbors(node):
    #                 if neighbor not in visited:
    #                     visited.add(neighbor)
    #                     queue.append((neighbor, dist + 1))
                        
    #                     weights.append(MST[node][neighbor]['weight'])
                                   
    #     mean_val = np.mean(weights)
    #     std = np.std(weights)
        
    #     if data['weight'] > mean_val + MST_CUTOFF * std:
    #         to_remove.append((u,v))
            
    # for u,v in to_remove:
    #     MST.remove_edge(u,v)
        
    # CC = nx.connected_components(MST)
    
    # j = 1
    # for i, cc in enumerate(CC):
    #     j += 1
    #     for n in cc:
    #         nn = nodes[n]
    #         G.nodes[nn]['cluster'] = i

    # for i, n in enumerate(nodes):
    #     G.nodes[n]['cluster'] = int(labels[i])

    # count = np.unique((labels[labels >= 0]))
    print(f"Detected {j} clusters.")

    MMST = nx.Graph()
    # for u,v in MST.edges():
    #     nn_u = nodes[u]
    #     nn_v = nodes[v]
    #     MMST.add_edge(nn_u, nn_v)

    # clustering = HDBSCAN(metric='precomputed', min_cluster_size=5)
    # cluster_labels = clustering.fit_predict(combined_distances)
    cluster_labels = []
    
    # for i, n in enumerate(nodes):
    #     G.nodes[n]['cluster_hdbscan'] = int(cluster_labels[i])
    
    # unique = np.unique(cluster_labels)
    # print(f"HDBSCAN detected {len(unique)} clusters.")

    return G_Del, cluster_labels


if __name__ == "__main__":
    # TODO: change this to your file

   
    Bundling = read_bundling('outputs_batch1/migration/epb.graphml')
    G = Bundling.G

    Bundling.draw(f'clustering', draw_nodes=False, color=False, plotSL=True)
    img_path = f'clustering{G.graph["name"]}.png'
    
    MST, cluster_labels = process(G, img_path)   
    Bundling.draw("bundled_cluster", color_vertices="cluster", plotSubgraph=MST, color=False)
    
    
    

