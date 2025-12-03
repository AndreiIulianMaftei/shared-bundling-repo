import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import networkx as nx

# ---------- basic preprocessing ----------

def preprocess_for_bundles(img_bgr: np.ndarray, dilate_iters: int = 1) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # dark lines -> white (255), bg -> 0
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    if dilate_iters > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.dilate(binary, kernel, iterations=dilate_iters)

    return binary


def skeleton_from_mask(bundle_mask: np.ndarray) -> np.ndarray:
    skel = skeletonize(bundle_mask > 0)
    return (skel.astype(np.uint8)) * 255


# ---------- graph construction on skeleton ----------

def compute_skeleton_degree(skel: np.ndarray) -> np.ndarray:
    skel_bin = (skel > 0).astype(np.uint8)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    degree = cv2.filter2D(skel_bin, -1, kernel)
    degree[skel_bin == 0] = 0
    return degree


def label_nodes(skel: np.ndarray, degree: np.ndarray):
    node_mask = ((degree != 2) & (degree > 0)).astype(np.uint8)
    num_labels, node_labels = cv2.connectedComponents(node_mask, connectivity=8)

    node_centers = {}
    for lab in range(1, num_labels):
        ys, xs = np.where(node_labels == lab)
        if len(ys) == 0:
            continue
        cy = int(ys.mean())
        cx = int(xs.mean())
        node_centers[lab] = (cy, cx)

    return node_labels, node_centers


def build_edges_from_skeleton(skel: np.ndarray,
                              node_labels: np.ndarray,
                              degree: np.ndarray,
                              min_length: int = 5):
    skel_bin = (skel > 0).astype(np.uint8)
    h, w = skel.shape

    visited = np.zeros_like(skel_bin, dtype=bool)
    edges = []

    neighbors = [(-1, -1), (-1, 0), (-1, 1),
                 (0, -1),           (0, 1),
                 (1, -1),  (1, 0),  (1, 1)]

    node_pixels = np.where((node_labels > 0) & (skel_bin > 0))
    node_coords = list(zip(node_pixels[0], node_pixels[1]))

    for sy, sx in node_coords:
        for dy, dx in neighbors:
            ny, nx = sy + dy, sx + dx
            if not (0 <= ny < h and 0 <= nx < w):
                continue
            if skel_bin[ny, nx] == 0 or visited[ny, nx]:
                continue

            path = [(sy, sx)]
            cy, cx = ny, nx
            prev_y, prev_x = sy, sx

            while True:
                path.append((cy, cx))
                visited[cy, cx] = True

                # hit a (different) node -> stop
                if node_labels[cy, cx] > 0 and not (cy == sy and cx == sx):
                    break

                if degree[cy, cx] != 2:
                    break  # spur / dead end

                next_candidates = []
                for ddy, ddx in neighbors:
                    ty, tx = cy + ddy, cx + ddx
                    if not (0 <= ty < h and 0 <= tx < w):
                        continue
                    if skel_bin[ty, tx] == 0:
                        continue
                    if ty == prev_y and tx == prev_x:
                        continue
                    next_candidates.append((ty, tx))

                if not next_candidates:
                    break

                ny2, nx2 = next_candidates[0]
                prev_y, prev_x = cy, cx
                cy, cx = ny2, nx2

            if len(path) >= min_length:
                edges.append(path)

    return edges

def cluster_node_centers(node_centers, merge_radius: float):
    """
    node_centers: dict[label] = (y, x)
    merge_radius: distance (in skeleton pixels) under which nodes become one.

    Returns:
      label_to_cluster: dict[original_label] -> cluster_id
      cluster_pos: dict[cluster_id] -> (cy, cx) mean position
    """
    labels = list(node_centers.keys())
    if not labels:
        return {}, {}

    coords = np.array([node_centers[l] for l in labels], dtype=float)  # shape (N,2)
    N = len(labels)

    # --- union–find for clustering ---
    parent = list(range(N))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    r2 = merge_radius * merge_radius
    for i in range(N):
        for j in range(i + 1, N):
            d2 = np.sum((coords[i] - coords[j]) ** 2)
            if d2 <= r2:
                union(i, j)

    # collect clusters
    root_to_indices = {}
    for i in range(N):
        r = find(i)
        root_to_indices.setdefault(r, []).append(i)

    label_to_cluster = {}
    cluster_pos = {}
    cid = 0
    for root, idxs in root_to_indices.items():
        pts = coords[idxs]
        cy, cx = pts.mean(axis=0)
        cluster_pos[cid] = (cy, cx)
        for i in idxs:
            lab = labels[i]
            label_to_cluster[lab] = cid
        cid += 1

    return label_to_cluster, cluster_pos


def skeleton_to_networkx(node_labels: np.ndarray,
                         node_centers: dict,
                         edges,
                         merge_radius: float = 3.0,
                         scale: float = 1.0):
    """
    node_labels: HxW int32 labels from label_nodes
    node_centers: dict[label] -> (y, x)
    edges: list of paths, each path is [(y, x), ...] from build_edges_from_skeleton
    merge_radius: how close endpoints must be to become one node (in skeleton pixels)
    scale: if the skeleton was computed on a downscaled image, set this
           to 1/scale_factor to get coordinates in original pixels.

    Returns:
      G: networkx.MultiGraph()
    """
    # 1) merge nearby node centers
    label_to_cluster, cluster_pos = cluster_node_centers(
        node_centers, merge_radius=merge_radius
    )

    G = nx.Graph()

    # 2) add merged nodes
    for cid, (cy, cx) in cluster_pos.items():
        G.add_node(
            cid,
            y=cy * scale,
            x=cx * scale,
            pos=(cx * scale, cy * scale),  # convenient for drawing
        )

    # 3) add one edge per path
    count = 0
    for path in edges:
        if len(path) < 2:
            continue

        sy, sx = path[0]
        ey, ex = path[-1]

        lab_start = node_labels[sy, sx]
        lab_end = node_labels[ey, ex]
        if lab_start <= 0 or lab_end <= 0:
            continue
        if lab_start not in label_to_cluster or lab_end not in label_to_cluster:
            continue

        u = label_to_cluster[lab_start]
        v = label_to_cluster[lab_end]

        if u == v:
            continue  # loop edge

        # coordinates in original image units (if scale != 1)
        coords = [(y, x) for (y, x) in path]
        length = float(len(coords))  # or use Euclidean sum along path

        G.add_edge(
            u,
            v,
            pixels=coords,
            length=length,
            index = count
        )

        count += 1

    return G

def compute_thickness_per_edge(bundle_mask: np.ndarray, edges):
    """
    bundle_mask: 0/255, 255 = bundle
    edges: list of polylines (each is a list of (y, x) skeleton points)

    Returns: list of thickness values (same length as edges),
             in pixels (of the image where you built the skeleton).
    """
    fg = (bundle_mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5)

    thicknesses = []
    for edge in edges:
        if not edge:
            thicknesses.append(0.0)
            continue

        ys = np.array([p[0] for p in edge], dtype=int)
        xs = np.array([p[1] for p in edge], dtype=int)

        vals = dist[ys, xs]
        vals = vals[vals > 0]

        if vals.size == 0:
            thicknesses.append(0.0)
        else:
            thickness = 2.0 * float(np.median(vals))  # radius -> thickness
            thicknesses.append(thickness)

    return thicknesses

def visualize_edges_on_image(img, G, thickness=1):
    """
    Draws each graph edge in a different random color on top of the original image.
    G must contain 'pixels' attribute per edge.
    """
    vis = img.copy()
    rng = np.random.default_rng(123)

    for u, v, data in G.edges(data=True):
        pixels = data["pixels"]   # list of (y, x) coordinates
        color = rng.integers(50, 255, size=3).tolist()
        color = (int(color[0]), int(color[1]), int(color[2]))

        cv2.line(vis, (int(pixels[0][1]), int(pixels[0][0])), (int(pixels[-1][1]), int(pixels[-1][0])), color, thickness)

    return vis

# ---------- coarse bundle counting ----------

def count_bundles_coarse(img_bgr: np.ndarray,
                         scale: float = 0.25,
                         dilate_iters: int = 1,
                         min_edge_length: int = 10):
    # downsample to coarsen bundles
    small = cv2.resize(
        img_bgr,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    bundle_mask = preprocess_for_bundles(small, dilate_iters=dilate_iters)
    skel = skeleton_from_mask(bundle_mask)
    degree = compute_skeleton_degree(skel)
    node_labels, node_centers = label_nodes(skel, degree)
    edges = build_edges_from_skeleton(
        skel, node_labels, degree, min_length=min_edge_length
    )
    bundle_count = len(edges)

    # thickness per edge at small scale
    thicknesses_small = compute_thickness_per_edge(bundle_mask, edges)
    # rescale thickness to original-image pixels
    thicknesses = [t / scale for t in thicknesses_small]
    avg_thick = float(np.mean(thicknesses)) if thicknesses else 0.0


    G = skeleton_to_networkx(
        node_labels,
        node_centers,
        edges,
        merge_radius=10.0,      # tweak this
        scale=1.0 / scale      # because we downsampled by 'scale'
    )

    setDot = []
    allPassed = False
    while not allPassed:
        
        allPassed = True

        for node in G.nodes():
            if G.degree[node] == 2:
                n1 = list(G.neighbors(node))[0]
                n2 = list(G.neighbors(node))[1]
                
                e1 = G.get_edge_data(node, n1)
                e2 = G.get_edge_data(node, n2)

                setDot.append((G.nodes[node]['x'], G.nodes[node]['y']))

                G.remove_edge(node, n1)
                G.remove_edge(node, n2)
                G.remove_node(node)

                coords = e1['pixels'] + e2['pixels']
                length = e1['length'] + e2['length']
                G.add_edge(n1, n2,             
                            pixels=coords,
                            length=length)
  
                allPassed = False
                break

    deg2 = 0

    edges = []
    for u,v, data in G.edges(data=True):
        edges.append(data['pixels'])

    print("Nodes with degree 2 (should be 0):", deg2)

    # now you can inspect G
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    # visualization (in small image coords)
    h, w = skel.shape
    vis_small = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for edge in edges:
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        for y, x in edge:
            vis_small[y, x] = color

    for x,y in setDot:
        cv2.circle(vis_small, (int(x*scale), int(y * scale)), 3, (0,0,255), 1)

    #
    # vis_edges = visualize_edges_on_image(vis_small, G, thickness=1)

    # scale back up just for display
    vis_full = cv2.resize(
        vis_small,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    print("Bundle count (coarse):", bundle_count)
    print("Average bundle thickness (original pixels):", avg_thick)

    return bundle_count, vis_full, thicknesses, edges


# ---------- your hook for side-by-side display ----------

def process_image(img_bgr: np.ndarray) -> np.ndarray:
    bundle_count, vis, thicknesses, edges = count_bundles_coarse(
        img_bgr,
        scale=0.5,        # tweak: smaller -> coarser, fewer bundles
        dilate_iters=3,
        min_edge_length=10
    )
    
    map_bundles_to_graph
    print("Bundles:", bundle_count)
    return bundle_count, vis, thicknesses, edges

def map_bundles_to_graph(G, shape, edges, scale = 0.25,):
    h, w = shape[0] * scale, shape[1] * scale

    ammap = np.zeros((int(w), int(h)), dtype=np.uint8)
    
    ambiguity = {}
    
    for i, edge in enumerate(edges):
        for y,x in edge:
            ammap[int(x), h - int(y)] = i + 1

        ambiguity[i + 1] = set()
        
    for u,v, data in G.edges(data=True):
        X = data['X']
        Y = data['Y']

        for x,y in zip(X, Y):
            xi = int(x * scale)
            yi = int(y * scale)
            
            amset = ammap[xi, yi]
            if amset > 0:
                ambiguity[amset].add((u,v))
                
    ambiguities = []
    for amset in ambiguity.items():
        endpoints = set()
        for u,v in amset:
            endpoints.add(u)
            endpoints.add(v)
            
        av = 0
        for u in endpoints:
            deg1 = set(G.neighbors(u))
            set.add(u)
            
            inter = deg1.intersection(endpoints)
            setminus = endpoints.difference(deg1)
            
            inter = len(inter)
            setminus = len(setminus)
            total = len(endpoints)
            
            av += inter / total + setminus / (2 * total)
            
        ambiguities.append(av / len(endpoints))
        
    return ambiguities

def show_side_by_side(original, processed, title="Original | Processed"):
    h1, w1 = original.shape[:2]
    h2, w2 = processed.shape[:2]

    if h1 != h2:
        scale = h1 / h2
        new_w = int(w2 * scale)
        processed = cv2.resize(processed, (new_w, h1), interpolation=cv2.INTER_AREA)

    # convert BGR->RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(processed_rgb)
    axes[1].set_title("Processed")
    axes[1].axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "image_outputs/airlines/cubu_1.png"   # <-- change this
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    processed = process_image(img)
    show_side_by_side(img, processed)