"""
bundle_clustering_local.py

Cluster polylines stored in a NetworkX graph into bundles based on
local geometric overlap (close + parallel over a significant length).

Each edge in the graph must contain attributes:
    - 'X': list/array of x-coordinates
    - 'Y': list/array of y-coordinates

After clustering, each edge is assigned an integer "bundle_id".
"""

import numpy as np
import networkx as nx

from metrics_pipeline import read_bundling

try:
    import cv2  # optional for visualization
except ImportError:
    cv2 = None


# =========================================================
# 1. Extract polylines from networkx graph
# =========================================================

def extract_edge_polylines(G):
    """
    Extract polylines for all edges from a networkx graph.

    Returns:
        edges: list of np.ndarray (shape (N, 2))
        edge_ids: list of edge identifiers
    """
    edges = []
    edge_ids = []

    if G.is_multigraph():
        it = G.edges(keys=True, data=True)
        for u, v, k, data in it:
            if "X" not in data or "Y" not in data:
                continue
            X = np.asarray(data["X"], dtype=float)
            Y = np.asarray(data["Y"], dtype=float)
            if X.shape != Y.shape:
                raise ValueError(f"Edge {(u, v, k)} has mismatched X/Y lengths.")
            pts = np.stack([X, Y], axis=1)
            edges.append(pts)
            edge_ids.append((u, v, k))
    else:
        it = G.edges(data=True)
        for u, v, data in it:
            if "X" not in data or "Y" not in data:
                continue
            X = np.asarray(data["X"], dtype=float)
            Y = np.asarray(data["Y"], dtype=float)
            if X.shape != Y.shape:
                raise ValueError(f"Edge {(u, v)} has mismatched X/Y lengths.")
            pts = np.stack([X, Y], axis=1)
            edges.append(pts)
            edge_ids.append((u, v))

    return edges, edge_ids


# =========================================================
# 2. Tangent + angle utility
# =========================================================

def tangent(pts, i, neighborhood=3):
    """
    Estimate tangent vector at index i on polyline pts using a neighborhood.
    Returns a unit vector.
    """
    i0 = max(0, i - neighborhood)
    i1 = min(len(pts) - 1, i + neighborhood)
    v = pts[i1] - pts[i0]
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([1.0, 0.0])


def angle_between(v1, v2):
    """
    Compute angle in degrees between two vectors.
    """
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 180.0
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# =========================================================
# 3. Local overlap test (the core of bundle detection)
# =========================================================

def edges_share_bundle_segment(
    P: np.ndarray,
    Q: np.ndarray,
    pos_thresh: float = 5.0,
    angle_thresh_deg: float = 20.0,
    min_overlap_length: float = 20.0,
    neighborhood: int = 3,
) -> bool:
    """
    Determine whether polylines P and Q share a significant, contiguous,
    locally parallel segment.

    Works for different sample counts and nonuniform sampling.

    Parameters
    ----------
    P, Q : np.ndarray
        Polylines with shapes (NP,2) and (NQ,2)
    pos_thresh : float
        Spatial distance tolerance
    angle_thresh_deg : float
        Maximum allowed angle difference between tangents
    min_overlap_length : float
        Minimum physical length (sum of segment lengths) that must overlap
    neighborhood : int
        Number of points used for tangent estimation

    Returns
    -------
    bool
        True if bundles overlap enough, False otherwise
    """
    NP, NQ = len(P), len(Q)
    if NP < 2 or NQ < 2:
        return False

    # Precompute tangents
    dirs_P = np.array([tangent(P, i, neighborhood) for i in range(NP)])
    dirs_Q = np.array([tangent(Q, j, neighborhood) for j in range(NQ)])

    # --- P -> Q nearest neighbor mapping ---
    mask_P = np.zeros(NP, dtype=bool)
    for i in range(NP):
        dists = np.linalg.norm(Q - P[i], axis=1)
        j = int(np.argmin(dists))
        if dists[j] <= pos_thresh:
            ang = angle_between(dirs_P[i], dirs_Q[j])
            if ang <= angle_thresh_deg or abs(ang - 180) <= angle_thresh_deg:
                mask_P[i] = True

    # Compute longest contiguous overlap *length* along P
    best_len_P = 0.0
    cur_len = 0.0
    for i in range(1, NP):
        if mask_P[i] and mask_P[i - 1]:
            seg = np.linalg.norm(P[i] - P[i - 1])
            cur_len += seg
            best_len_P = max(best_len_P, cur_len)
        else:
            cur_len = 0.0

    # --- Q -> P nearest neighbor mapping ---
    mask_Q = np.zeros(NQ, dtype=bool)
    for j in range(NQ):
        dists = np.linalg.norm(P - Q[j], axis=1)
        i = int(np.argmin(dists))
        if dists[i] <= pos_thresh:
            ang = angle_between(dirs_Q[j], dirs_P[i])
            if ang <= angle_thresh_deg or abs(ang - 180) <= angle_thresh_deg:
                mask_Q[j] = True

    # Compute longest contiguous overlap *length* along Q
    best_len_Q = 0.0
    cur_len = 0.0
    for j in range(1, NQ):
        if mask_Q[j] and mask_Q[j - 1]:
            seg = np.linalg.norm(Q[j] - Q[j - 1])
            cur_len += seg
            best_len_Q = max(best_len_Q, cur_len)
        else:
            cur_len = 0.0

    # Require both curves to have significant overlap
    return (best_len_P >= min_overlap_length and
            best_len_Q >= min_overlap_length)


# =========================================================
# 4. Clustering (union–find / disjoint-set)
# =========================================================

def cluster_edges_by_local_overlap(
    edges,
    pos_thresh: float = 5.0,
    angle_thresh_deg: float = 20.0,
    min_overlap_length: float = 20.0,
    neighborhood: int = 3,
):
    """
    Given a list of polylines, cluster them based on local geometric overlap.

    Returns:
        labels: np.ndarray with bundle IDs
    """
    M = len(edges)
    if M == 0:
        return np.array([], dtype=int)

    parent = np.arange(M)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Compare each pair (O(M^2), but fast for M ~ few thousand)
    for i in range(M):
        for j in range(i + 1, M):
            if edges_share_bundle_segment(
                edges[i], edges[j],
                pos_thresh=pos_thresh,
                angle_thresh_deg=angle_thresh_deg,
                min_overlap_length=min_overlap_length,
                neighborhood=neighborhood,
            ):
                union(i, j)

    # Compress cluster IDs
    root_to_cluster = {}
    labels = np.zeros(M, dtype=int)
    next_id = 0
    for i in range(M):
        r = find(i)
        if r not in root_to_cluster:
            root_to_cluster[r] = next_id
            next_id += 1
        labels[i] = root_to_cluster[r]

    return labels


# =========================================================
# 5. Assign bundle IDs back to graph
# =========================================================

def assign_bundle_ids_local(
    G,
    pos_thresh: float = 5.0,
    angle_thresh_deg: float = 20.0,
    min_overlap_length: float = 20.0,
    neighborhood: int = 3,
    edge_attr_name: str = "bundle_id",
):
    """
    End-to-end pipeline:
      - Extract polylines from G
      - Cluster edges by local overlap
      - Write bundle IDs back to edges
    """
    edges, edge_ids = extract_edge_polylines(G)

    labels = cluster_edges_by_local_overlap(
        edges,
        pos_thresh=pos_thresh,
        angle_thresh_deg=angle_thresh_deg,
        min_overlap_length=min_overlap_length,
        neighborhood=neighborhood,
    )

    # Write bundle IDs back
    if G.is_multigraph():
        for (u, v, k), lab in zip(edge_ids, labels):
            G[u][v][k][edge_attr_name] = int(lab)
    else:
        for (u, v), lab in zip(edge_ids, labels):
            G[u][v][edge_attr_name] = int(lab)

    return labels, edge_ids


# =========================================================
# 6. Optional visualization
# =========================================================

def visualize_bundles(G, edge_attr_name="bundle_id", img_size=(800, 800), thickness=2):
    if cv2 is None:
        print("cv2 not installed; skipping visualization.")
        return None

    h, w = img_size
    vis = np.ones((h, w, 3), dtype=np.uint8) * 255  # white background
    rng = np.random.default_rng(42)

    # assign random color per bundle
    bundle_ids = set()
    for _, _, data in G.edges(data=True):
        if edge_attr_name in data:
            bundle_ids.add(data[edge_attr_name])

    bundle_ids = sorted(bundle_ids)
    colors = {bid: rng.integers(0, 255, size=3).tolist() for bid in bundle_ids}

    # draw polylines
    for u, v, data in G.edges(data=True):
        if "X" not in data or "Y" not in data or edge_attr_name not in data:
            continue
        bid = data[edge_attr_name]
        color = colors.get(bid, [0, 0, 0])
        pts = np.column_stack([data["X"], data["Y"]]).astype(int)

        for i in range(len(pts) - 1):
            x1, y1 = pts[i]
            x2, y2 = pts[i + 1]
            cv2.line(vis, (x1, y1), (y2, x2), color, thickness=thickness)

    return vis


# =========================================================
# 7. Example usage (edit for your actual graph)
# =========================================================

if __name__ == "__main__":
    # Example: load your graph here
    # G = nx.read_gpickle("graph.gpickle")

    #G = nx.read_graphml('outputs/airlines/epb.graphml')  # placeholder
    G = read_bundling('outputs/airlines/epb.graphml').G
    
    print("Graph:", G.number_of_nodes(), "nodes,", G.number_of_edges(), "edges")

    labels, edge_ids = assign_bundle_ids_local(
        G,
        pos_thresh=5.0,          # distance tolerance
        angle_thresh_deg=20.0,   # angular tolerance
        min_overlap_length=20.0, # minimum overlap length
        neighborhood=3,
        edge_attr_name="bundle_id",
    )

    print("Clusters found:", len(np.unique(labels)) if len(labels) else 0)

    if cv2 is not None:
        vis = visualize_bundles(G)
        if vis is not None:
            cv2.imwrite("bundles.png", vis)
            print("Saved bundle visualization as bundles.png")
