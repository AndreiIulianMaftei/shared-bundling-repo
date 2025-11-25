import cv2
import numpy as np
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt


# ---------- basic preprocessing ----------

def preprocess_for_bundles(img_bgr: np.ndarray, dilate_iters: int = 1) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

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

    # visualization (in small image coords)
    h, w = skel.shape
    vis_small = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for edge in edges:
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        for y, x in edge:
            vis_small[y, x] = color

    # scale back up just for display
    vis_full = cv2.resize(
        vis_small,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    print("Bundle count (coarse):", bundle_count)
    print("Average bundle thickness (original pixels):", avg_thick)

    return bundle_count, vis_full, thicknesses


# ---------- your hook for side-by-side display ----------

def process_image(img_bgr: np.ndarray) -> np.ndarray:
    bundle_count, vis, thicknesses = count_bundles_coarse(
        img_bgr,
        scale=0.3,        # tweak: smaller -> coarser, fewer bundles
        dilate_iters=1,
        min_edge_length=10
    )
    print("Bundles:", bundle_count)
    return vis


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
    image_path = "migration/migration/migration/epb.png"   # <-- change this
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    processed = process_image(img)
    show_side_by_side(img, processed)