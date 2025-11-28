import numpy as np
import networkx as nx
import cv2
from sklearn.cluster import DBSCAN

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

def process(G, img_path, scale=.25):
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.resize(gray, (0,0), fx=scale, fy=scale)
    blur = cv2.GaussianBlur(blur, (5, 5), 0)


    nodes = list(G.nodes())

    euclidean_distances = np.zeros((len(nodes), len(nodes)))
    ink_distances = np.zeros((len(nodes), len(nodes)))

    for i1 in range(len(nodes)):
        for i2 in range(i1, len(nodes)):
            n1 = nodes[i1]
            n2 = nodes[i2]

            dist = np.sqrt((G.nodes[n1]['X'] - G.nodes[n2]['X'])**2 + (G.nodes[n1]['Y'] - G.nodes[n2]['Y'])**2)
            euclidean_distances[i1, i2] = dist
            euclidean_distances[i2, i1] = dist

            x1 = int(G.nodes[n1]['X'] * scale)
            y1 = int(G.nodes[n1]['Y'] * scale)
            x2 = int(G.nodes[n2]['X'] * scale)
            y2 = int(G.nodes[n2]['Y'] * scale)

            pts = bresenham_line(x1, y1, x2, y2)
            intensity_sum = 0
            for (x, y) in pts:
                if 0 <= x < blur.shape[0] and 0 <= y < blur.shape[1]:
                    intensity_sum += blur[x, y]

            avg_intensity = intensity_sum / len(pts) if len(pts) > 0 else 0
            avg_intensity_normalized = avg_intensity / 255.0
            ink_distances[i1, i2] = avg_intensity_normalized
            ink_distances[i2, i1] = avg_intensity_normalized

    geo_norm = euclidean_distances / np.sqrt(2 * (image.shape[0]**2 + image.shape[1]**2))

    alpha = 0.5  # weight between geometry and ink, tune as you like
    combined_distances = alpha * geo_norm + (1.0 - alpha) * ink_distances

    eps = 0.3
    min_samples = 3

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(combined_distances)

    for i, n in enumerate(nodes):
        G.nodes[n]['cluster'] = int(labels[i])

    count = np.unique((labels[labels >= 0]))
    print(f"Detected {len(count)} clusters.")

    return G


if __name__ == "__main__":
    # TODO: change this to your file
    image_path = "migration/migration/migration/cubu_1.png"
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not load image at path: {image_path}")
    
    G = nx.read_graphml('outputs/migration/cubu_1.graphml')

    process(G, image_path)

