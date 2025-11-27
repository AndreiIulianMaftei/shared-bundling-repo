#!/usr/bin/env python3
"""
plot_trl_by_pairs.py

Reads a .trl file where each line has the format:
    EDGE_ID: x0 y0 x1 y1 x2 y2 ...

We parse each line into a list of (x, y) tuples and plot all edges in matplotlib.

Usage:
    python plot_trl_by_pairs.py <filename.trl>

Requires:
    pip install matplotlib
"""

import sys
import matplotlib.pyplot as plt

def read_trl(filename):
    """
    Reads the .trl file line by line.
    Each line looks like:
       0: 593.901123 279.166687 607.859253 285.851654 ...
    We ignore the edge ID, parse pairs of floats, and form (x, y) points.
    Returns a list of edges; each edge is a list of (x, y) tuples.
    """
    ok = 1
    edges = []
    with open(filename, 'r') as f:
        for line in f:
            
            
            ok = 0
            

            line = line.strip()
            if not line:
                continue  # skip empty lines

            tokens = line.split()
            if len(tokens) < 3:
                # We expect at least "0:" plus two floats => 3 tokens
                # so skip or warn if there's not enough data
                continue

            # First token is something like "0:" or "2:"
            edge_id_str = tokens[0].replace(":", "")  # e.g. "0"
            # edge_id = int(edge_id_str)  # If you'd like to store or use the edge ID

            # The rest are an even number of float strings:
            float_tokens = tokens[1:]  # skip the "0:" part
            if len(float_tokens) % 2 != 0:
                # If it's not even, there's a parsing mismatch
                continue

            coords = []
            for i in range(0, len(float_tokens), 2):
                x_str = float_tokens[i]
                y_str = float_tokens[i + 1]
                x = float(x_str)
                y = float(y_str)
                coords.append((x, y))

            if coords:
                edges.append(coords)

    return edges

def plot_trl(filename):
    """
    Reads the .trl file and plots the edges.
    """
    edges = read_trl(filename)

    plt.figure(figsize=(6, 6))
    plt.title(f"Plot of {filename}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')

    # Plot each edge as a connected polyline
    for edge in edges:
        xs, ys = zip(*edge)
        plt.plot(xs, ys, color='black', linewidth=0.8)
    plt.savefig("plot.png")
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_trl_by_pairs.py <filename.trl>")
        sys.exit(1)

    trl_file = sys.argv[1]
    plot_trl(trl_file)
