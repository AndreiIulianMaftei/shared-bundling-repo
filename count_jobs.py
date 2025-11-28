#!/usr/bin/env python3
"""
Helper script to count the number of graphs in the input folder.
Use this to determine the array size for SLURM job submission.
"""

import os
import sys

def count_graphs(folder_path):
    """Count the number of graph directories in the input folder."""
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found")
        return -1
    
    graph_dirs = [d for d in os.listdir(folder_path) 
                  if os.path.isdir(os.path.join(folder_path, d))]
    
    num_graphs = len(graph_dirs)
    
    print(f"Found {num_graphs} graphs in '{folder_path}'")
    print(f"\nFor SLURM array job, use:")
    print(f"#SBATCH --array=0-{num_graphs-1}")
    print(f"\nOr when submitting:")
    print(f"sbatch --array=0-{num_graphs-1} run_metrics_parallel.sh")
    
    print(f"\nGraph directories:")
    for i, graph in enumerate(sorted(graph_dirs)):
        print(f"  {i}: {graph}")
    
    return num_graphs

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "inputs/all_outputs/"
    count_graphs(folder)
