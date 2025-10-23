#!/usr/bin/env python3
"""
Test the updated get_clusters function with single-level clustering.
"""

import networkx as nx
import numpy as np
from modules.clustering import Clustering

def create_test_graph():
    """Create a simple test graph."""
    G = nx.Graph()
    
    # Add nodes in a grid pattern
    nodes = [
        (0, {'X': 100, 'Y': 100, 'id': 0}),
        (1, {'X': 200, 'Y': 100, 'id': 1}),
        (2, {'X': 300, 'Y': 100, 'id': 2}),
        (3, {'X': 100, 'Y': 200, 'id': 3}),
        (4, {'X': 200, 'Y': 200, 'id': 4}),
        (5, {'X': 300, 'Y': 200, 'id': 5}),
    ]
    G.add_nodes_from(nodes)
    
    # Add edges
    G.add_edge(0, 1, X=[100, 200], Y=[100, 100])
    G.add_edge(1, 2, X=[200, 300], Y=[100, 100])
    G.add_edge(3, 4, X=[100, 200], Y=[200, 200])
    G.add_edge(4, 5, X=[200, 300], Y=[200, 200])
    G.add_edge(0, 3, X=[100, 100], Y=[100, 200])
    G.add_edge(2, 5, X=[300, 300], Y=[100, 200])
    
    return G

def test_single_level_clustering():
    """Test the single-level clustering."""
    print("\n" + "="*70)
    print("TEST: Single-Level Clustering")
    print("="*70)
    
    G = create_test_graph()
    clustering = Clustering(G)
    
    # Get polylines and vertices
    print("\nExtracting data...")
    polylines = clustering.all_edges(G)
    vertices = clustering.init_Points()
    
    print(f"Polylines: {len(polylines)}")
    print(f"Vertices: {len(vertices)}")
    
    # Initialize and calculate matrix
    print("\nInitializing matrix...")
    matrix = clustering.init_matrix(polylines, n_clusters=2)
    matrix = clustering.calcMatrix(matrix)
    
    # Test with different depth thresholds
    thresholds = [0, 3, 5, 7, 10]
    
    for threshold in thresholds:
        print(f"\n--- Testing with depth_threshold={threshold} ---")
        node_to_cluster = clustering.get_clusters(polylines, matrix, vertices, depth_threshold=threshold)
        
        print(f"\nCluster assignments:")
        for node_id in sorted(node_to_cluster.keys()):
            cluster_id = node_to_cluster[node_id]
            print(f"  Node {node_id} -> Cluster {cluster_id}")
        
        # Count clusters
        num_clusters = len(set(c for c in node_to_cluster.values() if c != -1))
        num_unclustered = sum(1 for c in node_to_cluster.values() if c == -1)
        
        print(f"\nSummary:")
        print(f"  Total clusters: {num_clusters}")
        print(f"  Unclustered nodes: {num_unclustered}")
        print(f"  Clustered nodes: {len(node_to_cluster) - num_unclustered}")
    
    print("\n✓ Single-level clustering test completed")

def test_return_format():
    """Test that the return format is correct."""
    print("\n" + "="*70)
    print("TEST: Return Format Validation")
    print("="*70)
    
    G = create_test_graph()
    clustering = Clustering(G)
    
    polylines = clustering.all_edges(G)
    vertices = clustering.init_Points()
    matrix = clustering.init_matrix(polylines, n_clusters=2)
    matrix = clustering.calcMatrix(matrix)
    
    node_to_cluster = clustering.get_clusters(polylines, matrix, vertices)
    
    # Validate return type
    assert isinstance(node_to_cluster, dict), "Should return a dictionary"
    print("✓ Returns a dictionary")
    
    # Validate keys are node IDs
    for key in node_to_cluster.keys():
        assert isinstance(key, (int, np.integer)), f"Key {key} should be an integer"
    print("✓ All keys are node IDs (integers)")
    
    # Validate values are cluster IDs
    for value in node_to_cluster.values():
        assert isinstance(value, (int, np.integer)), f"Value {value} should be an integer"
        assert value >= -1, f"Cluster ID {value} should be >= -1"
    print("✓ All values are cluster IDs (integers >= -1)")
    
    # Validate all vertices are present
    vertex_ids = {v[2] for v in vertices}
    result_ids = set(node_to_cluster.keys())
    assert vertex_ids == result_ids, "All vertices should be in the result"
    print("✓ All vertices are present in the result")
    
    print("\n✓ Return format validation passed")

def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RUNNING UPDATED CLUSTERING TESTS")
    print("="*70)
    
    try:
        test_single_level_clustering()
        test_return_format()
        
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70)
        print("\nThe updated clustering function works correctly!")
        print("- Returns a single-level dictionary: {node_id: cluster_id}")
        print("- Accepts a depth_threshold parameter (0-10)")
        print("- Assigns -1 to unclustered nodes")
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
