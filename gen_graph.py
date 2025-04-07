import networkx as nx 
import numpy as np

def gen_graphs(index):
    G = nx.random_geometric_graph(200,0.2)

    print(G.nodes[0]['pos'])
    for v in G.nodes():
        x,y = G.nodes[v]['pos']
        G.nodes[v]['x'] = float(x)
        G.nodes[v]['y'] = float(y)
        G.nodes[v]['X'] = float(x)
        G.nodes[v]['Y'] = float(y)
        del G.nodes[v]['pos']

    nx.write_graphml(G,f"inputs/g{index}.graphml")

def blockgraph():
    """
    Processes a graph from a GraphML file, computes shortest path distances, 
    applies t-SNE for dimensionality reduction, and saves the updated graph 
    with new node attributes.
    The function performs the following steps:
    1. Reads a graph from "input_data/block5_n1000.graphml".
    2. Converts node labels to integers for consistency.
    3. Computes the shortest path lengths between all pairs of nodes.
    4. Constructs a distance matrix based on the shortest path lengths.
    5. Applies t-SNE to reduce the dimensionality of the distance matrix.
    6. Updates the graph nodes with the t-SNE coordinates ('x' and 'y').
    7. Writes the updated graph to "inputs/block.graphml".
    Dependencies:
        - NetworkX (imported as nx)
        - NumPy (imported as np)
        - scikit-learn (TSNE from sklearn.manifold)
    Raises:
        FileNotFoundError: If the input GraphML file does not exist.
        ValueError: If t-SNE fails due to invalid input data.
    Notes:
        - The t-SNE initialization is set to 'random'.
        - The t-SNE metric is set to 'precomputed', as it uses the distance matrix.
    """
    G = nx.read_graphml("input_data/block5_n1000.graphml")
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()

    paths = nx.all_pairs_shortest_path_length(G)

    d = np.zeros((n,n),dtype=np.float32)

    for s, targets in paths:
        for t, distance in targets.items():
            d[s,t] = distance

    from sklearn.manifold import TSNE
    X = TSNE(metric='precomputed',init='random').fit_transform(d)

    for i,v in enumerate(G.nodes()):
        G.nodes()[v]['x'] = float(X[i,0])
        G.nodes()[v]['y'] = float(X[i,1])

    nx.write_graphml(G,"inputs/block.graphml")

def circular_graph(index, n=100, d=2):
    G = nx.barabasi_albert_graph(n, int(d))

    pos = nx.nx_agraph.graphviz_layout(G, 'circo', args='-Goneblock=true')

    for n, data in G.nodes(data=True):
        data['x'] = pos[n][0]
        data['y'] = pos[n][1]
        data['X'] = pos[n][0]
        data['Y'] = pos[n][1]

    nx.write_graphml(G,f"inputs/c{index}.graphml")

if __name__ == "__main__":
    blockgraph()