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