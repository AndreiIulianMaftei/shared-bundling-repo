import networkx as nx 

G = nx.random_geometric_graph(100,0.2)

print(G.nodes[0]['pos'])
for v in G.nodes():
    x,y = G.nodes[v]['pos']
    G.nodes[v]['x'] = float(x)
    G.nodes[v]['y'] = float(y)
    del G.nodes[v]['pos']

nx.write_graphml(G,"test.graphml")