import numpy as np 
import networkx as nx 
from sklearn.metrics import silhouette_score

import json 



if __name__ == "__main__":
    with open("dashboard/output_dashboard/sparse_block_model_300/epb.json", 'r') as fdata:
        data = json.load(fdata)

    G = nx.node_link_graph(data,link='edges')




    n = len(data['nodes'])


    max_depth = max(int(s.split("_")[1]) for s in data['nodes'][0].keys() if "cluster" in s)

    X = np.zeros((n,2),dtype=np.float32)
    labels = np.zeros((n,max_depth+1))

    

    for i,v in enumerate(data['nodes']):
        X[i,0] = float(v['X'])
        X[i,1] = float(v['Y'])
        for depth in range(max_depth+1):
            cid = v[f'cluster_{depth}']
            labels[i,depth] = n + cid if cid > 0 else i

    weights = dict()
    for u,v,data in G.edges(data=True):
        weights[(u,v)] = np.linalg.norm(X[u] - X[v])
    nx.set_edge_attributes(G,weights,'weight')
    apsp = nx.all_pairs_dijkstra_path_length(G)    

    mat = np.zeros((n,n))

    for u, dists in apsp:
        for v in dists:
            mat[u,v] = dists[v]
            mat[v,u] = dists[v]



    sil = list()
    rrange = list(range(max_depth-3,0,-1))
    for depth in rrange:
        sil.append(silhouette_score(mat,labels[:,depth]))
    
    import pylab as plt 
    plt.plot(rrange, sil)
    plt.show()

    for depth in range(max_depth+1):
        fig,ax = plt.subplots()

        lab = labels[:,depth].tolist()
        unique_objects = {item: id for id, item in enumerate(set(lab))}
        lab = [unique_objects[l] for l in lab]        

        ax.scatter(X[:,0], X[:,1], c=lab)
        fig.savefig(f"clustering/depth{depth}.png")
        plt.close(fig)

    np.savetxt("cluster.txt",labels,fmt="%d")