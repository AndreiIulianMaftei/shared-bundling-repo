import os 
import networkx as nx
import shutil
import json 

import gdMetriX as gd
import tqdm as tqdm


alldata = os.listdir("all_outputs")


crossings = dict()
for gname in tqdm.tqdm(alldata):
    G = nx.read_graphml(f"all_outputs/{gname}/epb.graphml")
    
    pos = {v: (float(G.nodes[v]['X']), float(G.nodes[v]['Y'])) for v in G.nodes()}

    ncrossings = gd.number_of_crossings(G,pos)

    crossings[gname] = ncrossings 

    with open("crossingsjson.json", 'w') as fdata:
        json.dump(crossings,fdata,indent=4)
