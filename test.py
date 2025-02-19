import pylab as plt
from modules.EPB.reader import Reader
from modules.abstractBundling import GWIDTH, GraphLoader
import networkx as nx


G = Reader.readGraphML("inputs/g9.graphml", G_width=GWIDTH, invertY=False, directed=False)


with open("bundled_output.trl", 'r') as fin:
    for line in fin:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  
        parts[0]= parts[0].replace(":","")
        edge_id = int(parts[0])
        coords  = parts[1:]  
        edge_list = list(G.edges())

        spline_x = [float(x)/1000 for x in coords[0::2]]
        spline_y = [float(x)/1000 for x in coords[1::2]]

        
        if edge_id < len(G.edges()):
            u, v = edge_list[edge_id]
            G[u][v]["Spline_X"] = " ".join(str(x)for x in spline_x)
            G[u][v]["Spline_Y"] = " ".join(str(y) for y in spline_y)

nx.write_graphml(G, "bundle_cubu.graphml")

with open("bundle_cubu.trl", 'r') as fin:
    for line in fin:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  
        parts[0]= parts[0].replace(":","")
        edge_id = int(parts[0])
        coords  = parts[1:]  
        edge_list = list(G.edges())

        


        spline_x = [float(x)/1000 for x in coords[0::2]]
        spline_y = [float(x)/1000 for x in coords[1::2]]

        
        if edge_id < len(G.edges()):
            u, v = edge_list[edge_id]
            G[u][v]["Spline_X"] = " ".join(str(x)for x in spline_x)
            G[u][v]["Spline_Y"] = " ".join(str(y) for y in spline_y)

nx.write_graphml(G, "bundled_output.graphml")


