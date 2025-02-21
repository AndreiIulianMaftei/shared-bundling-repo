import json
import os
import networkx as nx
from modules.EPB.epb import EPB_Biconn
from modules.EPB.reader import Reader
from modules.abstractBundling import GWIDTH, GraphLoader
# from tulip import tlp
from modules.EPB.sepb import SpannerBundlingFG

def postProcess(G):
    for v in G.nodes():
        G.nodes[v]["X"] = G.nodes[v]['x']
        G.nodes[v]["Y"] = G.nodes[v]['y']


def compute_epb(file, out_path):
    G = Reader.readGraphML(f'{file}', G_width=GWIDTH, invertY=False, directed=False)
    postProcess(G)
    bundling = EPB_Biconn(G)
    bundling.bundle()
    bundling.store(out_path)
    return

def compute_sepb(file, out_path):
    G = Reader.readGraphML(f'{file}', G_width=GWIDTH, invertY=False, directed=False)
    postProcess(G)
    bundling = SpannerBundlingFG(G)
    bundling.bundle()
    bundling.store(out_path)
    return

def compute_fd(file, out_path):
    G = Reader.readGraphML(f'{file}', G_width=GWIDTH, invertY=False, directed=False)
    import os
    
    json_data = nx.node_link_data(G, link='edges')
    with open('tmp.json', 'w') as f:
        json.dump(json_data, f)

    os.system(f"node modules/FD/bundleFD.js tmp.json")
    
    with open('outputs/edges.edge', 'r') as fdata:
        edgedata = fdata.read()
    os.remove("outputs/edges.edge")
    os.remove("tmp.json")

    polylines = [edge.split(" ") for edge in edgedata.split("\n")]

    for line in polylines: 
        u,v = (line[0], line[1])
        G[u][v]["Spline_X"] = " ".join([str(x) for x in line[2::2]])
        G[u][v]["Spline_Y"] = " ".join([str(y) for y in line[3::2]])

    postProcess(G)
    
    nx.write_graphml(G,out_path)
    
    return



def compute_cubu(file, out_path):
    """
    1) Read the input .graphml (or whatever) into a NetworkX graph via your Reader.
    2) Write a .trl file (edge list) for CuBu.
    3) Run CuBu with the system call "./cubu -i 1000 -f input.trl"
    4) Parse the 'bundled_output.trl' that CuBu generates.
    5) Store the polylines in the NetworkX graph's edge attributes.
    6) Write the final .graphml to `out_path`.
    """

    G = Reader.readGraphML(file, G_width=GWIDTH, invertY=False, directed=False)
    maxX = -999999
    minX = 999999
    for v in G.nodes():
        G.nodes[v]["X"] = G.nodes[v].get("x", 0.0)
        G.nodes[v]["Y"] = G.nodes[v].get("y", 0.0)


    temp_input = "input.trl"
    edge_list = list(G.edges())
    
    with open(temp_input, 'w') as fout:
        for idx, (u, v) in enumerate(edge_list):
            x1, y1 = G.nodes[u]["X"], G.nodes[u]["Y"]
            x2, y2 = G.nodes[v]["X"], G.nodes[v]["Y"]
            if(idx <= len(edge_list)):
                fout.write(f"{idx}: {x1} {y1} {x2} {y2} \n")
           

    cubu_command = f"./cubu -i 1000 -f {temp_input}"
    print(f"Running CuBu command: {cubu_command}")
    print(os.getcwd())
    os.system(cubu_command)

    bundled_output = "bundled_output.trl"

    if not os.path.exists(bundled_output):
        print("Error: 'bundled_output.trl' not found. CuBu may have failed.")
        nx.write_graphml(G, out_path)
        return

    with open(bundled_output, 'r') as fin:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  
            parts[0]= parts[0].replace(":","")
            edge_id = int(parts[0])
            coords  = parts[1:]  

    
            spline_x = [float(x)/1000 for x in coords[0::2]]
            spline_y = [float(x)/1000 for x in coords[1::2]]

         
            if edge_id < len(edge_list):
                u, v = edge_list[edge_id]
                G[u][v]["Spline_X"] = " ".join(str(x)for x in spline_x)
                G[u][v]["Spline_Y"] = " ".join(str(y) for y in spline_y)

    nx.write_graphml(G, out_path)
    print(f"Saved bundled graph to {out_path}")

  

def compute_wr(file, out_path):
    try: 
        from tulip import tlp
    except: 
        print("python-tulip is needed")
        return 
    
    G = tlp.loadGraph(f'{file}')
    G_out = nx.Graph()

    x = G.getDoubleProperty("x")
    y = G.getDoubleProperty("y")

    params1 = tlp.getDefaultPluginParameters("Edge bundling", G)
    params2 = tlp.getDefaultPluginParameters("Curve edges", G)
    viewLayout = G.getLayoutProperty("viewLayout")

    G.applyAlgorithm("Edge bundling", params1)
    G.applyAlgorithm("Curve edges", params2)
    G.applyAlgorithm("Edge bundling", params1)

    for n in G.getNodes():
        G_out.add_node(n.id, X=x.getNodeValue(n), Y=y.getNodeValue(n))

    for e in G.getEdges():
        controlPoints = viewLayout.getEdgeValue(e)

        controlPoints.insert(0, (x.getNodeValue(G.source(e)), y.getNodeValue(G.source(e))))
        controlPoints.append((x.getNodeValue(G.target(e)), y.getNodeValue(G.target(e))))

        src = G.source(e).id
        tgt = G.target(e).id

        pointList = tlp.computeBezierPoints(controlPoints, nbCurvePoints=50) 

        X = []
        Y = []
        for p in pointList:
            X.append(p[0])
            Y.append(p[1])

        X = ' '.join(map(str, X))
        Y = ' '.join(map(str, Y))

        G_out.add_edge(src, tgt, Spline_X=X, Spline_Y=Y)

    nx.write_graphml(G_out, out_path)


    return

def compute_bundling(file, algorithm,outfile):
    match algorithm:
        case 'epb':
            compute_epb(file, outfile)
        case 'sepb':
            compute_sepb(file, outfile)
        case 'fd':
            compute_fd(file, outfile)
        case 'cubu':
            compute_cubu(file, outfile)
        case 'wr':
            compute_wr(file, outfile)
    
def bundle_all(dir):
    import os 

    for file in os.listdir(dir):

        name = file.split('/')[-1]
        name = name.replace('.graphml','')

        if not os.path.isdir(f"outputs/{name}"): os.mkdir(f"outputs/{name}")

        for alg in ['wr','fd', 'epb', 'sepb', 'cubu']:
            compute_bundling(f"{dir}/{file}", alg, f"outputs/{name}/{alg}.graphml")    

if __name__ == "__main__":
    bundle_all("inputs")
