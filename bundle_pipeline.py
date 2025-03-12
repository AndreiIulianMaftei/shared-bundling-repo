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

    for v in G.nodes():
        G.nodes[v]['x'] = G.nodes[v]['X']
        G.nodes[v]['y'] = G.nodes[v]['Y']

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


import os
import networkx as nx
from collections import defaultdict

def compute_cubu(file, out_path, G_width=None):
    """
    1) Read the input .graphml (or whatever) into a NetworkX graph.
    2) Compute bounding box and normalize node coordinates to [0,1600]x[0,1600].
    3) Write a .trl file (edge list) for CuBu using the normalized coords.
    4) Run CuBu.
    5) Parse the 'bundled_output.trl' that CuBu generates (which is in normalized coords).
    6) De-normalize the polylines to the original coordinate space.
    7) Update node positions by averaging the first/last points from the polylines.
    8) Store the polylines in the NetworkX graph's edge attributes.
    9) Write the final .graphml to `out_path`.
    """


    G = nx.read_graphml(file)

    for v in G.nodes():
        G.nodes[v]["X"] = float(G.nodes[v].get("x", 0.0))
        G.nodes[v]["Y"] = float(G.nodes[v].get("y", 0.0))

    #Compute bounding box
    xs = [G.nodes[v]["X"] for v in G.nodes()]
    ys = [G.nodes[v]["Y"] for v in G.nodes()]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    range_x = max_x - min_x
    range_y = max_y - min_y

    #[0,1600]
    TARGET_SIZE = 1600.0

    
    #Normalize coords, store them temporarily in G
    for v in G.nodes():
        orig_x = G.nodes[v]["X"]
        orig_y = G.nodes[v]["Y"]

        if range_x != 0:
            norm_x = (orig_x - min_x) / range_x
        else:
            norm_x = 0.0

        if range_y != 0:
            norm_y = (orig_y - min_y) / range_y
        else:
            norm_y = 0.0

        G.nodes[v]["NormX"] = norm_x * TARGET_SIZE
        G.nodes[v]["NormY"] = norm_y * TARGET_SIZE

    #input.trl
    temp_input = "input.trl"
    edge_list = list(G.edges())

    with open(temp_input, 'w') as fout:
        for idx, (u, v) in enumerate(edge_list):
            x1, y1 = G.nodes[u]["NormX"], G.nodes[u]["NormY"]
            x2, y2 = G.nodes[v]["NormX"], G.nodes[v]["NormY"]
            fout.write(f"{idx}: {x1} {y1} {x2} {y2}\n")

    # CuBu
    cubu_command = f"./cubu/CUBu/cubu -i 1000 -f {temp_input}"
    print(f"Running CuBu command: {cubu_command}")
    os.system(cubu_command)

    #De-normalize
    bundled_output = "bundled_output.trl"
    if not os.path.exists(bundled_output):
        print("Error: 'bundled_output.trl' not found. CuBu may have failed.")
        nx.write_graphml(G, out_path)
        return

    # We'll store the final polylines for each edge
    # and also track the first/last coordinates for each edge
    # so we can update node positions afterward.
    edge_first_last = dict()  # edge_id -> ( (firstX, firstY), (lastX, lastY) )

    with open(bundled_output, 'r') as fin:
        for line in fin:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            edge_id_str = parts[0].replace(":", "")
            edge_id = int(edge_id_str)

            coords = parts[1:]  
            raw_x = [float(x) for x in coords[0::2]]
            raw_y = [float(y) for y in coords[1::2]]

            final_x = [min_x + (val / TARGET_SIZE) * range_x for val in raw_x]
            final_y = [min_y + (val / TARGET_SIZE) * range_y for val in raw_y]

        
            if edge_id < len(edge_list):
                u, v = edge_list[edge_id]

                G[u][v]["Spline_X"] = " ".join(str(x) for x in final_x)
                G[u][v]["Spline_Y"] = " ".join(str(y) for y in final_y)

                if final_x and final_y:
                    first_point = (final_x[0], final_y[0])
                    last_point = (final_x[-1], final_y[-1])
                    edge_first_last[edge_id] = (first_point, last_point)

    # ------------------------------------------------------------------
    # 7) Update node positions from the polylines' first/last points
    # ------------------------------------------------------------------
    # We'll collect all relevant first/last points for each node, then
    # average them to get the final node coordinate.

    # For convenience, create a dictionary: node -> list of (x,y)
    node_positions = defaultdict(list)

    for edge_id, (u, v) in enumerate(edge_list):
        if edge_id in edge_first_last:
            (fx, fy), (lx, ly) = edge_first_last[edge_id]
            # The first coordinate belongs to node u
            node_positions[u].append((fx, fy))
            # The last coordinate belongs to node v
            node_positions[v].append((lx, ly))

    # Now average them for each node
    for node, coords in node_positions.items():
        if len(coords) == 1:
            # If there's only one edge for that node
            avg_x, avg_y = coords[0]
        else:
            # Otherwise, average across all edges
            sum_x = sum(pt[0] for pt in coords)
            sum_y = sum(pt[1] for pt in coords)
            n = len(coords)
            avg_x = sum_x / n
            avg_y = sum_y / n

        # Update the node's X/Y
        G.nodes[node]["X"] = avg_x
        G.nodes[node]["Y"] = avg_y

    isolated_nodes = [n for n in G.nodes() if G.degree(n) == 0]
    G.remove_nodes_from(isolated_nodes)
    # ------------------------------------------------------------------
    # 9) Write final .graphml
    # ------------------------------------------------------------------
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
    import tqdm
    if not os.path.isdir("outputs"): os.mkdir("outputs")

    REMOVE_ISOLATES   = True
    REMOVE_COMPONENTS = True

    for file in tqdm.tqdm(os.listdir(dir)):

        name = file.split('/')[-1]
        name = name.replace('.graphml','')

        G = nx.read_graphml(f'{dir}/{file}')
        G = nx.Graph(G)

        if REMOVE_ISOLATES: G.remove_nodes_from(list(nx.isolates(G)))
        if REMOVE_COMPONENTS:
            largest_component = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_component)

        G = nx.convert_node_labels_to_integers(G)

        nx.write_graphml(G,f'{dir}/{file}')

        if not os.path.isdir(f"outputs/{name}"): os.mkdir(f"outputs/{name}")

        algs = ['wr','fd', 'epb', 'sepb']
        if os.path.exists('cubu'): algs += ['cubu']

        for alg in algs:
            print(f"Computing {alg} for {file}")
            compute_bundling(f"{dir}/{file}", alg, f"outputs/{name}/{alg}.graphml")    

if __name__ == "__main__":
    bundle_all("inputs")
