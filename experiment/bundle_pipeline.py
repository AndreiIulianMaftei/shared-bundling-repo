import os
import networkx as nx
from other_code.EPB.epb import EPB_Biconn
from other_code.EPB.reader import Reader
from other_code.EPB.abstractBundling import GWIDTH, GraphLoader
from tulip import tlp
from other_code.EPB.sepb import SpannerBundlingFG

output = 'output'

def compute_epb(file, out_path):
    G = Reader.readGraphML(f'{file}', G_width=GWIDTH, invertY=False, directed=False)
    bundling = EPB_Biconn(G)
    bundling.bundle()
    bundling.store(out_path)
    return

def compute_sepb(file, out_path):
    G = Reader.readGraphML(f'{file}', G_width=GWIDTH, invertY=False, directed=False)
    bundling = SpannerBundlingFG(G)
    bundling.bundle()
    bundling.store(out_path)
    return

def compute_fd(file, out_path):
    G = Reader.readGraphML(f'{file}', G_width=GWIDTH, invertY=False, directed=False)
    #tell python to run my app.js file here 
    #and my convert from xml to graphml file
    

    return

def compute_cubu(file, out_path):
    return

def compute_wr(file, out_path):
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

def compute_bundling(file, algorithm):

    name = file.split('/')[-1]
    name = name.replace('.graphml','')

    out_path = f'{output}/{name}/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    match algorithm:
        case 'epb':
            compute_epb(file, out_path + "epb.graphml")
        case 'sepb':
            compute_sepb(file, out_path + "sepb.graphml")
        case 'fd':
            compute_fd(file, out_path + "fd.graphml")
        case 'cubu':
            compute_cubu(file, out_path)
        case 'wr':
            compute_wr(file, out_path + "wr.graphml")


def read_epb(folder):
    bundling = GraphLoader(None)
    bundling.is_graphml = True
    bundling.filename = "epb"
    bundling.filepath = folder
    bundling.bundle()

    return bundling

def read_sepb(folder):
    bundling = GraphLoader(None)
    bundling.is_graphml = True
    bundling.filename = "sepb"
    bundling.filepath = folder
    bundling.bundle()

    return bundling

def read_fd(folder):
    path = folder + "/fd.graphml"
    bundling = GraphLoader(None)
    bundling.is_graphml = True
    bundling.filename = "fd"
    bundling.filepath = folder
    bundling.bundle()

    return bundling

def read_cubu(folder):
    return

def read_wr(folder):
    path = folder + "/wr.graphml"
    bundling = GraphLoader(None)
    bundling.is_graphml = True
    bundling.filename = "wr"
    bundling.filepath = folder
    bundling.bundle()
    
    return bundling

def read_bundling(folder, algorithm):
    match algorithm:
        case 'epb':
            G = read_epb(folder)
        case 'sepb':
            G = read_sepb(folder)
        case 'fd':
            G = read_fd(folder)
        case 'cubu':
            G = read_cubu(folder)
        case 'wr':
            G = read_wr(folder)

    return G
    