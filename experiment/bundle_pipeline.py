import os
import networkx as nx
from other_code.EPB.epb import EPB_Biconn
from other_code.EPB.reader import Reader
from other_code.EPB.abstractBundling import GWIDTH, GraphLoader

output = 'output'

def compute_epb(file, out_path):
    G = Reader.readGraphML(f'{file}', G_width=GWIDTH, invertY=False, directed=False)
    bundling = EPB_Biconn(G)
    bundling.bundle()
    bundling.store(out_path)
    return

def compute_sepb(file, out_path):
    return

def compute_fd(file, out_path):
    
    return

def compute_cubu(file, out_path):
    return

def compute_wr(file, out_path):
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
            compute_sepb(file, out_path)
        case 'fd':
            compute_fd(file, out_path)
        case 'cubu':
            compute_cubu(file, out_path)
        case 'wr':
            compute_wr(file, out_path)


def read_epb(folder):

    path = folder + "/epb.graphml"
    
    bundling = GraphLoader(None)
    bundling.is_graphml = True
    bundling.filename = "epb"
    bundling.filepath = folder
    bundling.bundle()

    return bundling

def read_sepb(folder):
    return

def read_fd(folder):
    return

def read_cubu(folder):
    return

def read_wr(folder):
    return

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
    