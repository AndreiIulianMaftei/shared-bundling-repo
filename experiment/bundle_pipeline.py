import os
import networkx as nx
from other_code.EPB.epb import EPB_Biconn
from other_code.EPB.reader import Reader
from other_code.EPB.abstractBundling import GWIDTH

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
    name = name.trim()

    out_path = f'{output}/{name}/'

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    match algorithm:
        case 'epb':
            compute_epb(file, out_path)
        case 'sepb':
            compute_sepb(file, out_path)
        case 'fd':
            compute_fd(file, out_path)
        case 'cubu':
            compute_cubu(file, out_path)
        case 'wr':
            compute_wr(file, out_path)


def read_epb(file):
    return

def read_sepb(file):
    return

def read_fd(file):
    return

def read_cubu(file):
    return

def read_wr(file):
    return

def read_bundling(file, algorithm):
    match algorithm:
        case 'epb':
            G = read_epb(file)
        case 'sepb':
            G = read_sepb(file)
        case 'fd':
            G = read_fd(file)
        case 'cubu':
            G = read_cubu(file)
        case 'wr':
            G = read_wr(file)

    return G
    