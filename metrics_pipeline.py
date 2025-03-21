import argparse
import csv
import glob
import os.path
from modules.abstractBundling import GWIDTH, GraphLoader, RealizedBundling
from modules.EPB.experiments import Experiment
from modules.EPB.straight import StraightLine
from modules.EPB.clustering import Clustering
from modules.EPB.plot import Plot
from modules.EPB.plotTest import PlotTest
import networkx as nx
import numpy as np
import json
from modules.metrics import Metrics
import pylab as plt
import csv


from modules.metrics import Metrics


def read_bundling(fname, invertX=False, invertY=False):
    graph_name = os.path.basename(fname).replace(".graphml", "")

    G = GraphLoader.readData_graphml(
        self=None, 
        path=os.path.dirname(fname),
        file=graph_name,
        invertX=invertX,
        invertY=invertY,
        reverse=None
    )
    G.graph['name'] = graph_name
    bundling = RealizedBundling(G,graph_name)

    return bundling

def write_json(Bundle:RealizedBundling, M:Metrics, path:str, algorithm:str):
    G = Bundle.G

    for u,v,data in G.edges(data=True):
        del data['Spline_X']
        del data['Spline_Y']
        del data['Xapprox']
        del data['Yapprox']

    if 'Name' in G.graph:
        del G.graph['Name']


    for metric in Metrics.getLocalMetrics():
        if metric not in M.metricvalues: continue

        counter = 0
        for u,v,data in G.edges(data=True):
            data[metric] = M.metricvalues[metric][counter]
            counter += 1

        G.graph[metric] = np.mean(M.metricvalues[metric])

    for metric in Metrics.getGlobalMetrics():
        if metric not in M.metricvalues: continue
        
        if metric == 'ambiguity':
            for i in range(0, 5):
                G.graph[f'ambiguity_{i+1}'] = M.metricvalues[metric][(i * 5)]
                G.graph[f'accuracy_{i+1}'] = M.metricvalues[metric][(i * 5 + 1)]
                G.graph[f'precision_{i+1}'] = M.metricvalues[metric][(i * 5 + 2)]
                G.graph[f'specificity_{i+1}'] = M.metricvalues[metric][(i * 5 + 3)]
                G.graph[f'FPR_{i+1}'] = M.metricvalues[metric][(i * 5 + 4)]
        else:
            G.graph[metric] = M.metricvalues[metric]
   
    data = nx.node_link_data(G, link="edges")

    if not os.path.isdir(f"{path}"): os.mkdir(f"{path}")
    with open(f'{path}/{algorithm}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
    G.graph['instance'] = path.split('/')[-1]
    
    file_exists = os.path.isfile('dashboard/output_dashboard/instances.csv')
    with open('dashboard/output_dashboard/instances.csv', 'a') as f:    
        writer = csv.DictWriter(f, delimiter=';', lineterminator='\n',fieldnames=G.graph)

        if not file_exists:
            writer.writeheader()
        
        writer.writerow(G.graph)
        
        

def process(input, filename, algorithm, output="dashboard/output_dashboard", metrics='all',verbose=False):
    if not os.path.isdir(output): os.mkdir(output)

    Bundle = read_bundling(f"{input}/{filename}/{algorithm}.graphml")

    M = Metrics(Bundle,verbose=verbose)
    
    if metrics == 'all': metrics_to_compute = Metrics.getImplementedMetrics()
    else: metrics_to_compute = metrics

    for metric in metrics_to_compute:
        if metric == "all_intersections" or metric == "self_intersections": continue
        # try:
        if verbose: print(f"calculating {metric} on {filename}/{algorithm}")
        mvalue = M.compute_metric(metric,return_mean=False)
        M.store_metric(metric, mvalue)
        # except:
        #     print("Problem with the metric")
        #     print(f"Failed on metric {metric} on graph {filename}/{algorithm}")

    write_json(Bundle, M, f'{output}/{filename}', algorithm)


def main():
    '''
    Process a single dataset item by computing a specific metric. Perform this for all or a specific algorithm

    arguments:
        --folder: path to input folder
        --metric: metric which we want to compute
        --algorithm: which algorithm do we want to evaluate. Default is all
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", default="outputs/",type=str, help="Path to input folder")
    parser.add_argument("--metric", type=str, default='all', help="which metric/s should be evaluated")
    parser.add_argument("--verbose", type=bool, default=False, help = "verbosity level")
    # parser.add_argument("--algorithm", help="Which algorithm should be evaluated. Default 'all'", default='all')
    # parser.add_argument("--draw", help="Should the bundling be drawn and saved as an image. {0,1}, Default '1'", default='1')


    args = parser.parse_args()

    inputfolder = args.folder
    if not os.path.isdir(inputfolder): 
        raise("Input folder not found")
    
    metrics = args.metric
    # metrics = ['inkratio', 'distortion', 'frechet', 'directionality', 'monotonicity', 'SL_angle']

    import matplotlib 
    matplotlib.use("TkAgg")
    
    import tqdm
    for gdata in tqdm.tqdm(os.listdir(inputfolder)):
        for algfile in os.listdir(f"{inputfolder}/{gdata}"):
            alg = algfile.replace(".graphml", "")
            print(f"Processing {gdata}/{alg}")
            process(inputfolder, gdata, alg, metrics=metrics, verbose=args.verbose)

if __name__ == "__main__":
    main()

    

    

