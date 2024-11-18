import argparse
import glob
import os.path
from bundle_pipeline import read_bundling
from other_code.EPB.experiments import Experiment
from other_code.EPB.straight import StraightLine
import networkx as nx
import numpy as np

metrics = ["drawing", "distortion", "ink", "frechet"]
#algorithms = ["epb", "sepb", "fd", "cubu", "wr", "straight"]
algorithms = ["epb"]

def process_single_metric(file, metric, algorithms):

    ## TODO this needs to be changed to not rely on epb. Maybe also have the original input in the output?
    #test if the file is accesible

    G = nx.read_graphml(file + "/fd.graphml")
    G = nx.Graph(G)
    print(G)
 
    straight = StraightLine(G)

    for algorithm in algorithms:
        bundling = read_bundling(file, algorithm)
        experiment = Experiment(bundling, straight)

        match metric:
            case "distortion":
                _,_,_,_,_,distortions = experiment.calcDistortion()
                experiment.plotHistogram(distortions)
                
                break
            case "ink":
                ## TODO should crash here as the ink requires a drawing.
                ink = experiment.calcInk()
                break
            case "frechet":
                

                rez = experiment.calcFrechet()
                MAX = max(rez)
                MIN = min(rez)

                #MIN = min(min(rez_EPB), min(rez_FD))
                #print(np.mean(rez_EPB))
                experiment.plotFrechet(rez, MIN, MAX)

                

    return

def main():
    '''
    Process a single dataset item by computing a specific metric. Perform this for all or a specific algorithm

    arguments:
        --file: path to the file. disregard file ending
        --metric: metric which we want to compute
        --algorithm: which algorithm do we want to evaluate. Default is all
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", help="path to the file. File extension irrelevant")
    parser.add_argument("--metric", help="which metric should be evaluated")
    parser.add_argument("--algorithm", help="Which algorithm should be evaluated. Default 'all'", default='all')

    args = parser.parse_args()

    if args.folder and args.metric:
        folder = args.folder
        metric = args.metric

        if not metric in metrics:
            print("Error: metric does not exist")
            return
        if not os.path.isdir(folder):
            print("Folder does not exist")
            return

        if args.algorithm in algorithms:
            algorithm = [args.algorithm]
        elif args.algorithm == "all":
            algorithm = algorithms
        else:
            print("Unknown algorithm")

        
        if algorithm:

            process_single_metric(folder, metric, algorithm)


    else:
        print("Error: no file or metric given in arguments")


if __name__ == "__main__":
    main()