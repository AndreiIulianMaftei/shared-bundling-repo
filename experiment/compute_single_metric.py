import argparse
import glob
import os.path
from bundle_pipeline import read_bundling
from other_code.EPB.experiments import Experiment
from other_code.EPB.straight import StraightLine
import networkx as nx

metrics = ["distortion", "ink", "ambiguity"]
algorithms = ["epb", "sepb", "fd", "cubu", "wr", "straight"]

def process_single_metric(file, metric, algorithms):

    ## TODO this needs to be changed to not rely on epb. Maybe also have the original input in the output?
    G = nx.read_graphml(file + "/epb.graphml")
    straight = StraightLine(G)

    for algorithm in algorithms:
        bundling = read_bundling(file, algorithm)
        experiment = Experiment(bundling, straight)

        match metric:
            case "distortion":
                break
            case "ink":
                ## TODO should crash here as the ink requires a drawing.
                ink = experiment.calcInk()
                break
            case "ambiguity":
                break

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