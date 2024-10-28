import argparse
import glob
import os.path
from bundle_pipeline import read_bundling

metrics = ["distortion", "ink", "ambiguity"]
algorithms = ["epb", "sepb", "fd", "cubu", "wr", "straight"]

def process_single_metric(file, metric, algorithms):

    for algorithm in algorithms:
        G = read_bundling(file, algorithm)

        match metric:
            case "distortion":
                break
            case "ink":
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

    parser.add_argument("--file", help="path to the file. File extension irrelevant")
    parser.add_argument("--metric", help="which metric should be evaluated")
    parser.add_argument("--algorithm", help="Which algorithm should be evaluated. Default 'all'", default='all')

    args = parser.parse_args()

    if args.file and args.metric:
        file = args.file
        metric = args.metric

        if not metric in metrics:
            print("Error: metric does not exist")
        if file.lower().endswith(".graphml"):
            print("Wrong file type. Graphml required")
        if not os.path.isfile(file):
            print("File does not exist")
        
        if args.algorithm in algorithms:
            algorithm = [args.algorithm]
        elif args.algorithm == "all":
            algorithm = algorithms
        else:
            print("Unknown algorithm")

        
        if algorithm:

            process_single_metric(file, metric, algorithm)


    else:
        print("Error: no file or metric given in arguments")


if __name__ == "__main__":
    main()