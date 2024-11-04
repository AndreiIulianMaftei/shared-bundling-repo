import argparse
import glob
import os.path
from bundle_pipeline import compute_bundling

metrics = ["distortion", "ink", "ambiguity"]
algorithms = ["epb", "sepb", "fd", "cubu", "wr", "straight"]

def process_single_bundling(file, algorithms):
    for algorithm in algorithms:
         compute_bundling(file, algorithm)      

    return

def main():
    '''
    Process a single dataset item by computing a specific metric. Perform this for all or a specific algorithm

    arguments:
        --file: path to the file. disregard file ending
        --algorithm: which algorithm do we want to evaluate. Default is all
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", help="path to the file. File extension irrelevant")
    parser.add_argument("--algorithm", help="Which algorithm should be evaluated. Default 'all'", default='all')

    args = parser.parse_args()

    if args.file and args.algorithm:
        file = args.file

        if not file.lower().endswith(".graphml"):
            print("Wrong file type. Graphml required")
            return
        if not os.path.isfile(file):
            print("File does not exist")
            return
        
        if args.algorithm in algorithms:
            algorithm = [args.algorithm]
        elif args.algorithm == "all":
            algorithm = algorithms
        else:
            print("Unknown algorithm")

        
        if algorithm:

            process_single_bundling(file, algorithm)


    else:
        print("Error: no file or metric given in arguments")


if __name__ == "__main__":
    main()