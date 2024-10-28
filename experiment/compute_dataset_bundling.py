import argparse
import glob
import os.path

metrics = ["distortion", "ink", "ambiguity"]
algorithms = ["epb", "sepb", "fd", "cubu", "wr", "straight"]

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--file", help="path to the file. File extension must be .graphml")
    parser.add_argument("--algorithm", help="Which algorithm should be evaluated. Default 'all'", default='all')
    ### TODO add option to parameterize certain algorithms.

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
            
            process()


    else:
        print("Error: no file or metric given in arguments")


if __name__ == "__main__":
    main()