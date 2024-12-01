import argparse
import glob
import os.path
from bundle_pipeline import read_bundling
from other_code.EPB.abstractBundling import GWIDTH
from other_code.EPB.experiments import Experiment
from other_code.EPB.straight import StraightLine
import networkx as nx
import numpy as np

metrics = ["drawing", "distortion", "ink", "frechet", "monotonicity", "all"]
#algorithms = ["epb", "sepb", "fd", "cubu", "wr", "straight"]
algorithms = ["epb", "fd", "wr"]

def process_single_metric(file, metric, algorithms, draw):

    ## TODO this needs to be changed to not rely on epb. Maybe also have the original input in the output?
    #test if the file is accesible

    G = nx.read_graphml(file + "/epb.graphml")
    G = nx.Graph(G)
    #print(G)
 
    straight = StraightLine(G)
    straight.scaleToWidth(GWIDTH)
    straight.bundle()
 
    
    rez_all = []

    if draw:
        if not os.path.exists(f"{file}/images"):
            os.makedirs(f"{file}/images")

        straight.draw(f"{file}/images/")

    for algorithm in algorithms:
        bundling = read_bundling(file, algorithm)

        #if draw:
         #   bundling.draw(f"{file}/images/")
        bundling.draw(f"{file}/images/")
        experiment = Experiment(bundling, straight)
        match metric:
            case "distortion":
                
                _,_,_,_,_,distortions = experiment.calcDistortion()
                rez_all.append((distortions, algorithm))
                #print(algorithm)
                #print(distortions)
                #experiment.plotHistogram(distortions)
            case "monotonicity":
                mono = experiment.calcMonotonicity(algorithm)
                print(algorithm)
                print(mono)
                rez_all.append((mono, algorithm))
            case "ink":
                ## TODO should crash here as the ink requires a drawing.
                ink = experiment.calcInk()
            case "frechet":
                frok = 1
                
                rez = experiment.calcFrechet(algorithm)
                print(rez.__len__())
                rez_all.append((rez, algorithm))


    if(metric == "monotonicity"):
        print(rez_all.__len__())
        experiment.plotMonotonicity(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["monotonicity_normalisation_fd.png", "monotonicity_normalisation_epb.png", "monotonicity_normalisation_wr.png"])

    if(metric == "distortion"):
        print(rez_all.__len__())
        experiment.plotDistortionHistogram(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["distortion_normalisation_fd.png", "distortion_normalisation_epb.png", "distortion_normalisation_wr.png"])

    if(metric == "frechet"):
        print(rez_all.__len__())
        experiment.plotFrechet(rez_all)   
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["frechet_normalisation_fd.png", "frechet_normalisation_epb.png", "frechet_normalisation_wr.png"])
 
    if(metric == "all"):
        print(rez_all.__len__())
        experiment.plotAll(["fd", "epb", "wr"], ["distortion", "monotonicity", "frechet"])  
         
    return

def main():
    '''
    Process a single dataset item by computing a specific metric. Perform this for all or a specific algorithm

    arguments:
        --folder: path to the file. disregard file ending
        --metric: metric which we want to compute
        --algorithm: which algorithm do we want to evaluate. Default is all
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", help="path to the file. File extension irrelevant")
    parser.add_argument("--metric", help="which metric should be evaluated")
    parser.add_argument("--algorithm", help="Which algorithm should be evaluated. Default 'all'", default='all')
    parser.add_argument("--draw", help="Should the bundling be drawn and saved as an image. {0,1}, Default '1'", default='1')

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
            return

        draw = False
        if args.draw == '1':
            draw = True
        
        if algorithm:

            process_single_metric(folder, metric, algorithm, draw)


    else:
        print("Error: no file or metric given in arguments")


if __name__ == "__main__":
    main()