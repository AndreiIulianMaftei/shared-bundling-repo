import argparse
import glob
import os.path
from bundle_pipeline import read_bundling
from other_code.EPB.abstractBundling import GWIDTH
from other_code.EPB.experiments import Experiment
from other_code.EPB.straight import StraightLine
from other_code.EPB.plot import Plot
from other_code.EPB.plotTest import PlotTest
import networkx as nx
import numpy as np

metrics = ["angle", "drawing", "distortion", "ink", "frechet", "monotonicity", "monotonicity_projection", "all", "intersect_all", "self_intersect"]
#algorithms = ["epb", "sepb", "fd", "cubu", "wr", "straight"]
algorithms = ["fd", "epb", "wr"]



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
    ink_ratios = []

    if draw:
        if not os.path.exists(f"{file}/images"):
            os.makedirs(f"{file}/images")

        straight.draw(f"{file}/images/")

    all_edges = []
    for algorithm in algorithms:

        bundling = read_bundling(file, algorithm)

        if draw:
           bundling.draw(f"{file}/images/")

        G = nx.Graph()
        bundling.draw(f"{file}/images/")
        experiment = Experiment(bundling, straight)
        all_edges= experiment.all_edges()
        if(metric == "intersect_all"):
            print(experiment.all_intersection(all_edges))

        int_aux = experiment.calcInkRatio('/home/andrei/c++/shared-bundling-repo/output/airlines/images/')
        ink_ratios.append((int_aux, algorithm))
        
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
                ink = experiment.calcInk(f'/home/andrei/c++/shared-bundling-repo/output/airlines/images/')
                print(experiment.calcInkRatio('/home/andrei/c++/shared-bundling-repo/output/airlines/images/'))
            case "frechet":
                frok = 1
                
                rez = experiment.fastFrechet(algorithm)
                
                print(rez.__len__())
                rez_all.append((rez, algorithm))
            case "monotonicity_projection":
                mono = experiment.projection_Monotonicty(algorithm)
                print(mono)
                rez_all.append((mono, algorithm))
            case "angle":
                angles = experiment.calcAngle(algorithm)
                print(angles)
                rez_all.append((angles, algorithm))
            case "self_intersect":
                all_intersections, intersect = experiment.count_self_intersections(algorithm)
                
                print(all_intersections)
                
                rez_all.append((all_intersections, algorithm))

    #print(rez_all)
    #print(rez__all_aux)
    #return
    plotter = PlotTest()
    if(metric == "monotonicity_projection"):
        print(rez_all.__len__())
        plotter.plotProjectedMonotonicity(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["monotonicity_projected_fd.png", "monotonicity_projected_epb.png", "monotonicity_projected_wr.png"], ink_ratios)
    if(metric == "monotonicity"):
        print(rez_all.__len__())
        plotter.plotMonotonicity(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["monotonicity_fd.png", "monotonicity_epb.png", "monotonicity_wr.png"], ink_ratios)

    if(metric == "distortion"):
        print(rez_all.__len__())
        plotter.plotDistortionHistogram(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["distortion_fd.png", "distortion_epb.png", "distortion_wr.png"], ink_ratios)

    if(metric == "frechet"):
        print(rez_all.__len__())
        plotter.plotFrechet(rez_all)   
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["frechet_fd.png", "frechet_epb.png", "frechet_wr.png"], ink_ratios)
 
    if(metric == "all"):
        print(rez_all.__len__())
        experiment.plotAll(["fd", "epb", "wr"], ["distortion", "monotonicity", "frechet"])  

    if(metric == "angle"):
        print(rez_all.__len__())
        plotter.plotAngles(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["angle_fd.png", "angle_epb.png", "angle_wr.png"], ink_ratios)

    if(metric == "self_intersect"):
        print(rez_all.__len__())
        plotter.plotSelfIntersect(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["self_intersect_fd.png", "self_intersect_epb.png", "self_intersect_wr.png"], ink_ratios)
        
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