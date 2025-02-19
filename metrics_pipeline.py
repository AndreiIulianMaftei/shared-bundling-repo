import argparse
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

metrics = ["angle", "drawing", "distortion", "ink", "frechet", "monotonicity", "monotonicity_projection", "all", "intersect_all", "self_intersect"]
#algorithms = ["epb", "sepb", "fd", "cubu", "wr", "straight"]
algorithms = ["fd", "epb", "fd"]

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


def process_single_metric(file, metric, algorithms, draw):

    ## TODO this needs to be changed to not rely on epb. Maybe also have the original input in the output?
    #test if the file is accesible

    gname = file.split("/")[-1].replace(".graphml", '')

    G = nx.read_graphml(file)
    G = nx.Graph(G)
    #print(G)
 
    straight = StraightLine(G)
    straight.scaleToWidth(GWIDTH)

    #formats the straightline drawing
    straight.bundle()
 
    
    rez_all = []
    ink_ratios = []

    if draw:
        if not os.path.exists(f"drawings/"):
            os.makedirs(f"drawings/")

        straight.draw(f"drawings/straight")

    all_edges = []
    for algorithm in algorithms:

        bundling = read_bundling(file)

        if draw:
           bundling.draw(f"drawings/{algorithm}_{gname}")

        experiment = Experiment(bundling, straight)

        if(metric == "intersect_all"):
            print(experiment.all_intersection(all_edges))

        #Gets path to straight line and bundled drawing
        int_aux = experiment.calcInkRatio('/home/andrei/c++/shared-bundling-repo/output/airlines/images/')
        # ink_ratios.append((int_aux, algorithm))
        # print(experiment.calcNumberOfSegments(algorithm))
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
                print(angles.__len__())
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
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["monotonicity_projection_fd.png", "monotonicity_projection_epb.png", "monotonicity_projection_wr.png"], ink_ratios, )
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


def all_metrics(dir):
    for graph in os.listdir(dir):
        # metrics = ["angle", "drawing", "distortion", "ink", "frechet", "monotonicity", "monotonicity_projection", "all", "intersect_all", "self_intersect"]
        metrics = ['all']
        algs = ['epb', 'sepb', 'fd']
        for metric in metrics:
            print(metric)
            try: 
                process_single_metric(f"{dir}{graph}", metric, algs, 1)
            except: print(f"Couldnt do {metric} metric")

def write_json(G, M, path, algorithm):
    G = G.G

    for u,v,data in G.edges(data=True):
        del data['Spline_X']
        del data['Spline_Y']
        del data['Xapprox']
        del data['Yapprox']

    for metric in ['distortion', 'SL_angle']:
        counter = 0
        for u,v,data in G.edges(data=True):
            data[metric] = M.metricvalues[metric][counter]
            counter += 1

        G.graph[metric] = np.mean(M.metricvalues[metric])

    G.graph['inkratio'] = M.metricvalues['inkratio']
    data = nx.node_link_data(G, link="edges")

    if not os.path.isdir(f"{path}"): os.mkdir(f"{path}")
    with open(f'{path}/{algorithm}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def process(input, output, filename, algorithm):

    G = read_bundling("inputs/bundle_cubu.graphml")
    G.draw("draw")

    G = read_bundling("inputs/bundled_output.graphml")
    G.draw("draw2")
    

    # import pylab as plt

    # G = read_bundling("outputs/epb_airlines.graphml")
    # print(type(G))

    # M = Metrics(G)
    
    for metric in ['distortion', 'inkratio', 'SL_angle']:#M.implemented_metrics:
        mvalue = M.compute_metric(metric,return_mean=False)
        M.store_metric(metric, mvalue)

    write_json(G, M, f'{output}/{filename}', algorithm)

if __name__ == "__main__":
    from modules.metrics import Metrics
    import pylab as plt

    for dataset in ['airlines', 'migration']:
        for algo in ['epb', 'fd', 'sepb', 'wr']:
            process("outputs", 'dashboard/output_dashboard', dataset, algo)

    

    

