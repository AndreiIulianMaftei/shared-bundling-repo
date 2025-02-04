import os
import networkx as nx
from bundle_pipeline import read_bundling
from other_code.EPB.abstractBundling import GWIDTH
from other_code.EPB.experiments import Experiment
from other_code.EPB.straight import StraightLine
from other_code.EPB.clustering import Clustering
from other_code.EPB.plot import Plot
from other_code.EPB.plotTest import PlotTest

"""

"""
metrics = [
    "angle",
    "drawing",
    "distortion",
    "frechet",
    "monotonicity",
    "monotonicity_projection",
    "all",
    "intersect_all",
    "self_intersect",
    "ink"
]

algorithms = ["fd", "epb", "wr"]

def process_single_metric(folder: str, metric: str, algorithms: list, draw: bool, input_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    epb_graph_path = os.path.join(folder, "epb.graphml")
    if not os.path.isfile(epb_graph_path):
        print(f"[Warning] No 'epb.graphml' found in: {folder}")
        return
    G = nx.read_graphml(epb_graph_path)
    G = nx.Graph(G)
    straight = StraightLine(G)
    straight.scaleToWidth(GWIDTH)
    straight.bundle()
    images_dir = os.path.join("output", "airlines", "images")
    if draw and not os.path.exists(images_dir):
        os.makedirs(images_dir)
    if draw:
        straight.draw(images_dir)
    rez_all = []
    experiment_for_all_edges = Experiment(straight, straight)
    all_edges_ref = experiment_for_all_edges.all_edges()
    for alg in algorithms:
        bundling = read_bundling(folder, alg)
        if bundling is None:
            print(f"[Warning] Could not read bundling '{alg}' in: {folder}")
            continue
        if draw:
            bundling.draw(images_dir)
        experiment = Experiment(bundling, straight)

        if metric == "intersect_all":
            intersections = experiment.all_intersection(all_edges_ref)
            print(f"[{alg}] total intersections: {intersections}")
            rez_all.append((intersections, alg))
        elif metric == "ink":
            ink_ratio = experiment.calcInkRatio(images_dir)
            rez_all.append((ink_ratio, alg))
        elif metric == "distortion":
            dist = experiment.calcDistortion()[5]
            rez_all.append((dist, alg))
        elif metric == "monotonicity":
            mono_vals = experiment.calcMonotonicity(alg)
            rez_all.append((mono_vals, alg))
        elif metric == "frechet":
            frechet_vals = experiment.fastFrechet(alg)
            rez_all.append((frechet_vals, alg))
        elif metric == "monotonicity_projection":
            mono_proj = experiment.projection_Monotonicty(alg)
            rez_all.append((mono_proj, alg))
        elif metric == "angle":
            angles = experiment.calcAngle(alg)
            rez_all.append((angles, alg))
        elif metric == "self_intersect":
            all_int, total = experiment.count_self_intersections(alg)
            print(f"[{alg}] self-intersections = {total}")
            rez_all.append((all_int, alg))
        elif metric == "all":
            distortions = experiment.calcDistortion()[5]
            frechet_vals = experiment.fastFrechet(alg)
            monotonicities = experiment.calcMonotonicity(alg)
            rez_all.append((distortions, f"distortion-{alg}"))
            rez_all.append((frechet_vals, f"frechet-{alg}"))
            rez_all.append((monotonicities, f"monotonicity-{alg}"))
    if len(rez_all) == 0:
        return
    ink_ratios = []
    plotter = PlotTest(output_folder=output_folder)
    if(metric == "monotonicity_projection"):
        print(rez_all.__len__())
        plotter.plotProjectedMonotonicity(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["monotonicity_projection_fd.png", "monotonicity_projection_epb.png", "monotonicity_projection_wr.png"], ink_ratios, input_folder, output_folder )
    if(metric == "monotonicity"):
        print(rez_all.__len__())
        plotter.plotMonotonicity(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["monotonicity_fd.png", "monotonicity_epb.png", "monotonicity_wr.png"], ink_ratios,  input_folder, output_folder)

    if(metric == "distortion"):
        print(rez_all.__len__())
        plotter.plotDistortionHistogram(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["distortion_fd.png", "distortion_epb.png", "distortion_wr.png"], ink_ratios, input_folder,  output_folder)

    if(metric == "frechet"):
        print(rez_all.__len__())
        plotter.plotFrechet(rez_all)   
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["frechet_fd.png", "frechet_epb.png", "frechet_wr.png"], ink_ratios,  input_folder, output_folder)
 
    if(metric == "all"):
        print(rez_all.__len__())
        experiment.plotAll(["fd", "epb", "wr"], ["distortion", "monotonicity", "frechet"])  

    if(metric == "angle"):
        print(rez_all.__len__())
        plotter.plotAngles(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["angle_fd.png", "angle_epb.png", "angle_wr.png"], ink_ratios, input_folder,  output_folder)

    if(metric == "self_intersect"):
        print(rez_all.__len__())
        plotter.plotSelfIntersect(rez_all)
        experiment.plotMegaGraph(["fd", "epb", "wr"], metric, ["self_intersect_fd.png", "self_intersect_epb.png", "self_intersect_wr.png"], ink_ratios, input_folder,  output_folder)
        

def run_all_experiments(input_folder="testingDatabases", output_folder="results"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for dataset_name in os.listdir(input_folder):
        dataset_path = os.path.join(input_folder, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
        epb_path = os.path.join(dataset_path, "epb.graphml")
        if not os.path.isfile(epb_path):
            print(f"[Info] Skipping {dataset_name}, no epb.graphml found.")
            continue
        print(f"=== Found dataset: {dataset_name} ===")
        dataset_output_folder = os.path.join(output_folder, dataset_name)
        dataset_input_folder = os.path.join(input_folder, dataset_name)
        if not os.path.exists(dataset_output_folder):
            os.makedirs(dataset_output_folder)
        for metric in metrics:
            print(f"  -> Running metric={metric} for dataset={dataset_name}")
            if(dataset_name == "Oceania"):
                process_single_metric(
                    folder=dataset_path,
                    metric=metric,
                    algorithms=algorithms,
                    draw=True,
                    input_folder = dataset_input_folder, 
                    output_folder=dataset_output_folder
                )
        print(f"=== Finished dataset: {dataset_name} ===\n")
        
        # Add megaplot functionality
        G_dummy = nx.Graph()
        straight_dummy = StraightLine(G_dummy)
        experiment = Experiment(straight_dummy, straight_dummy)
        
        graph_images = []
        image_output_folder = "output"
        images_base_path = os.path.join(image_output_folder, "airlines", "images")
        for alg in algorithms:
            image_path = os.path.join(images_base_path, f"{alg}.png")
            if os.path.exists(image_path):
                graph_images.append(image_path)
            else:
                print(f"[Warning] Missing graph image: {image_path}")

        histogram_images = []
        for alg in algorithms:
            images_base_path = os.path.join(output_folder, "airlines", "images")
            for alg in algorithms:
                hist_path = os.path.join(images_base_path, "violin_plot_distortion.png")
                if os.path.exists(hist_path):
                    histogram_images.append(hist_path)
                else:
                    print(f"[Warning] Missing histogram image: {hist_path}")

            if graph_images and histogram_images:
                experiment.plotMegaGraph(
                    algorithms,
                    "output_comparison",
                    histogram_images,
                    [],
                    dataset_input_folder,
                    dataset_output_folder
                )
                print(f"[INFO] Megaplot created for {dataset_name}.")
            else:
                print(f"[Warning] Skipping megaplot due to missing images.")

if __name__ == "__main__":
    run_all_experiments(
        input_folder="testingDatabases",
        output_folder="results"
    )
