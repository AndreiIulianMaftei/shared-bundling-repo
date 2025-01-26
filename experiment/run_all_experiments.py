import os
import networkx as nx

# Import your existing code
from bundle_pipeline import read_bundling
from other_code.EPB.abstractBundling import GWIDTH
from other_code.EPB.experiments import Experiment
from other_code.EPB.straight import StraightLine
from other_code.EPB.clustering import Clustering
from other_code.EPB.plot import Plot
from other_code.EPB.plotTest import PlotTest  # We'll customize this to accept an output folder.

# ---------------------------------------------------------
# Example metrics list WITHOUT 'ink'
# ---------------------------------------------------------
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
]

algorithms = ["fd", "epb", "wr"]

def process_single_metric(folder: str, metric: str, algorithms: list, draw: bool, output_folder: str):
    """
    Process a single metric for a dataset folder containing multiple algorithm results,
    saving all images / plots to `output_folder`.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Expect a file named epb.graphml
    epb_graph_path = os.path.join(folder, "epb.graphml")
    if not os.path.isfile(epb_graph_path):
        print(f"[Warning] No 'epb.graphml' found in: {folder}")
        return

    # Read main reference graph
    G = nx.read_graphml(epb_graph_path)
    G = nx.Graph(G)

    # Prepare straight line reference
    straight = StraightLine(G)
    straight.scaleToWidth(GWIDTH)
    straight.bundle()

    # Create an "images" directory in output_folder for (optional) drawn edges
    images_dir = os.path.join(output_folder, "images")
    if draw and not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Optionally draw the straight layout
    if draw:
        straight.draw(images_dir)

    # We'll store results for plotting
    # The format is list of (dataArray, "whichAlgorithm").
    rez_all = []

    # Precompute edges from the reference "straight" for intersection checks
    experiment_for_all_edges = Experiment(straight, straight)
    all_edges_ref = experiment_for_all_edges.all_edges()

    # Evaluate each chosen algorithm
    for alg in algorithms:
        bundling = read_bundling(folder, alg)
        if bundling is None:
            print(f"[Warning] Could not read bundling '{alg}' in: {folder}")
            continue

        # Draw if requested
        if draw:
            bundling.draw(images_dir)

        experiment = Experiment(bundling, straight)

        # Evaluate metrics
        if metric == "intersect_all":
            intersections = experiment.all_intersection(all_edges_ref)
            print(f"[{alg}] total intersections: {intersections}")
            rez_all.append((intersections, alg))

        elif metric == "distortion":
            (
                dist_min,
                dist_mean,
                dist_std,
                dist_var,
                dist_median,
                distortions,
            ) = experiment.calcDistortion()
            rez_all.append((distortions, alg))

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
            # If 'all' is chosen, do multiple computations
            distortions = experiment.calcDistortion()[5]
            frechet_vals = experiment.fastFrechet(alg)
            monotonicities = experiment.calcMonotonicity(alg)
            # Add them all to rez_all with identifying labels
            rez_all.append((distortions, f"distortion-{alg}"))
            rez_all.append((frechet_vals, f"frechet-{alg}"))
            rez_all.append((monotonicities, f"monotonicity-{alg}"))

    # If no results, skip
    if len(rez_all) == 0:
        return

    # Generate final plots.
    # We pass `output_folder` to PlotTest so it can store plots in the correct place.
    plotter = PlotTest(output_folder=output_folder)

    if metric == "monotonicity_projection":
        plotter.plotProjectedMonotonicity(rez_all)
    elif metric == "monotonicity":
        plotter.plotMonotonicity(rez_all)
    elif metric == "distortion":
        plotter.plotDistortionHistogram(rez_all)
    elif metric == "frechet":
        plotter.plotFrechet(rez_all)
    elif metric == "angle":
        plotter.plotAngles(rez_all)
    elif metric == "self_intersect":
        plotter.plotSelfIntersect(rez_all)
    elif metric == "all":
        # If you did multiple computations, you might produce multiple plots here
        pass


def run_all_experiments(input_folder="testingDatabases", output_folder="results"):
    """
    Look in `testingDatabases/` for subfolders (each subfolder = a dataset).
    For each subfolder with 'epb.graphml', run all metrics in `metrics`,
    then save the results in `results/<dataset_name>/`.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over subfolders in testingDatabases
    for dataset_name in os.listdir(input_folder):
        dataset_path = os.path.join(input_folder, dataset_name)
        if not os.path.isdir(dataset_path):
            continue  # skip files, only process subdirectories

        epb_path = os.path.join(dataset_path, "epb.graphml")
        if not os.path.isfile(epb_path):
            print(f"[Info] Skipping {dataset_name}, no epb.graphml found.")
            continue

        print(f"=== Found dataset: {dataset_name} ===")

        # Make a dedicated results subfolder
        dataset_output_folder = os.path.join(output_folder, dataset_name)
        if not os.path.exists(dataset_output_folder):
            os.makedirs(dataset_output_folder)

        # Now run each metric on that dataset
        for metric in metrics:
            print(f"  -> Running metric={metric} for dataset={dataset_name}")
            process_single_metric(
                folder=dataset_path,
                metric=metric,
                algorithms=algorithms,
                draw=True,  # choose whether to draw or not
                output_folder=dataset_output_folder
            )

        print(f"=== Finished dataset: {dataset_name} ===\n")


if __name__ == "__main__":
    run_all_experiments(
        input_folder="testingDatabases",
        output_folder="results"
    )
