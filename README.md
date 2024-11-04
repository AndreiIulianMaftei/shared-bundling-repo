# shared-bundling-repo

# Testing

two ways of testing code: computing a bundled drawing or compute metric(s) of a bundled drawing

## Computing a bundling

this is handled with the file compute_single_bundling.py. Here, we have to give as argument an input file (.graphml with x,y positions) and one bundling algorithm. For example:

$python compute_single_bundling.py --algorithm epb --file input_data/airlines.graphml

At the moment, it will compute the bundling and store the output in the folder output/{filename}/{algorithm.graphml} in the case of EPB or SEPB. In case of other algorithms it could potentially store them as .node and .edge files. E.g. FD.node and FD.edge.

## Computing a metric

this is handled with the file compute_single_metric.py. It has a similar structure. We need to give it a folder, an algorithm and a metric as argument

$python compute_single_metric.py --algorithm epb --folder output/airlines --metric ink

In the example we assume that the output was computed with the other script above and is stored in output/airlines. It checks if this is a folder and will then look for the file {folder}/epb.graphml. The loading for a specific algorithm needs to be handled. see the example for EPB. Once this is done, it will try to compute the metric 'ink'.