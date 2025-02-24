from gen_graph import gen_graphs, circular_graph
from bundle_pipeline import bundle_all
#from metrics_pipeline import all_metrics

import os 
if not os.path.isdir("inputs/"): os.mkdir("inputs")

for i in range(10):
    gen_graphs(i)
    circular_graph(i, d=(i / 2 + 2))

bundle_all("inputs/")

# all_metrics("inputs/")