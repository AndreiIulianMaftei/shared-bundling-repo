from gen_graph import gen_graphs
from bundle_pipeline import bundle_all
from metrics_pipeline import all_metrics

import os 
if not os.path.isdir("inputs/"): os.mkdir("inputs")
for i in range(10):
    gen_graphs(i)

bundle_all("inputs/")

# all_metrics("inputs/")