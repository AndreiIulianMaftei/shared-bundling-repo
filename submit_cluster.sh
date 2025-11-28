#!/bin/bash
# Helper script to submit jobs to the Ginkgo cluster

# Python3 is available by default on the cluster
# First, count the number of graphs
echo "Counting graphs in inputs/all_outputs/..."
python3 count_jobs.py inputs/all_outputs/

# Get the count
NUM_GRAPHS=$(python3 -c "import os; print(len([d for d in os.listdir('inputs/all_outputs/') if os.path.isdir(os.path.join('inputs/all_outputs/', d))]))")
ARRAY_MAX=$((NUM_GRAPHS - 1))

# Check if we got a valid number
if [ -z "$NUM_GRAPHS" ] || [ "$NUM_GRAPHS" -eq 0 ]; then
    echo "Error: No graphs found or could not count graphs"
    exit 1
fi

echo ""
echo "Found ${NUM_GRAPHS} graphs"
echo "Submitting SLURM array job with range 0-${ARRAY_MAX}"
echo ""

# Submit the job with the correct array size
sbatch --array=0-${ARRAY_MAX} run_metrics_parallel.slurm

echo ""
echo "Job submitted! Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/metrics_*"
