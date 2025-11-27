#!/bin/bash
#SBATCH --job-name=metrics_pipeline
#SBATCH --output=logs/out_%A_%a.log
#SBATCH --error=logs/err_%A_%a.log
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-158  # Update this: 0 to (num_graphs - 1)
# #SBATCH --mail-user=YOUR_EMAIL   # optional
# #SBATCH --mail-type=END,FAIL     # optional

set -euo pipefail

# Activate conda environment
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate bundling
cd ~/shared-bundling-repo

# Keep logs tidy and outputs timestamped
STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p logs
mkdir -p short_outputs

# Optionally suppress the networkx runtime warning
export PYTHONWARNINGS="ignore:networkx backend defined more than once:RuntimeWarning"

# Set input folder and metrics
INPUT_FOLDER="inputs/all_outputs/"
METRICS="['ambiguity', 'clustering']"

# Print job information
echo "Starting job ${SLURM_ARRAY_TASK_ID} on $(hostname) at ${STAMP}"
echo "Working directory: $(pwd)"

# Run the script with the SLURM array task ID
python metrics_pipeline.py \
    --folder "${INPUT_FOLDER}" \
    --metric "${METRICS}" \
    --job-index ${SLURM_ARRAY_TASK_ID} \
    --smartorder True \
    --verbose False

echo "Done. Job ${SLURM_ARRAY_TASK_ID} completed at $(date)"
