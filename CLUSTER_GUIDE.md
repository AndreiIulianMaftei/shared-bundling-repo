# Ginkgo Cluster Setup Guide

## Initial Setup

1. **Connect to the cluster** (from MWN network or via eduVPN):
```bash
ssh username@cluster.ginkgo-project.de
# or if DNS issues:
ssh username@10.152.225.230
```

2. **Upload your code and data**:
```bash
# From your local machine
scp -r /path/to/shared-bundling-repo username@cluster.ginkgo-project.de:/storage/home/username/
```

3. **Set up Python environment on the cluster**:
```bash
# Load Python module
module load python

# Or if using conda (use Miniforge, not Anaconda!)
# Download miniforge
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# Create environment
conda create -n bundling python=3.9
conda activate bundling

# Install dependencies
pip install -r requirements.txt
```

## Running Jobs

### Option 1: Automatic Submission
```bash
# Make scripts executable
chmod +x submit_cluster.sh run_metrics_parallel.sh count_jobs.py

# Submit all jobs at once
./submit_cluster.sh
```

### Option 2: Manual Submission
```bash
# First, check how many graphs you have
python count_jobs.py inputs/all_outputs/

# Submit with the correct array size (if you have 7 graphs, use 0-6)
sbatch --array=0-6 run_metrics_parallel.sh
```

### Option 3: Test on a Single Graph First
```bash
# Run interactively to test
salloc -n 1 -t 02:00:00
python metrics_pipeline.py --folder inputs/all_outputs/ --job-index 0 --metric "['ambiguity', 'clustering']"
exit
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job_id>

# View output logs (they appear in logs/ directory)
tail -f logs/metrics_<job_id>_<array_id>.out

# Check errors
tail -f logs/metrics_<job_id>_<array_id>.err

# Cancel a job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER

# Cancel specific array tasks
scancel <job_id>_[0-5]  # cancels tasks 0-5
```

## Data Locations

- **Home directory**: `/storage/home/<username>/` (backed up daily)
- **No-backup storage**: `/storage/nobackup/<username>/` or `~/nobackup` (use for large datasets)
- **Local scratch**: Available via symlink in your home (fast, not permanent)

Recommendation: Keep code in home directory, put large input data in `~/nobackup/inputs/`

## Modifying the SLURM Script

Edit `run_metrics_parallel.sh` to adjust:

- `--time=24:00:00` - Maximum runtime per job
- `--mem=16G` - Memory per job (increase if needed)
- `--cpus-per-task=1` - Number of CPU cores
- `INPUT_FOLDER` - Path to your input data
- `METRICS` - Which metrics to compute

## Useful Commands

```bash
# See available software
module avail

# Load specific module
module load python

# Check loaded modules
module list

# See cluster resources
sinfo

# See detailed node info
scontrol show node

# Get job efficiency statistics (after job completes)
seff <job_id>
```

## Tips

1. **Test first**: Run a single job interactively before submitting array jobs
2. **Resource limits**: The cluster has limited resources, don't request more than you need
3. **Login node**: Don't run heavy computations on the login node
4. **Storage**: Use `~/nobackup` for large datasets to avoid filling up backed-up storage
5. **Long jobs**: For jobs >24h, you may need to adjust time limits or split work differently
