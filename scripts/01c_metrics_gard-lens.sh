#!/bin/bash
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=40            # Number of tasks per node (1 task to initialize Dask cluster on each node)
#SBATCH --cpus-per-task=1
#SBATCH --mem=180GB                    # Memory per node (adjust as needed)
#SBATCH --time=24:00:00                # Maximum run time (adjust as needed)
#SBATCH --exclusive
#SBATCH --output=/home/fs01/dcl257/projects/conus_comparison_lafferty-etal-2024/code/logs/01c_metrics_gard-lens_%j.out

echo "Job started on `hostname` at `date`"

cd /home/fs01/dcl257/projects/conus_comparison_lafferty-etal-2024/code

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ~/miniforge3/envs/climate-stack-2024-10/

# Run your Dask script
python 01c_metrics_gard-lens.py

echo "Job Ended at `date`"