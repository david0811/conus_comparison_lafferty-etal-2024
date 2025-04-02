#!/bin/bash
#SBATCH --output=../scripts/logs/jobs/array_%A_%x.log
#SBATCH --error=../scripts/logs/jobs/array_%A_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=10GB
#SBATCH --time=01:00:00
#SBATCH --account=open

# Read the parameter file
PARAM_FILE=$1

# Get the parameters for this task
PARAMS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $PARAM_FILE)
read ENSEMBLE GCM MEMBER SSP METRIC_ID <<< $PARAMS

# Code directory
CODE_DIR="/storage/home/dcl5300/work/current_projects/conus_comparison_lafferty-etal-2024"
cd $CODE_DIR

echo "Job started on $(hostname) at $(date)"
echo "SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

echo "ENSEMBLE: $ENSEMBLE"
echo "GCM: $GCM"
echo "MEMBER: $MEMBER"
echo "SSP: $SSP"
echo "METRIC_ID: $METRIC_ID"

# Load modules
module load r/4.4.2

# Run
uv run src/fit_gev_nonstat_mle.py --ensemble $ENSEMBLE --gcm $GCM --member $MEMBER --ssp $SSP --metric_id $METRIC_ID

echo "Job completed at $(date)"