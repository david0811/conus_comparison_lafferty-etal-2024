#!/bin/bash
#SBATCH --output=../scripts/logs/jobs/%x.log
#SBATCH --error=../scripts/logs/jobs/%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --time=04:00:00
#SBATCH --account=open

# Code directory
CODE_DIR="/storage/home/dcl5300/work/current_projects/conus_comparison_lafferty-etal-2024"
cd $CODE_DIR

echo "Job started on $(hostname) at $(date)"

# Read arguments
ENSEMBLE=$1
GCM=$2
MEMBER=$3
SSP=$4
METRIC_ID=$5

echo "ENSEMBLE: $ENSEMBLE"
echo "GCM: $GCM"
echo "MEMBER: $MEMBER"
echo "SSP: $SSP"
echo "METRIC_ID: $METRIC_ID"

# Load modules
module load r/4.4.2

# Run
uv run src/fit_gev_nonstat_mle_R.py --ensemble $ENSEMBLE --gcm $GCM --member $MEMBER --ssp $SSP --metric_id $METRIC_ID

echo "Job completed at $(date)"