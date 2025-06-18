#!/bin/bash
#SBATCH --output=../scripts/logs/jobs/%x.log
#SBATCH --error=../scripts/logs/jobs/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80GB
#SBATCH --time=12:00:00
#SBATCH --partition=basic
########## SBATCH --account=pches_cr_default

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
BOOTSTRAP=$6
SCALE=$7

echo "ENSEMBLE: $ENSEMBLE"
echo "GCM: $GCM"
echo "MEMBER: $MEMBER"
echo "SSP: $SSP"
echo "METRIC_ID: $METRIC_ID"
echo "BOOTSTRAP: $BOOTSTRAP"
echo "SCALE: $SCALE"

# Run
uv run src/fit_gev_nonstat_mle.py --ensemble $ENSEMBLE --gcm $GCM --member $MEMBER --ssp $SSP --metric_id $METRIC_ID --bootstrap $BOOTSTRAP --scale $SCALE

echo "Job completed at $(date)"