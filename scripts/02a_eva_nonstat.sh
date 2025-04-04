#!/bin/bash
#SBATCH --output=./logs/jobs/meta_eva_nonstat-%a.log
#SBATCH --error=./logs/jobs/meta_eva_nonstat-%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=2-00:00:00
#SBATCH --partition=basic

# This script creates a parameter file and submits a SLURM job for each climate output. It will
# monitor the queue and ensure no more than $MAX_QUEUED jobs are running at any time.

#######################################################################
# Set the metric IDs to search
METRIC_IDS=("max_pr" "max_tasmax" "min_tasmin" "max_cdd" "max_hdd")

# Define allowed ensembles
ALLOWED_ENSEMBLES=("STAR-ESDM" "LOCA2" "GARD-LENS")

# Set bootstrap or not
BOOTSTRAP=0
#######################################################################

# Get climate info for all
PROJECT_DATA_DIR="/storage/group/pches/default/users/dcl5300/conus_comparison_lafferty-etal-2024/"
CLIMATE_INFO_FILE="${PROJECT_DATA_DIR}/ensemble_info.csv"
PARAM_FILE="${PROJECT_DATA_DIR}/job_params.txt"

# Check if CSV file exists
if [ ! -f "$CLIMATE_INFO_FILE" ]; then
    echo "Error: CSV file not found at $CLIMATE_INFO_FILE"
    exit 1
fi

# Clear the parameter file if it exists
> $PARAM_FILE

# Skip header line and read CSV file
sed 1d "$CLIMATE_INFO_FILE" | while IFS=, read -r ensemble gcm member ssp remainder
do
    # Trim potential whitespace
    ensemble=$(echo "$ensemble" | xargs)
    gcm=$(echo "$gcm" | xargs)
    member=$(echo "$member" | xargs)
    ssp=$(echo "$ssp" | xargs)
    
    # Check if current ensemble is in the allowed list
    if [[ " ${ALLOWED_ENSEMBLES[@]} " =~ " ${ensemble} " ]]; then
        echo "Processing: Ensemble=$ensemble, GCM=$gcm, Member=$member, SSP=$ssp"
        
        # Process each metric ID for the current row
        for metric_id in "${METRIC_IDS[@]}"; do
            # Check if done
            SUFFIX=$([ "$BOOTSTRAP" -eq 1 ] && echo "100boot" || echo "main")
            if [ -f "${PROJECT_DATA_DIR}/extreme_value/original_grid/${metric_id}/${ensemble}_${gcm}_${member}_${ssp}_1950-2100_nonstat_mle_${SUFFIX}.nc" ]; then
                echo "  Skipping: $metric_id, $ensemble, $gcm, $member, $ssp (already done)"
                continue
            fi
            
            # Add parameters to file
            echo "$ensemble $gcm $member $ssp $metric_id" >> $PARAM_FILE
        done
    fi
done

# Count the number of jobs
TOTAL_JOBS=$(wc -l < $PARAM_FILE)
MAX_QUEUED=10  # Set to just below your submission limit
SLEEP_TIME=600  # 10 minutes between checks

for ((i=1; i<=$TOTAL_JOBS; i++)); do
    # Check current queue count
    while true; do
        QUEUE_COUNT=$(squeue -u $USER -p open | wc -l)
        QUEUE_COUNT=$((QUEUE_COUNT - 1))  # Subtract header line
        
        if [ $QUEUE_COUNT -lt $MAX_QUEUED ]; then
            break
        fi
        
        echo "$(date): Queue has $QUEUE_COUNT jobs. Waiting for count to drop below $MAX_QUEUED..."
        sleep $SLEEP_TIME
    done
    
    # Submit next job
    PARAMS=$(sed -n "${i}p" $PARAM_FILE)
    read ENSEMBLE GCM MEMBER SSP METRIC_ID <<< $PARAMS
    JOB_NAME="${METRIC_ID}_${ENSEMBLE}_${GCM}_${MEMBER}_${SSP}"
    
    echo "$(date): Submitting job $i/$TOTAL_JOBS: $JOB_NAME"
    sbatch -J $JOB_NAME ../src/submit_gev_nonstat.sh $ENSEMBLE $GCM $MEMBER $SSP $METRIC_ID $BOOTSTRAP
    
    # Small delay between submissions to not overwhelm scheduler
    sleep 5
done

# Remove parameter file
rm $PARAM_FILE

# # Test
# job_name="test_job_5gb"
# ensemble="GARD-LENS"
# gcm="canesm5"
# member="r1i1p1f1"
# ssp="ssp370"
# metric_id="min_tasmin"
# bootstrap=1
# echo "Submitting job: $job_name"
# sbatch -J $job_name ../src/submit_gev_nonstat.sh $ensemble $gcm $member $ssp $metric_id $bootstrap