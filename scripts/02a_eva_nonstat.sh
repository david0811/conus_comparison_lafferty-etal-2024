#!/bin/bash

#  This script loops through all climate outputs defined in CLIMATE_INFO_FILE and submits
#  a SLURM jobscript to fit a non-stationary GEV using a linear time trend in the location parameter.

#######################################################################
# Set the metric IDs to search
# METRIC_IDS=("max_pr" "max_tasmax" "min_tasmin")
METRIC_IDS=("max_tasmax" "max_pr")

# Define allowed ensembles
ALLOWED_ENSEMBLES=("STAR-ESDM" "LOCA2" "GARD-LENS")
# ALLOWED_ENSEMBLES=("GARD-LENS")
#######################################################################

# Get climate info for all
PROJECT_DATA_DIR="/storage/group/pches/default/users/dcl5300/conus_comparison_lafferty-etal-2024/"
CLIMATE_INFO_FILE="${PROJECT_DATA_DIR}/all_climate_info.csv"

# Check if CSV file exists
if [ ! -f "$CLIMATE_INFO_FILE" ]; then
    echo "Error: CSV file not found at $CLIMATE_INFO_FILE"
    exit 1
fi

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
            echo "  For metric: $metric_id"

            # Check if done
            if [ -f "${PROJECT_DATA_DIR}/extreme_value/original_grid/${metric_id}/${ensemble}_${gcm}_${member}_${ssp}_1950-2100_nonstat_mle_main.nc" ]; then
                echo "  Skipping: $metric_id, $ensemble, $gcm, $member, $ssp (already done)"
                continue
            fi
            
            # Create a job name
            job_name="${metric_id}_${ensemble}_${gcm}_${member}_${ssp}"
            
            # Submit SLURM job
            sbatch -J $job_name ../src/submit_gev_nonstat.sh $ensemble $gcm $member $ssp $metric_id
            
            echo "  Submitted job: $job_name"
        done
    fi
done

# # Test
# job_name="test_job_10gb"
# ensemble="GARD-LENS"
# gcm="canesm5"
# member="r2i1p1f1"
# ssp="ssp370"
# metric_id="max_pr"
# echo "Submitting job: $job_name"
# sbatch -J $job_name ../src/submit_gev_nonstat.sh $ensemble $gcm $member $ssp $metric_id