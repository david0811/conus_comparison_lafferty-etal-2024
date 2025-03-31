#!/bin/bash

# Get climate info for all
PROJECT_DATA_DIR="/storage/group/pches/default/users/dcl5300/conus_comparison_lafferty-etal-2024/"
CLIMATE_INFO_FILE="${PROJECT_DATA_DIR}/all_climate_info.csv"

# Set the metric IDs to search for
# METRIC_IDS=("max_pr" "max_tasmax" "min_tasmin")
METRIC_IDS=("max_pr")

# Check if CSV file exists
if [ ! -f "$CLIMATE_INFO_FILE" ]; then
    echo "Error: CSV file not found at $CLIMATE_INFO_FILE"
    exit 1
fi

# # Skip header line and read CSV file
# sed 1d "$CLIMATE_INFO_FILE" | while IFS=, read -r ensemble gcm member ssp remainder
# do
#     # Trim potential whitespace
#     ensemble=$(echo "$ensemble" | xargs)
#     gcm=$(echo "$gcm" | xargs)
#     member=$(echo "$member" | xargs)
#     ssp=$(echo "$ssp" | xargs)
    
#     echo "Processing: Ensemble=$ensemble, GCM=$gcm, Member=$member, SSP=$ssp"
    
#     # Process each metric ID for the current row
#     for metric_id in "${METRIC_IDS[@]}"; do
#         echo "  For metric: $metric_id"
        
#         # Create a job name
#         job_name="${metric_id}_${ensemble}_${gcm}_${member}_${ssp}"
        
#         # Submit SLURM job
#         sbatch -J $job_name ../src/submit_gev_nonstat_R.sh $ensemble $gcm $member $ssp $metric_id
        
#         echo "  Submitted job: $job_name"
#     done
# done

# Test
job_name="test_job"
ensemble="LOCA2"
gcm="CanESM5"
member="r1i1p1f1"
ssp="ssp370"
metric_id="max_pr"
echo "Submitting job: $job_name"
sbatch -J $job_name ../src/submit_gev_nonstat_R.sh $ensemble $gcm $member $ssp $metric_id