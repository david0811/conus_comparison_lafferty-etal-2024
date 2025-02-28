from glob import glob

import pandas as pd

################
# Main metrics
################
gev_metric_ids = ["max_tasmax", "min_tasmin", "max_cdd", "max_hdd", "max_pr"]
trend_metric_ids = ["avg_tas", "sum_pr", "sum_hdd", "sum_cdd"]

################
# Climate info
################
gard_gcms = ["CanESM5", "CESM2-LENS", "EC-Earth3"]
ensembles = ["LOCA2", "GARD-LENS", "STAR-ESDM"]
ssps = ["ssp245", "ssp370", "ssp585"]

################
# Paths
################
roar_code_path = "/storage/home/dcl5300/work/current_projects/conus_comparison_lafferty-etal-2024"
roar_data_path = "/storage/group/pches/default/users/dcl5300/conus_comparison_lafferty-etal-2024"
hopper_code_path = (
    "/home/fs01/dcl257/projects/conus_comparison_lafferty-etal-2024"
)
hopper_data_path = (
    "/home/fs01/dcl257/projects/data/conus_comparison_lafferty-etal-2024"
)

################
# Cities
################
city_list = {
    "chicago": [41.881944, -87.627778],
    "seattle": [47.609722, -122.333056],
    "houston": [29.762778, -95.383056],
    "denver": [39.7392, -104.985],
    "nyc": [40.712778, -74.006111],
    "sanfrancisco": [37.7775, -122.416389],
}

#########################################
# Mappping GARD members to LOCA members
#########################################
loca_gard_mapping = {
    "r1i1p1f1": "1001_01",
    "r2i1p1f1": "1021_02",
    "r3i1p1f1": "1041_03",
    "r4i1p1f1": "1061_04",
    "r5i1p1f1": "1081_05",
    "r6i1p1f1": "1101_06",
    "r7i1p1f1": "1121_07",
    "r8i1p1f1": "1141_08",
    "r9i1p1f1": "1161_09",
    "r10i1p1f1": "1181_10",
}


#################################
# LOCA members
#################################
def get_unique_loca_metrics(metric_id, project_data_path=roar_data_path):
    """
    Return unique LOCA2 combinations for given metric_id.
    """
    # Read all
    files = glob(f"{project_data_path}/metrics/LOCA2/{metric_id}_*")

    # Extract all info
    df = pd.DataFrame(columns=["gcm", "member", "ssp"])
    for file in files:
        _, _, gcm, member, ssp, _ = file.split("/")[-1].split("_")
        df = pd.concat(
            [
                df,
                pd.DataFrame(
                    {"gcm": gcm, "member": member, "ssp": ssp}, index=[0]
                ),
            ]
        )

    # Return unique
    return df.drop_duplicates().reset_index()
