from glob import glob

import pandas as pd

################
# Main metrics
################
gev_metric_ids = ["max_tasmax", "min_tasmin", "max_pr", "max_cdd", "max_hdd"]
trend_metric_ids = [
    "avg_tas",
    "avg_tasmax",
    "avg_tasmin",
    "sum_pr",
    "sum_hdd",
    "sum_cdd",
    "max_tasmax",
    "min_tasmin",
    "max_pr",
]

################
# Climate info
################
gard_gcms = ["CanESM5", "CESM2-LENS", "EC-Earth3"]
ensembles = ["LOCA2", "GARD-LENS", "STAR-ESDM"]
ssps = ["ssp245", "ssp370", "ssp585"]
tgw_scenarios = [
    "historical",
    "rcp45cooler",
    "rcp85cooler",
    "rcp45hotter",
    "rcp85hotter",
]

################
# Paths
################
roar_code_path = (
    "/storage/home/dcl5300/work/current_projects/conus_comparison_lafferty-etal-2024"
)
roar_data_path = (
    "/storage/group/pches/default/users/dcl5300/conus_comparison_lafferty-etal-2024"
)
hopper_code_path = "/home/fs01/dcl257/projects/conus_comparison_lafferty-etal-2024"
hopper_data_path = "/home/fs01/dcl257/data/conus_comparison_lafferty-etal-2024"

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
    "boston": [42.360278, -71.057778],
    "raleigh": [35.854167, -78.761944],
    "orlando": [28.5384, -81.3789],
    "atlanta": [33.749, -84.388],
    "stlouis": [38.6274, -90.1982],
    "minneapolis": [44.9778, -93.265],
    "bozeman": [45.6793, -111.0373],
    "albuquerque": [35.0844, -106.6504],
    "oklahoma_city": [35.4689, -97.5195],
    "lasvegas": [36.1716, -115.1391],
    "sandiego": [32.7157, -117.1611],
    "pittsburgh": [40.4406, -79.9959],
    "boise": [43.615, -116.2023],
    "bismarck": [46.8042, -100.7878],
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
# Other
#################################
def map_store_names(ensemble, gcm, member):
    """
    Map GCM and ensemble member names to standardized storage naming conventions.
    
    This function standardizes GCM (Global Climate Model) names and maps ensemble
    member identifiers, particularly handling special cases for LOCA2 ensemble data.
    
    Args:
        ensemble (str): The ensemble name (e.g., "LOCA2", "GARD").
        gcm (str): The GCM identifier, which may use abbreviated naming conventions.
        member (str): The ensemble member identifier.
    
    Returns:
        tuple: A tuple containing:
            - gcm_name (str): The standardized GCM name.
            - member_name (str): The mapped or original ensemble member identifier.
    
    Notes:
        - GARD GCM names are automatically converted to their full names:
          "canesm5" -> "CanESM5", "ecearth3" -> "EC-Earth3", "cesm2" -> "CESM2-LENS"
        - For LOCA2 ensemble with CESM2-LENS GCM, member names are mapped using
          the loca_gard_mapping dictionary if available.
        - Otherwise, member names are returned unchanged.
    """
    # Update GARD GCMs
    gcm_name = (
        gcm.replace("canesm5", "CanESM5")
        .replace("ecearth3", "EC-Earth3")
        .replace("cesm2", "CESM2-LENS")
    )

    # Fix LOCA CESM mapping
    if ensemble == "LOCA2" and gcm == "CESM2-LENS":
        member_name = (
            loca_gard_mapping[member] if member in loca_gard_mapping.keys() else member
        )
    else:
        member_name = member

    return gcm_name, member_name



def check_data_length(data, ensemble, gcm, ssp, years, include_neighbors=False):
    """
    Check length function.
    If data is None, just return the expected length.
    """
    # If including neighbors, add 4 extra years
    if include_neighbors:
        multiplier = 5
    else:
        multiplier = 1

    # Check length is as expected
    if (
        ensemble == "GARD-LENS"
        and gcm in ["ecearth3", "EC-Earth3"]
        and years[0] == 1950
    ):
        expected_length = (years[1] - 1970 + 1) * multiplier  # GARD-LENS EC-Earth3
        if data is not None:
            assert len(data) == expected_length, (
                f"ds length is {len(data)}, expected {expected_length}"
            )
    else:
        expected_length = (years[1] - years[0] + 1) * multiplier
        if data is not None:
            assert len(data) == expected_length, (
                f"ds length is {len(data)}, expected {expected_length}"
            )
    return expected_length


def get_starting_year(ensemble, gcm, ssp, years):
    """
    Get starting year function.
    """
    # Check length is as expected
    if ensemble == "GARD-LENS" and gcm in ["ecearth3", "EC-Earth3"]:
        starting_year = 1970
    else:
        starting_year = 1950
    return starting_year


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
                pd.DataFrame({"gcm": gcm, "member": member, "ssp": ssp}, index=[0]),
            ]
        )

    # Return unique
    return df.drop_duplicates().reset_index()
