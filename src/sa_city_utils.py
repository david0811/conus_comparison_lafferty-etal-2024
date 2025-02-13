import os
from glob import glob

import dask
import numpy as np
import pandas as pd
import xarray as xr

import sa_utils as sau
from utils import city_list, get_unique_loca_metrics, loca_gard_mapping
from utils import roar_data_path as project_data_path

#####################################
# Calculating city timeseries metrics
#####################################


def get_city_timeseries(
    city,
    ensemble,
    gcm,
    member,
    ssp,
    metric_id,
    project_data_path=project_data_path,
):
    """
    Reads and returns the annual max timeseries for a
    selected city, ensemble, GCMs, SSPs.
    """
    # Read file
    if ensemble == "LOCA2":
        files = glob(
            f"{project_data_path}/metrics/LOCA2/{metric_id}_{gcm}_{member}_{ssp}_*.nc"
        )
        ds = xr.concat([xr.open_dataset(file) for file in files], dim="time")
    else:
        ds = xr.open_dataset(
            f"{project_data_path}/metrics/{ensemble}/{metric_id}_{gcm}_{member}_{ssp}.nc"
        )

    # Update GARD GCMs
    gcm_name = (
        gcm.replace("canesm5", "CanESM5")
        .replace("ecearth3", "EC-Earth3")
        .replace("cesm2", "CESM2-LENS")
    )
    # Fix LOCA CESM mapping
    if ensemble == "LOCA2" and gcm == "CESM2-LENS":
        member_name = (
            loca_gard_mapping[member]
            if member in loca_gard_mapping.keys()
            else member
        )
    else:
        member_name = member

    # Add all info
    ds = ds.expand_dims(
        {
            "gcm": [gcm_name],
            "member": [member_name],
            "ssp": [ssp],
            "ensemble": [ensemble],
        }
    )
    ds["time"] = ds["time"].dt.year

    # Extract city data
    lat, lon = city_list[city]
    if ensemble == "LOCA2":
        df_loc = (
            ds.sel(lat=lat, lon=360 + lon, method="nearest")
            .to_dataframe()
            .drop(columns=["lat", "lon"])
            .dropna()
        )
    elif ensemble == "STAR-ESDM":
        df_loc = (
            ds.sel(latitude=lat, longitude=360 + lon, method="nearest")
            .to_dataframe()
            .drop(columns=["latitude", "longitude"])
            .dropna()
        )
    else:
        df_loc = (
            ds.sel(lat=lat, lon=lon, method="nearest")
            .to_dataframe()
            .dropna()
            .drop(columns=["lat", "lon"])
        )

    # Return
    return df_loc.reset_index()


def get_city_timeseries_all(
    city,
    metric_id,
    project_data_path=project_data_path,
):
    """
    Loop through all meta-ensemble members and calculate the city timeseries.
    """
    # Check if done
    if os.path.exists(
        f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv"
    ):
        return None

    #### LOCA2
    ensemble = "LOCA2"
    df_loca = get_unique_loca_metrics(metric_id)

    # Loop through
    delayed = []
    for index, row in df_loca.iterrows():
        # Get info
        gcm, member, ssp = row["gcm"], row["member"], row["ssp"]

        out = dask.delayed(get_city_timeseries)(
            city=city,
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
        )
        delayed.append(out)

    #### STAR-ESDM
    ensemble = "STAR-ESDM"
    files = glob(f"{project_data_path}/metrics/{ensemble}/{metric_id}_*")

    # Loop through
    for file in files:
        # Get info
        _, _, gcm, member, ssp = file.split("/")[-1].split(".")[0].split("_")

        # Calculate
        out = dask.delayed(get_city_timeseries)(
            city=city,
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
        )
        delayed.append(out)

    #### GARD-LENS
    ensemble = "GARD-LENS"
    files = glob(f"{project_data_path}/metrics/{ensemble}/{metric_id}_*")

    # Loop through
    for file in files:
        # Get info
        info = file.split("/")[-1].split("_")
        gcm = info[2]
        ssp = info[-1].split(".")[0]
        member = f"{info[3]}_{info[4]}" if gcm == "cesm2" else info[3]

        # Calculate
        out = dask.delayed(get_city_timeseries)(
            city=city,
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
        )
        delayed.append(out)

    # Compute all
    df = pd.concat(dask.compute(*delayed))

    # Store
    df.to_csv(
        f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv",
        index=False,
    )


#################################
# Store all city GEV results
#################################


def remap_latlon(ds):
    # Make sure lat/lon is named correctly
    if "latitude" in ds.dims and "longitude" in ds.dims:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    # Set lon to [-180,180] if it is not already in that range
    if ds["lon"].max() > 180:
        ds["lon"] = ((ds["lon"] + 180) % 360) - 180

    return ds


def store_all_cities(
    metric_id,
    grid,
    regrid_method,
    proj_slice,
    hist_slice,
    stationary,
    fit_method,
    cols_to_keep,
    col_identifier,
    city_list,
):
    """
    Store all cities GEV results as csv files for a given metric.
    """
    stat_str = "stat" if stationary else "nonstat"
    grid_names = {
        "LOCA2": "loca_grid",
        "GARD-LENS": "gard_grid",
        "original": "original_grid/freq",
    }
    # Check if done for all cities
    if grid == "original":
        regrid_str = ""
    else:
        regrid_str = f"_{regrid_method}"

    file_names = [
        f"{city}_{metric_id}_{proj_slice}_{hist_slice}_{col_identifier}_{fit_method}_{stat_str}{regrid_str}.csv"
        for city in list(city_list.keys())
    ]

    if not np.all(
        [
            os.path.exists(
                f"{project_data_path}/extreme_value/cities/{grid_names[grid]}/{file_name}"
            )
            for file_name in file_names
        ]
    ):
        # Read all
        if grid == "original":
            ds_loca = sau.read_loca(
                metric_id=metric_id,
                grid="LOCA2",
                regrid_method=None,
                proj_slice=proj_slice,
                hist_slice=hist_slice,
                stationary=stationary,
                fit_method=fit_method,
                cols_to_keep=cols_to_keep,
            )
            ds_star = sau.read_star(
                metric_id=metric_id,
                grid="STAR-ESDM",
                regrid_method=None,
                proj_slice=proj_slice,
                hist_slice=hist_slice,
                stationary=stationary,
                fit_method=fit_method,
                cols_to_keep=cols_to_keep,
            )
            ds_gard = sau.read_gard(
                metric_id=metric_id,
                grid="GARD-LENS",
                regrid_method=None,
                proj_slice=proj_slice,
                hist_slice=hist_slice,
                stationary=stationary,
                fit_method=fit_method,
                cols_to_keep=cols_to_keep,
            )
        else:
            ds_loca, ds_star, ds_gard = sau.read_all(
                metric_id=metric_id,
                grid=grid,
                regrid_method=regrid_method,
                proj_slice=proj_slice,
                hist_slice=hist_slice,
                stationary=stationary,
                fit_method=fit_method,
                cols_to_keep=cols_to_keep,
            )

        # Remap lat/lons
        ds_loca = remap_latlon(ds_loca)
        ds_star = remap_latlon(ds_star)
        ds_gard = remap_latlon(ds_gard)

        # Loop through cities
        for city in city_list:
            # Read
            lat, lon = city_list[city]
            df_loca = (
                ds_loca.sel(lat=lat, lon=lon, method="nearest")
                .to_dataframe()
                .dropna()
                .drop(columns=["lat", "lon"])
                .reset_index()
            )
            df_star = (
                ds_star.sel(lat=lat, lon=lon, method="nearest")
                .to_dataframe()
                .dropna()
                .drop(columns=["lat", "lon"])
                .reset_index()
            )
            df_gard = (
                ds_gard.sel(lat=lat, lon=lon, method="nearest")
                .to_dataframe()
                .dropna()
                .drop(columns=["lat", "lon"])
                .reset_index()
            )

            # Concat
            df_all = pd.concat([df_loca, df_star, df_gard])

            # Store
            if grid == "original":
                regrid_str = ""
            else:
                regrid_str = f"_{regrid_method}"

            file_name = f"{city}_{metric_id}_{proj_slice}_{hist_slice}_{col_identifier}_{fit_method}_{stat_str}{regrid_str}.csv"
            df_all.to_csv(
                f"{project_data_path}/extreme_value/cities/{grid_names[grid]}/{file_name}",
                index=False,
            )
