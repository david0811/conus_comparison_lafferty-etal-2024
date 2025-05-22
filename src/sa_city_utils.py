import os
from glob import glob

import dask
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs

from utils import city_list, get_unique_loca_metrics, loca_gard_mapping, tgw_scenarios
from utils import roar_data_path as project_data_path

#####################################
# Calculating city timeseries metrics
#####################################
def get_nearest_cells(ds, target_lat, target_lon, ensemble):
    """
    Select the central cell and its immediate north, south, east, and west neighbors.
    Assumes lat/lon are 1D and sorted.

    Parameters:
    - ds: xarray.Dataset or xarray.DataArray with lat/lon coordinates
    - target_lat, target_lon: float, target location

    Returns:
    - xarray.Dataset or DataArray with selected 5 grid cells
    """

    # Find nearest lat/lon indices
    if ensemble == "TGW":
        lat_name = "south_north"
        lon_name = "west_east"
        proj_string = "+proj=lcc +lat_0=40.0000076293945 +lon_0=-97 +lat_1=30 +lat_2=45 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs"
        ds_crs = ccrs.CRS(proj_string)
        x, y = ds_crs.transform_point(
            target_lon, target_lat, src_crs=ccrs.PlateCarree()
        )
        lat_idx = np.abs(ds[lat_name] - y).argmin().item()
        lon_idx = np.abs(ds[lon_name] - x).argmin().item()
    elif ensemble == "LOCA2":
        lat_name = "lat"
        lon_name = "lon"
        lat_idx = np.abs(ds[lat_name] - target_lat).argmin().item()
        lon_idx = np.abs(ds[lon_name] - (360. + target_lon)).argmin().item()
    elif ensemble == "STAR-ESDM":
        lat_name = "latitude"
        lon_name = "longitude"
        lat_idx = np.abs(ds[lat_name] - target_lat).argmin().item()
        lon_idx = np.abs(ds[lon_name] - (360. + target_lon)).argmin().item()
    else:
        lat_name = "lat"
        lon_name = "lon"
        lat_idx = np.abs(ds[lat_name] - target_lat).argmin().item()
        lon_idx = np.abs(ds[lon_name] - target_lon).argmin().item()

    # Neighbor indices (handle boundaries)
    lat_indices = [lat_idx]
    if lat_idx > 0:
        lat_indices.append(lat_idx - 1)  # south
    if lat_idx < ds.sizes[lat_name] - 1:
        lat_indices.append(lat_idx + 1)  # north

    lon_indices = [lon_idx]
    if lon_idx > 0:
        lon_indices.append(lon_idx - 1)  # west
    if lon_idx < ds.sizes[lon_name] - 1:
        lon_indices.append(lon_idx + 1)  # east

    # Create boolean mask to select only center + N/S/E/W
    selected = ds.isel(
        {
            lat_name: xr.DataArray(lat_indices, dims="points"),
            lon_name: xr.DataArray(lon_indices, dims="points"),
        }
    )

    # Now mask to retain only the 5 unique (lat, lon) pairs: center, N/S/E/W
    # Build (lat, lon) pairs around the center
    pairs = [
        (lat_idx, lon_idx, "center"),  # center
        (lat_idx - 1, lon_idx, "south") if lat_idx > 0 else None,  # south
        (lat_idx + 1, lon_idx, "north")
        if lat_idx < ds.sizes[lat_name] - 1
        else None,  # north
        (lat_idx, lon_idx - 1, "west") if lon_idx > 0 else None,  # west
        (lat_idx, lon_idx + 1, "east")
        if lon_idx < ds.sizes[lon_name] - 1
        else None,  # east
    ]
    pairs = [p for p in pairs if p is not None]

    # Stack to filter
    selected = ds.isel(
        {
            lat_name: xr.DataArray([p[0] for p in pairs], dims="points"),
            lon_name: xr.DataArray([p[1] for p in pairs], dims="points"),
        }
    )

    # Add point type coordinate
    selected.coords["point"] = xr.DataArray([p[2] for p in pairs], dims="points")

    return (
        selected.to_dataframe()
        .dropna()
        .reset_index()[[
            "point",
            "gcm", 
            "ssp",
            "member", 
            "ensemble",
            "time",
            list(ds.keys())[0],
            ]]
    )


def select_point(ds, lat, lon, ensemble, include_neighbors=False):
    """
    Select the gridpoint for a given city.
    """
    if include_neighbors:
        df_loc = get_nearest_cells(ds, lat, lon, ensemble)
    else:
        if ensemble == "LOCA2":
            df_loc = (
                ds.sel(lat=lat, lon=360 + lon, method="nearest")
                .to_dataframe()
                .drop(columns=["lat", "lon"])
                .dropna().reset_index()
            )
        elif ensemble == "STAR-ESDM":
            df_loc = (
                ds.sel(latitude=lat, longitude=360 + lon, method="nearest")
                .to_dataframe()
                .drop(columns=["latitude", "longitude"])
                .dropna().reset_index()
            )
        elif ensemble == "TGW":
            # Get projection string
            # https://tgw-data.msdlive.org/
            proj_string = "+proj=lcc +lat_0=40.0000076293945 +lon_0=-97 +lat_1=30 +lat_2=45 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs"
            ds_crs = ccrs.CRS(proj_string)

            # Select location
            x, y = ds_crs.transform_point(lon, lat, src_crs=ccrs.PlateCarree())
            ds_sel = ds.sel({"west_east": x, "south_north": y}, method="nearest")

            df_loc = ds_sel.to_dataframe().drop(
                columns=["lat", "lon", "west_east", "south_north"]
            )
        else:
            df_loc = (
                ds.sel(lat=lat, lon=lon, method="nearest")
                .to_dataframe()
                .drop(columns=["lat", "lon"])
                .dropna().reset_index()
            )

    return df_loc


def get_city_timeseries(
    city,
    ensemble,
    gcm,
    member,
    ssp,
    metric_id,
    include_neighbors=False,
    project_data_path=project_data_path,
):
    """
    Reads and returns the annual max timeseries for a
    selected city, ensemble, GCMs, SSPs.
    """
    # Read file
    try:
        if ensemble == "LOCA2":
            files = glob(
                f"{project_data_path}/metrics/LOCA2/{metric_id}_{gcm}_{member}_{ssp}_*.nc"
            )
            ds = xr.concat([xr.open_dataset(file) for file in files], dim="time")
        elif ensemble == "TGW":
            files = glob(f"{project_data_path}/metrics/TGW/{metric_id}_{ssp}_*.nc")
            ds = xr.concat(
                [xr.open_dataset(file).isel(time=slice(None, -1)) for file in files],
                dim="time",
            )
        else:
            ds = xr.open_dataset(
                f"{project_data_path}/metrics/{ensemble}/{metric_id}_{gcm}_{member}_{ssp}.nc"
            )

        # Update GARD GCMs
        gcm_name = (
            gcm.replace("canesm5", "CanESM5")
            .replace("cesm2", "CESM2-LENS")
            .replace("ecearth3", "EC-Earth3")
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
        
        df_loc = select_point(
            ds, lat, lon, ensemble, include_neighbors=include_neighbors
        )

        # Return
        return df_loc

    except Exception as e:
        print(f"Error reading {city} {metric_id} {ensemble} {gcm} {member} {ssp}: {e}")
        return None


def get_city_timeseries_all(
    city,
    metric_id,
    include_neighbors=False,
    project_data_path=project_data_path,
):
    """
    Loop through all meta-ensemble members and calculate the city timeseries.
    """
    # Check if done
    if include_neighbors:
        store_path = f"{project_data_path}/metrics/cities/{city}_{metric_id}_neighbors.csv"
    else:
        store_path = f"{project_data_path}/metrics/cities/{city}_{metric_id}_neighbors.csv"
    
    if os.path.exists(store_path):
        return None

    delayed = []
    
    #### LOCA2
    ensemble = "LOCA2"
    df_loca = get_unique_loca_metrics(metric_id)

    # Loop through
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
            include_neighbors=include_neighbors
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
            include_neighbors=include_neighbors
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
            include_neighbors=include_neighbors
        )
        delayed.append(out)

    #### TGW
    ensemble = "TGW"
    gcm = "none"
    member = "none"

    # Loop through scenarios
    for ssp in tgw_scenarios:
        # Calculate
        out = dask.delayed(get_city_timeseries)(
            city=city,
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
            include_neighbors=include_neighbors
        )
        delayed.append(out)

    # Compute all
    df = pd.concat(dask.compute(*delayed), ignore_index=True)

    # Store
    df.to_csv(store_path, index=False,
    )


#######################
# UC for city df
#######################
def calculate_df_uc(df, plot_col, n_min_members=5):
    """
    Calculate the uncertainty decomposition based on pd DataFrame.
    """

    # Just in case: drop TaiESM1 from STAR (too hot!)
    if "STAR-ESDM" in df["ensemble"].unique():
        df = df[df["member"] != "TaiESM1"]

    # Range functions
    def get_range(x):
        return x.max() - x.min()

    def get_quantile_range(df, groupby_cols, plot_col):
        df_tmp = pd.merge(
            df[df["quantile"] == "q975"].rename(
                columns={plot_col: f"{plot_col}_upper"}
            ),
            df[df["quantile"] == "q025"].rename(
                columns={plot_col: f"{plot_col}_lower"}
            ),
            on=groupby_cols,
        )

        df_tmp[f"{plot_col}_95range"] = (
            df_tmp[f"{plot_col}_upper"] - df_tmp[f"{plot_col}_lower"]
        )
        return df_tmp

    # Get combos to include
    if "n_boot" in df.columns:
        df_main = df[df["n_boot"] == "main"]
        df_boot = (
            df[df["n_boot"] != "main"]
            .groupby(["ensemble", "gcm", "member", "ssp"])
            .quantile([0.025, 0.975], numeric_only=True)
            .reset_index()
            .rename(columns={"level_4": "quantile"})
        )
        # Map quantiles to strings
        df_boot["quantile"] = df_boot["quantile"].map({0.025: "q025", 0.975: "q975"})
    elif "quantile" in df.columns:
        df_main = df[df["quantile"] == "main"]
        df_boot = df[df["quantile"] != "main"]
    else:
        df_main = df
        df_boot = None

    combos_to_include = (
        df_main.groupby(["ensemble", "gcm", "ssp"]).count()[plot_col] >= n_min_members
    )

    # Scenario uncertainty
    ssp_uc_by_gcm = (
        df_main.groupby(["ensemble", "gcm", "ssp"])[plot_col]
        .mean()
        .loc[combos_to_include]
        .groupby(["gcm", "ensemble"])
        .apply(get_range)
    )
    ssp_uc_by_gcm_mean = ssp_uc_by_gcm.replace(0.0, np.nan).mean()
    ssp_uc_by_gcm_std = ssp_uc_by_gcm.replace(0.0, np.nan).std()

    ssp_uc = (
        df_main.groupby(["ensemble", "ssp"])[plot_col]
        .mean()
        .groupby("ensemble")
        .apply(get_range)
    )
    ssp_uc_mean = ssp_uc.replace(0.0, np.nan).mean()
    ssp_uc_std = ssp_uc.replace(0.0, np.nan).std()

    # Response uncertainty
    gcm_uc = (
        df_main.groupby(["ensemble", "gcm", "ssp"])[plot_col]
        .mean()
        .loc[combos_to_include]
        .groupby(["ssp", "ensemble"])
        .apply(get_range)
    )
    gcm_uc_mean = gcm_uc.replace(0.0, np.nan).mean()
    gcm_uc_std = gcm_uc.replace(0.0, np.nan).std()

    # Internal variability
    iv_uc = (
        df_main.groupby(["ensemble", "gcm", "ssp"])[plot_col]
        .apply(get_range)
        .loc[combos_to_include]
    )
    iv_uc_mean = iv_uc.replace(0.0, np.nan).mean()
    iv_uc_std = iv_uc.replace(0.0, np.nan).std()

    # Downscaling uncertainty
    ds_uc = df_main.groupby(["gcm", "ssp", "member"])[plot_col].apply(get_range)
    ds_uc_mean = ds_uc.replace(0.0, np.nan).mean()
    ds_uc_std = ds_uc.replace(0.0, np.nan).std()

    # Total uncertainty
    if "n_boot" in df.columns:
        df_samples = df[df["n_boot"] != "main"]
        uc_99w_boot = df_samples[plot_col].quantile(0.995) - df_samples[
            plot_col
        ].quantile(0.005)
        uc_99w_main = df_main[plot_col].quantile(0.995) - df_main[plot_col].quantile(
            0.005
        )
    elif "quantile" in df.columns:
        upper = df[df["quantile"] == "q975"][plot_col].quantile(0.995)
        lower = df[df["quantile"] == "q025"][plot_col].quantile(0.005)
        uc_99w_boot = upper - lower
        uc_99w_main = df_main[plot_col].quantile(0.995) - df_main[plot_col].quantile(
            0.005
        )
    else:
        uc_99w_main = df[plot_col].quantile(0.995) - df[plot_col].quantile(0.005)
        uc_99w_boot = np.nan

    # Fit uncertainty if included
    fit_uc = get_quantile_range(
        df=df_boot,
        groupby_cols=["gcm", "ensemble", "member", "ssp"],
        plot_col=plot_col,
    )
    fit_uc_mean = fit_uc[f"{plot_col}_95range"].mean()
    fit_uc_std = fit_uc[f"{plot_col}_95range"].std()

    # Return all
    return pd.DataFrame(
        {
            "uncertainty_type": [
                "ssp_uc",
                "ssp_uc_by_gcm",
                "gcm_uc",
                "iv_uc",
                "dsc_uc",
                "fit_uc",
                "uc_99w_boot",
                "uc_99w_main",
            ],
            "mean": [
                ssp_uc_mean,
                ssp_uc_by_gcm_mean,
                gcm_uc_mean,
                iv_uc_mean,
                ds_uc_mean,
                fit_uc_mean,
                uc_99w_boot,
                uc_99w_main,
            ],
            "std": [
                ssp_uc_std,
                ssp_uc_by_gcm_std,
                gcm_uc_std,
                iv_uc_std,
                ds_uc_std,
                fit_uc_std,
                uc_99w_boot,
                uc_99w_main,
            ],
        }
    )


# #################################
# # Store all city GEV results
# #################################
# def remap_latlon(ds):
#     # Make sure lat/lon is named correctly
#     if "latitude" in ds.dims and "longitude" in ds.dims:
#         ds = ds.rename({"latitude": "lat", "longitude": "lon"})
#     # Set lon to [-180,180] if it is not already in that range
#     if ds["lon"].max() > 180:
#         ds["lon"] = ((ds["lon"] + 180) % 360) - 180

#     return ds


# def store_all_cities(
#     metric_id,
#     grid,
#     regrid_method,
#     proj_slice,
#     hist_slice,
#     stationary,
#     fit_method,
#     cols_to_keep,
#     col_identifier,
#     city_list,
# ):
#     """
#     Store all cities GEV results as csv files for a given metric.
#     """
#     stat_str = "stat" if stationary else "nonstat"
#     grid_names = {
#         "LOCA2": "loca_grid",
#         "GARD-LENS": "gard_grid",
#         "original": "original_grid/freq",
#     }
#     # Check if done for all cities
#     if grid == "original":
#         regrid_str = ""
#     else:
#         regrid_str = f"_{regrid_method}"

#     file_names = [
#         f"{city}_{metric_id}_{proj_slice}_{hist_slice}_{col_identifier}_{fit_method}_{stat_str}{regrid_str}.csv"
#         for city in list(city_list.keys())
#     ]

#     if not np.all(
#         [
#             os.path.exists(
#                 f"{project_data_path}/extreme_value/cities/{grid_names[grid]}/{file_name}"
#             )
#             for file_name in file_names
#         ]
#     ):
#         # Read all
#         if grid == "original":
#             ds_loca = sau.read_loca(
#                 metric_id=metric_id,
#                 grid="LOCA2",
#                 regrid_method=None,
#                 proj_slice=proj_slice,
#                 hist_slice=hist_slice,
#                 stationary=stationary,
#                 fit_method=fit_method,
#                 cols_to_keep=cols_to_keep,
#             )
#             ds_star = sau.read_star(
#                 metric_id=metric_id,
#                 grid="STAR-ESDM",
#                 regrid_method=None,
#                 proj_slice=proj_slice,
#                 hist_slice=hist_slice,
#                 stationary=stationary,
#                 fit_method=fit_method,
#                 cols_to_keep=cols_to_keep,
#             )
#             ds_gard = sau.read_gard(
#                 metric_id=metric_id,
#                 grid="GARD-LENS",
#                 regrid_method=None,
#                 proj_slice=proj_slice,
#                 hist_slice=hist_slice,
#                 stationary=stationary,
#                 fit_method=fit_method,
#                 cols_to_keep=cols_to_keep,
#             )
#         else:
#             ds_loca, ds_star, ds_gard = sau.read_all(
#                 metric_id=metric_id,
#                 grid=grid,
#                 regrid_method=regrid_method,
#                 proj_slice=proj_slice,
#                 hist_slice=hist_slice,
#                 stationary=stationary,
#                 fit_method=fit_method,
#                 cols_to_keep=cols_to_keep,
#             )

#         # Remap lat/lons
#         ds_loca = remap_latlon(ds_loca)
#         ds_star = remap_latlon(ds_star)
#         ds_gard = remap_latlon(ds_gard)

#         # Loop through cities
#         for city in city_list:
#             # Read
#             lat, lon = city_list[city]
#             df_loca = (
#                 ds_loca.sel(lat=lat, lon=lon, method="nearest")
#                 .to_dataframe()
#                 .dropna()
#                 .drop(columns=["lat", "lon"])
#                 .reset_index()
#             )
#             df_star = (
#                 ds_star.sel(lat=lat, lon=lon, method="nearest")
#                 .to_dataframe()
#                 .dropna()
#                 .drop(columns=["lat", "lon"])
#                 .reset_index()
#             )
#             df_gard = (
#                 ds_gard.sel(lat=lat, lon=lon, method="nearest")
#                 .to_dataframe()
#                 .dropna()
#                 .drop(columns=["lat", "lon"])
#                 .reset_index()
#             )

#             # Concat
#             df_all = pd.concat([df_loca, df_star, df_gard])

#             # Store
#             if grid == "original":
#                 regrid_str = ""
#             else:
#                 regrid_str = f"_{regrid_method}"

#             file_name = f"{city}_{metric_id}_{proj_slice}_{hist_slice}_{col_identifier}_{fit_method}_{stat_str}{regrid_str}.csv"
#             df_all.to_csv(
#                 f"{project_data_path}/extreme_value/cities/{grid_names[grid]}/{file_name}",
#                 index=False,
#             )

# def calculate_df_uc_bayesian(df, plot_col, n_min_members=5):
#     """
#     Calculate the uncertainty decomposition based on pd DataFrame.
#     """
#     get_range = lambda x: x.max() - x.min()

#     def calculate_quantile_range(df, groupby_cols, plot_col):
#         df_tmp = pd.merge(
#             df[df["quantile"] == "q975"].rename(
#                 columns={plot_col: f"{plot_col}_upper"}
#             ),
#             df[df["quantile"] == "p025"].rename(
#                 columns={plot_col: f"{plot_col}_lower"}
#             ),
#             on=groupby_cols,
#         )

#         df_tmp[f"{plot_col}_diff"] = (
#             df_tmp[f"{plot_col}_upper"] - df_tmp[f"{plot_col}_lower"]
#         )
#         return df_tmp

#     combos_to_include = (
#         df.groupby(["ensemble", "gcm", "ssp"]).count()[plot_col] >= n_min_members
#     )

#     # Regular uncertainties with median
#     df_median = df[df["quantile"] == "main"]

#     # Scenario uncertainty
#     ssp_uc_by_gcm = (
#         df_median.groupby(["ensemble", "gcm", "ssp"])[plot_col]
#         .mean()
#         .loc[combos_to_include]
#         .groupby(["gcm", "ensemble"])
#         .apply(get_range)
#         .replace(0.0, np.nan)
#         .mean()
#     )
#     ssp_uc = (
#         df_median.groupby(["ensemble", "ssp"])[plot_col]
#         .mean()
#         .groupby("ensemble")
#         .apply(get_range)
#         .replace(0.0, np.nan)
#         .mean()
#     )

#     # Response uncertainty
#     gcm_uc = (
#         df_median.groupby(["ensemble", "gcm", "ssp"])[plot_col]
#         .mean()
#         .loc[combos_to_include]
#         .groupby(["ssp", "ensemble"])
#         .apply(get_range)
#         .replace(0.0, np.nan)
#         .mean()
#     )

#     # Internal variability
#     iv_uc = (
#         df_median.groupby(["ensemble", "gcm", "ssp"])[plot_col]
#         .apply(get_range)
#         .loc[combos_to_include]
#         .replace(0.0, np.nan)
#         .mean()
#     )

#     # Downscaling uncertainty
#     ds_uc = (
#         df_median.groupby(["gcm", "ssp", "member"])[plot_col]
#         .apply(get_range)
#         .replace(0.0, np.nan)
#         .mean()
#     )

#     # Fit uncertainty
#     gev_uc = calculate_quantile_range(
#         df=df,
#         groupby_cols=["gcm", "ensemble", "member", "ssp"],
#         plot_col=plot_col,
#     )[f"{plot_col}_diff"].mean()

#     # Return all
#     return pd.DataFrame(
#         {
#             "ssp_uc": [ssp_uc],
#             "ssp_uc_by_gcm": [ssp_uc_by_gcm],
#             "gcm_uc": [gcm_uc],
#             "iv_uc": [iv_uc],
#             "dsc_uc": [ds_uc],
#             "gev_uc": [gev_uc],
#         }
#     )
