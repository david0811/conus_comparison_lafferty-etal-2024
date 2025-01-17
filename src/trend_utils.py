import os
from glob import glob

import dask
import numpy as np
import pandas as pd
import xarray as xr

from utils import loca_gard_mapping
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path


# Linear regression function
def linear_regression(X, y):
    if not np.isfinite(y).all():
        return np.array([np.nan, np.nan])
    else:
        return np.polyfit(X, y, 1)


# Fit trend for single output
def trend_fit_single(
    ensemble,
    gcm,
    member,
    ssp,
    metric_id,
    years,
    store_path,
    project_data_path=project_data_path,
    project_code_path=project_code_path,
):
    """
    Read a single metric file and fit the GEV.
    """
    try:
        # Check if done
        if years == [1950, 2014]:
            ssp_name = "historical"
        else:
            ssp_name = ssp
        time_name = f"{years[0]}-{years[1]}" if years is not None else "all"
        store_name = f"{ensemble}_{gcm}_{member}_{ssp_name}_{time_name}.nc"

        if os.path.exists(f"{store_path}/{store_name}"):
            return None

        # Read file
        if ensemble == "LOCA2":
            files = glob(
                f"{project_data_path}/metrics/LOCA2/{metric_id}_{gcm}_{member}_{ssp}_*.nc"
            )
            ds = xr.concat(
                [xr.open_dataset(file) for file in files], dim="time"
            )
        else:
            ds = xr.open_dataset(
                f"{project_data_path}/metrics/{ensemble}/{metric_id}_{gcm}_{member}_{ssp}.nc"
            )

        # Apply time slice if needed
        ds["time"] = ds["time"].dt.year
        if years is not None:
            ds = ds.sel(time=slice(years[0], years[1]))

        # Fit trend
        var_id = metric_id.split("_")[1]

        # Linear trend
        result = xr.apply_ufunc(
            linear_regression,
            ds["time"],  # input x data
            ds[var_id],  # input y data
            input_core_dims=[
                ["time"],
                ["time"],
            ],  # specify core dimensions for inputs
            output_core_dims=[["coef"]],  # specify core dimensions for output
            vectorize=True,  # apply function element-wise
            dask="forbidden",  # enable parallelization with dask
            output_dtypes=[float],  # specify output data type
            dask_gufunc_kwargs={"output_sizes": {"coef": 2}},
        )
        ds_out = xr.Dataset({metric_id: result})
        ds_out["coef"] = ["trend", "intcp"]

        ## Store
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
        ds_out = ds_out.expand_dims(
            {
                "gcm": [gcm_name],
                "member": [member_name],
                "ssp": [ssp_name],
                "ensemble": [ensemble],
            }
        )
        ds_out.to_netcdf(f"{store_path}/{store_name}")
    # Log if error
    except Exception as e:
        except_path = f"{project_code_path}/scripts/logs"
        with open(
            f"{except_path}/{ensemble}_{gcm}_{member}_{ssp}_{metric_id}.txt",
            "w",
        ) as f:
            f.write(str(e))


###############################
# Trend fit across whole ensemble
###############################
def get_unique_loca_metrics(metric_id):
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


def trend_fit_all(
    metric_id, future_years=[2015, 2100], hist_years=[1950, 2014]
):
    """
    Fits a trend to the entire meta-ensemble of outputs.
    """
    # Store results location
    store_path = f"{project_data_path}/trends/original_grids/{metric_id}"

    #### LOCA2
    ensemble = "LOCA2"
    df_loca = get_unique_loca_metrics(metric_id)

    # Loop through
    delayed = []
    for index, row in df_loca.iterrows():
        # Get info
        gcm, member, ssp = row["gcm"], row["member"], row["ssp"]
        years = hist_years if ssp == "historical" else future_years

        out = dask.delayed(trend_fit_single)(
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
            years=years,
            store_path=store_path,
        )
        delayed.append(out)

    #### STAR-ESDM
    ensemble = "STAR-ESDM"
    files = glob(f"{project_data_path}/metrics/{ensemble}/{metric_id}_*")

    # Loop through
    for file in files:
        # Get info
        _, _, gcm, member, ssp = file.split("/")[-1].split(".")[0].split("_")

        # Fit for historical and ssp
        for ssp_id in ["historical", ssp]:
            years = hist_years if ssp_id == "historical" else future_years
            out = dask.delayed(trend_fit_single)(
                ensemble=ensemble,
                gcm=gcm,
                member=member,
                ssp=ssp,
                metric_id=metric_id,
                years=years,
                store_path=store_path,
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

        # Do for historical and ssp
        for ssp_id in ["historical", ssp]:
            years = hist_years if ssp_id == "historical" else future_years
            out = dask.delayed(trend_fit_single)(
                ensemble=ensemble,
                gcm=gcm,
                member=member,
                ssp=ssp,
                metric_id=metric_id,
                years=years,
                store_path=store_path,
            )
            delayed.append(out)

    # Compute all
    _ = dask.compute(*delayed)
