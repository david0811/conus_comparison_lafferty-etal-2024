import os
from glob import glob

import dask
import numpy as np
import xarray as xr

from utils import check_data_length, loca_gard_mapping
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path
from utils import get_unique_loca_metrics


# Fit trend for single output
def avg_calc_single(
    ensemble,
    gcm,
    member,
    ssp,
    metric_id,
    proj_years,
    store_path,
    hist_years=None,
    project_data_path=project_data_path,
    project_code_path=project_code_path,
):
    """
    Read a single metric file and calculate the average.
    """
    if proj_years is None and hist_years is None:
        return None
    
    try:
        # Check if done
        proj_name = (
            f"_{proj_years[0]}-{proj_years[1]}" if proj_years is not None else ""
        )
        hist_name = (
            f"_{hist_years[0]}-{hist_years[1]}" if hist_years is not None else ""
        )
        store_name = f"{ensemble}_{gcm}_{member}_{ssp}{proj_name}{hist_name}.nc"

        if os.path.exists(f"{store_path}/{store_name}"):
            return None

        ## Read data file
        # LOCA2
        if ensemble == "LOCA2":
            # Projection
            proj_files = glob(
                f"{project_data_path}/metrics/LOCA2/{metric_id}_{gcm}_{member}_{ssp}_*.nc"
            )
            ds = xr.concat([xr.open_dataset(file) for file in proj_files], dim="time")
            # Historical
            ds_hist = xr.open_dataset(
                f"{project_data_path}/metrics/LOCA2/{metric_id}_{gcm}_{member}_historical_1950-2014.nc"
            )
            ds = xr.concat([ds_hist, ds], dim="time")
        # GARD-LENS
        elif ensemble == "GARD-LENS":
            ds = xr.open_dataset(
                f"{project_data_path}/metrics/GARD-LENS/{metric_id}_{gcm}_{member}_{ssp}.nc"
            )
        # STAR-ESDM
        elif ensemble == "STAR-ESDM":
            ds = xr.open_dataset(
                f"{project_data_path}/metrics/STAR-ESDM/{metric_id}_{gcm}_{member}_{ssp}.nc"
            )
        else:
            raise ValueError(f"Ensemble {ensemble} not supported")

        # Apply time slices
        ds["time"] = ds["time"].dt.year

        if proj_years is not None:
            ds_proj = ds.sel(time=slice(proj_years[0], proj_years[1]))
        if hist_years is not None:
            ds_hist = ds.sel(time=slice(hist_years[0], hist_years[1]))

        # Check length is as expected
        if proj_years is not None:
            expected_length = check_data_length(
                ds_proj["time"], ensemble, gcm, ssp, proj_years
            )
        if hist_years is not None:
            expected_length = check_data_length(
                ds_hist["time"], ensemble, gcm, ssp, hist_years
            )

        # Calculate mean
        if proj_years is not None:
            ds_proj_out = ds_proj.mean(dim="time")
        if hist_years is not None:
            ds_hist_out = ds_hist.mean(dim="time")

        # Combine outputs based on what was calculated
        if proj_years is not None and hist_years is not None:
            # Both periods: calculate difference
            ds_out = ds_proj_out - ds_hist_out
        elif proj_years is not None:
            # Only projection period
            ds_out = ds_proj_out
        elif hist_years is not None:
            # Only historical period
            ds_out = ds_hist_out

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
                "ssp": [ssp],
                "ensemble": [ensemble],
            }
        )
        # Make sure not all NaNs
        var_id = metric_id.split("_")[1]
        assert np.count_nonzero(~np.isnan(ds_out[var_id])), "all NaNs"
        # Or alternatively, check number of unique values
        assert len(np.unique(ds_out[var_id])) > 1, "all values are identical"

        # Store
        ds_out.to_netcdf(f"{store_path}/{store_name}")

    # Log if error
    except Exception as e:
        except_path = f"{project_code_path}/scripts/logs/avgs"
        with open(
            f"{except_path}/{ensemble}_{gcm}_{member}_{ssp}_{metric_id}.txt",
            "w",
        ) as f:
            f.write(str(e))


##################################
# Trend fit across whole ensemble
##################################
def avg_calc_all(metric_id, proj_years, hist_years=None):
    """
    Fits a trend to the entire meta-ensemble of outputs.
    """
    # Store results location
    store_path = f"{project_data_path}/averages/original_grid/{metric_id}"

    #### LOCA2
    ensemble = "LOCA2"
    df_loca = get_unique_loca_metrics(metric_id)

    # Loop through
    delayed = []
    for index, row in df_loca.iterrows():
        # Get info
        gcm, member, ssp = row["gcm"], row["member"], row["ssp"]
        if ssp == "historical":
            continue
        out = dask.delayed(avg_calc_single)(
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
            proj_years=proj_years,
            hist_years=hist_years,
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

        # Calculate
        out = dask.delayed(avg_calc_single)(
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
            proj_years=proj_years,
            hist_years=hist_years,
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

        # Calculate
        out = dask.delayed(avg_calc_single)(
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
            proj_years=proj_years,
            hist_years=hist_years,
            store_path=store_path,
        )
        delayed.append(out)

    # Compute all
    _ = dask.compute(*delayed)
