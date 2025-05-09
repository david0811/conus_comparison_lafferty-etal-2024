import numpy as np
import xarray as xr


###########################
# Variable transformations
###########################
def transform_precipitation(ds_in, var_id):
    if var_id == "pr":
        if "pr" not in ds_in.variables:
            if "pcp" in ds_in.variables:
                # STAR
                ds_in = ds_in.rename({"pcp": "pr"})
            elif "prcp" in ds_in.variables:
                # Obs
                ds_in = ds_in.rename({"prcp": "pr"})
            else:
                raise ValueError(f"No precipitation variable found in {ds_in.name}")

        ds_in.pr.attrs["units"] = "mm"
    return ds_in


def transform_temperature(ds_in, var_id):
    """
    Gets the correct temperature variable from each possible ensemble.
    """
    if var_id == "tas":
        if "tas" not in ds_in.variables:
            # STAR
            if "t_mean" in ds_in.variables:  # STAR
                ds_in = ds_in.rename({"t_mean": "tas"})
            # Obs
            elif "tmean" in ds_in.variables:
                ds_in = ds_in.rename({"tmean": "tas"})
            # LOCA2
            elif "tasmin" in ds_in.variables and "tasmax" in ds_in.variables:
                ds_in["tas"] = (ds_in["tasmax"] + ds_in["tasmin"]) / 2
    elif var_id == "tasmin":
        if "tasmin" not in ds_in.variables:
            if "tmin" in ds_in.variables:
                # Obs
                ds_in = ds_in.rename({"tmin": "tasmin"})
            else:
                # STAR
                ds_in["tasmin"] = ds_in["t_mean"] - ds_in["t_range"] / 2.0
    elif var_id == "tasmax":
        if "tasmax" not in ds_in.variables:
            if "tmax" in ds_in.variables:
                # Obs
                ds_in = ds_in.rename({"tmax": "tasmax"})
            else:
                # STAR
                ds_in["tasmax"] = ds_in["t_mean"] + ds_in["t_range"] / 2.0
    elif var_id in ["cdd", "hdd"]:
        if "tasmin" not in ds_in.variables or "tasmax" not in ds_in.variables:
            if "tmin" in ds_in.variables and "tmax" in ds_in.variables:
                # Obs
                ds_in = ds_in.rename({"tmin": "tasmin", "tmax": "tasmax"})
            else:
                # STAR
                ds_in["tasmin"] = ds_in["t_mean"] - ds_in["t_range"] / 2.0
                ds_in["tasmax"] = ds_in["t_mean"] + ds_in["t_range"] / 2.0

    return ds_in


####################################
# Metric calculation functions
####################################
def calculate_avg(ds_in, var_id):
    ds_in = transform_temperature(ds_in, var_id)
    return ds_in[[var_id]].resample(time="YE").mean()


def calculate_sum(ds_in, var_id):
    # Assume we only ever want precip sum
    ds_in = transform_precipitation(ds_in, var_id)
    ds_out = ds_in[[var_id]].resample(time="YE").sum()
    return ds_out


def calculate_max(ds_in, var_id):
    # Might want to calculate temp or precip max
    if var_id == "pr":
        ds_in = transform_precipitation(ds_in, var_id)
    elif var_id in ["tas", "tasmin", "tasmax"]:
        ds_in = transform_temperature(ds_in, var_id)

    ds_out = ds_in[[var_id]].resample(time="YE").max()

    return ds_out


def calculate_min(ds_in, var_id):
    # Assume we only ever to temperature minima
    ds_in = transform_temperature(ds_in, var_id)
    ds_out = ds_in[[var_id]].resample(time="YE").min()

    return ds_out


#####################################
# Degree day calculate functions
#####################################
def f_to_c(fahrenheit):
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5.0 / 9.0


thresh_c = f_to_c(65.0)


# CDD
def calculate_cdd(tasmin, tasmax, b):
    # Calculate t_bar only for the valid range where tasmin < b < tasmax
    t_bar = np.arccos(np.clip((2 * b - tasmax - tasmin) / (tasmax - tasmin), -1, 1))

    # CDD formula components
    cdd_thresh_below = (tasmax + tasmin) / 2.0 - b
    cdd_thresh_between = (t_bar / np.pi) * ((tasmax + tasmin) / 2 - b) + (
        tasmax - tasmin
    ) / (2 * np.pi) * np.sin(t_bar)
    cdd_thresh_above = 0.0

    return np.where(
        b <= tasmin,
        cdd_thresh_below,
        np.where(b >= tasmax, cdd_thresh_above, cdd_thresh_between),
    )


# ufunc for dask
def cdd_ufunc(tasmin, tasmax, threshold=thresh_c):
    return xr.apply_ufunc(calculate_cdd, tasmin, tasmax, threshold, dask="allowed")


# HDD
def calculate_hdd(tasmin, tasmax, b):
    # Calculate t_bar only for the valid range where tasmin < b < tasmax
    t_bar = np.arccos(np.clip((2 * b - tasmax - tasmin) / (tasmax - tasmin), -1, 1))

    # HDD formula components
    hdd_thresh_below = 0.0
    hdd_thresh_between = (1 - t_bar / np.pi) * (b - (tasmax + tasmin) / 2) + (
        tasmax - tasmin
    ) / (2 * np.pi) * np.sin(t_bar)
    hdd_thresh_above = b - (tasmax + tasmin) / 2.0

    return np.where(
        b <= tasmin,
        hdd_thresh_below,
        np.where(b >= tasmax, hdd_thresh_above, hdd_thresh_between),
    )


# ufunc for dask
def hdd_ufunc(tasmin, tasmax, threshold=thresh_c):
    return xr.apply_ufunc(calculate_hdd, tasmin, tasmax, threshold, dask="allowed")


# Degree days
def calculate_dd_sum(ds_in, var_id):
    # Get correct temperature variable
    ds_in = transform_temperature(ds_in, var_id)

    # CDD
    if var_id == "cdd":
        ds_out = cdd_ufunc(ds_in["tasmin"], ds_in["tasmax"]).resample(time="YE").sum()
    # HDD
    elif var_id == "hdd":
        ds_out = hdd_ufunc(ds_in["tasmin"], ds_in["tasmax"]).resample(time="YE").sum()

    return xr.Dataset({var_id: ds_out})


def calculate_dd_max(ds_in, var_id):
    # Get correct temperature variable
    ds_in = transform_temperature(ds_in, var_id)

    # CDD
    if var_id == "cdd":
        ds_out = cdd_ufunc(ds_in["tasmin"], ds_in["tasmax"]).resample(time="YE").max()
    # HDD
    elif var_id == "hdd":
        ds_out = hdd_ufunc(ds_in["tasmin"], ds_in["tasmax"]).resample(time="YE").max()

    return xr.Dataset({var_id: ds_out})


###############################
# TGW specific functions
###############################
def tgw_hourly_to_daily(
    file_path, var_id_in, var_id_out, agg_func, log_path, threshold=thresh_c
):
    """
    Convert hourly TGW data to daily data using the provided aggregation function.

    Parameters:
    -----------
    file_path : str
        Path to the WRF output file
    var_id_in : str
        Input variable ID in the WRF dataset
    var_id_out : str
        Output variable ID for the resulting dataset
    agg_func : str
        Aggregation function to apply ('max', 'min', 'mean', 'sum', 'cdd', 'hdd')
    log_path : str
        Path to write error logs
    threshold : float, optional
        Temperature threshold in degrees Celsius for CDD/HDD calculations (default: 18.0)

    Returns:
    --------
    xarray.Dataset
        Daily aggregated dataset
    """
    try:
        import salem

        #  Read
        ds = salem.open_wrf_dataset(file_path)[var_id_in]

        # Correct naming
        ds = xr.Dataset({var_id_out: ds})

        # Aggregate
        if agg_func == "max":
            ds_out = ds.resample(time="D").max()
        elif agg_func == "min":
            ds_out = ds.resample(time="D").min()
        elif agg_func == "mean":
            ds_out = ds.resample(time="D").mean()
        elif agg_func == "sum":
            ds_out = ds.resample(time="D").sum()
        elif agg_func == "cdd":
            # For CDD: max(0, T - threshold) for each hour, then sum for the day
            # Only count degrees above the threshold
            cdd = ds.clip(min=threshold) - threshold
            ds_out = cdd.resample(time="D").sum()
        elif agg_func == "hdd":
            # For HDD: max(0, threshold - T) for each hour, then sum for the day
            # Only count degrees below the threshold
            hdd = threshold - ds.clip(max=threshold)
            ds_out = hdd.resample(time="D").sum()
        else:
            raise ValueError(f"Invalid aggregation function: {agg_func}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        with open(f"{log_path}/{file_path.split('/')[-1]}.log", "a") as f:
            f.write(f"Error reading {file_path}: {e}\n")
            return None

    return ds_out
