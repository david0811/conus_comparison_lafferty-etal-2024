import numpy as np
import xarray as xr

####################################
# Metric calculation functions
####################################

################
# Simple
################


def calculate_avg(ds_in, var_id):
    # Calculate tas if needed
    if var_id == "tas" and "tas" not in ds_in.variables:
        if "t_mean" in ds_in.variables:  # STAR
            ds_in = ds_in.rename({"t_mean": "tas"})
        elif "tasmin" in ds_in.variables and "tasmax" in ds_in.variables:  # LOCA
            ds_in["tas"] = (ds_in["tasmin"] + ds_in["tasmax"]) / 2.0

    # Calculate average
    return ds_in[[var_id]].resample(time="YE").mean()


def calculate_sum(ds_in, var_id):
    # Calculate sum
    ds_out = ds_in[[var_id]].resample(time="YE").sum()
    # Update units
    if var_id == "pr":
        ds_out.pr.attrs["units"] = "mm"

    return ds_out


def calculate_max(ds_in, var_id):
    # for GARD-LENS
    if var_id == "tasmax" and "tasmax" not in ds_in.variables:
        ds_in["tasmax"] = ds_in["t_mean"] + ds_in["t_range"] / 2.0
    if var_id == "pr":
        if "pr" not in ds_in.variables and "pcp" in ds_in.variables:
            ds_in = ds_in.rename({"pcp": "pr"})

        ds_in.pr.attrs["units"] = "mm"

    # Calculate max
    ds_out = ds_in[[var_id]].resample(time="YE").max()

    return ds_out


def calculate_min(ds_in, var_id):
    # for GARD-LENS
    if var_id == "tasmin" and "tasmin" not in ds_in.variables:
        ds_in["tasmin"] = ds_in["t_mean"] - ds_in["t_range"] / 2.0

    # Calculate min
    ds_out = ds_in[[var_id]].resample(time="YE").min()

    return ds_out


#########################
# Degree days
#########################
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
    cdd_thresh_between = (t_bar / np.pi) * ((tasmax + tasmin) / 2 - b) + (tasmax - tasmin) / (
        2 * np.pi
    ) * np.sin(t_bar)
    cdd_thresh_above = 0.0

    return np.where(
        b <= tasmin, cdd_thresh_below, np.where(b >= tasmax, cdd_thresh_above, cdd_thresh_between)
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
    hdd_thresh_between = (1 - t_bar / np.pi) * (b - (tasmax + tasmin) / 2) + (tasmax - tasmin) / (
        2 * np.pi
    ) * np.sin(t_bar)
    hdd_thresh_above = b - (tasmax + tasmin) / 2.0

    return np.where(b <= tasmin, hdd_thresh_below, np.where(b >= tasmax, hdd_thresh_above, hdd_thresh_between))


# ufunc for dask
def hdd_ufunc(tasmin, tasmax, threshold=thresh_c):
    return xr.apply_ufunc(calculate_hdd, tasmin, tasmax, threshold, dask="allowed")


# Degree days
def calculate_dd_sum(ds_in, var_id):
    # for GARD-LENS
    if "tasmin" not in ds_in.variables or "tasmax" not in ds_in.variables:
        ds_in["tasmin"] = ds_in["t_mean"] - ds_in["t_range"] / 2.0
        ds_in["tasmax"] = ds_in["t_mean"] + ds_in["t_range"] / 2.0

    # CDD
    if var_id == "cdd":
        ds_out = cdd_ufunc(ds_in["tasmin"], ds_in["tasmax"]).resample(time="YE").sum()
    # HDD
    elif var_id == "hdd":
        ds_out = hdd_ufunc(ds_in["tasmin"], ds_in["tasmax"]).resample(time="YE").sum()

    return xr.Dataset({var_id: ds_out})


def calculate_dd_max(ds_in, var_id):
    # for GARD-LENS
    if "tasmin" not in ds_in.variables or "tasmax" not in ds_in.variables:
        ds_in["tasmin"] = ds_in["t_mean"] - ds_in["t_range"] / 2.0
        ds_in["tasmax"] = ds_in["t_mean"] + ds_in["t_range"] / 2.0

    # CDD
    if var_id == "cdd":
        ds_out = cdd_ufunc(ds_in["tasmin"], ds_in["tasmax"]).resample(time="YE").max()
    # HDD
    elif var_id == "hdd":
        ds_out = hdd_ufunc(ds_in["tasmin"], ds_in["tasmax"]).resample(time="YE").max()

    return xr.Dataset({var_id: ds_out})
