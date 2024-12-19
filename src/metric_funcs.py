import numpy as np
import xarray as xr


###########################
# Variable transformations
###########################
def transform_precipitation(ds_in, var_id):
    if var_id == "pr":
        if "pr" not in ds_in.variables and "pcp" in ds_in.variables:
            ds_in = ds_in.rename({"pcp": "pr"})

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
            # LOCA2
            elif "tasmin" in ds_in.variables and "tasmax" in ds_in.variables:
                ds_in["tas"] = (ds_in["tasmax"] + ds_in["tasmin"]) / 2
    elif var_id == "tasmin":
        if "tasmin" not in ds_in.variables:
            # STAR
            ds_in["tasmin"] = ds_in["t_mean"] - ds_in["t_range"] / 2.0
    elif var_id == "tasmax":
        if "tasmax" not in ds_in.variables:
            # STAR
            ds_in["tasmax"] = ds_in["t_mean"] + ds_in["t_range"] / 2.0
    elif var_id in ["cdd", "hdd"]:
        # GARD-LENS
        if "tasmin" not in ds_in.variables or "tasmax" not in ds_in.variables:
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
        return transform_precipitation(ds_in)
    else:
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
    t_bar = np.arccos(
        np.clip((2 * b - tasmax - tasmin) / (tasmax - tasmin), -1, 1)
    )

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
    return xr.apply_ufunc(
        calculate_cdd, tasmin, tasmax, threshold, dask="allowed"
    )


# HDD
def calculate_hdd(tasmin, tasmax, b):
    # Calculate t_bar only for the valid range where tasmin < b < tasmax
    t_bar = np.arccos(
        np.clip((2 * b - tasmax - tasmin) / (tasmax - tasmin), -1, 1)
    )

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
    return xr.apply_ufunc(
        calculate_hdd, tasmin, tasmax, threshold, dask="allowed"
    )


# Degree days
def calculate_dd_sum(ds_in, var_id):
    # Get correct temperature variable
    ds_in = transform_temperature(ds_in, var_id)

    # CDD
    if var_id == "cdd":
        ds_out = (
            cdd_ufunc(ds_in["tasmin"], ds_in["tasmax"])
            .resample(time="YE")
            .sum()
        )
    # HDD
    elif var_id == "hdd":
        ds_out = (
            hdd_ufunc(ds_in["tasmin"], ds_in["tasmax"])
            .resample(time="YE")
            .sum()
        )

    return xr.Dataset({var_id: ds_out})


def calculate_dd_max(ds_in, var_id):
    # Get correct temperature variable
    ds_in = transform_temperature(ds_in, var_id)

    # CDD
    if var_id == "cdd":
        ds_out = (
            cdd_ufunc(ds_in["tasmin"], ds_in["tasmax"])
            .resample(time="YE")
            .max()
        )
    # HDD
    elif var_id == "hdd":
        ds_out = (
            hdd_ufunc(ds_in["tasmin"], ds_in["tasmax"])
            .resample(time="YE")
            .max()
        )

    return xr.Dataset({var_id: ds_out})
