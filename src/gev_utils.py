import numpy as np
import xarray as xr
from scipy.stats import genextreme as gev


##############################################
# Return level utils
# https://github.com/jdossgollin/2021-TXtreme
##############################################
def estimate_return_level(return_period, loc, scale, shape):
    """
    Calculate the return level given GEV parameters.
    Works for both scalars and arrays.
    """
    quantile = 1 - 1 / return_period
    if np.isscalar(shape) and np.isclose(shape, 0):
        level = loc - scale * np.log(-np.log(quantile))
    else:
        level = np.where(
            np.isclose(shape, 0),
            loc - scale * np.log(-np.log(quantile)),
            loc + scale / shape * (1 - (-np.log(quantile)) ** shape),
        )
    return level


def xr_estimate_return_level(
    return_period,
    ds,
    scalar,
    stationary=True,
    return_period_year=None,
    starting_year=None,
    return_params=False,
):
    """
    Calculate the return level given GEV parameters.
    """
    # Get location parameters
    if not stationary:
        locs = ds["loc_intcp"] + ds["loc_trend"] * (return_period_year - starting_year)
    else:
        locs = ds["loc"]

    # Calculate return level
    dims = ds["shape"].dims
    return_level = xr.apply_ufunc(
        estimate_return_level,
        return_period,
        locs,
        ds["scale"],
        ds["shape"],
        input_core_dims=[[], dims, dims, dims],
        output_core_dims=[dims],
        vectorize=True,
        dask="allowed",
    )

    # Scale the return level
    return_level = scalar * return_level

    ds_out = xr.Dataset({f"{int(return_period)}yr_return_level": return_level})

    if not return_params:
        return ds_out
    else:
        return xr.merge([ds, ds_out])


def estimate_return_period(threshold, loc, scale, shape):
    """
    Calculate the return period of a threshold given GEV parameters.
    """
    y_cdf = gev.cdf(threshold, loc=loc, scale=scale, c=shape)
    period = 1 / (1 - y_cdf)

    return period


def xr_estimate_return_period(threshold, loc, scale, shape):
    """
    Manually calculate the return period of a threshold given GEV parameters
    using xarray.

    NOTE: this follows scipy convention of negating the shape parameter.
    """
    cdf = xr.where(
        shape != 0,
        np.exp(-np.power(1 + -shape * (threshold - loc) / scale, -1 / -shape)),
        np.exp(-np.exp(-(threshold - loc) / scale)),
    )

    period = 1 / (1 - cdf)

    period = period.where(period < 1000, 1000)  # cap at 1000 years
    period = period.where(~loc.isnull(), np.nan)  # filter nans
    return period
