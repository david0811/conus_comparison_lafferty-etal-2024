from functools import partial
import os
from glob import glob

import dask.array as da
import numpy as np
import xarray as xr
from numba import jit
from scipy.stats import genextreme as gev
from scipy.optimize import minimize
import sdfc_classes as sd

from lmom_utils import (
    pargev_numba,
    samlmom3_numba,
)
from gev_utils import xr_estimate_return_level, estimate_return_level

from utils import roar_data_path as project_data_path
from utils import roar_code_path as project_code_path
from utils import map_store_names
from utils import check_data_length, get_starting_year


@jit(nopython=True, cache=True, parallel=False)
def negative_log_likelihood_numba(params, data, covariate):
    shape, loc_intcp, loc_trend, log_scale_intcp, log_scale_trend = params
    loc = loc_intcp + loc_trend * covariate
    scale = np.exp(log_scale_intcp + log_scale_trend * covariate)
    # Manual implementation of GEV logpdf for Numba compatibility
    y = 1 + -shape * ((data - loc) / scale)

    return -np.sum(-np.log(scale) - (1 + 1 / -shape) * np.log(y) - y ** (-1 / -shape))


def negative_log_likelihood(params, data, covariate):
    shape, loc_intcp, loc_trend, log_scale_intcp, log_scale_trend = params
    loc = loc_intcp + loc_trend * covariate
    scale = np.exp(log_scale_intcp + log_scale_trend * covariate)
    return -gev.logpdf(data, shape, loc, scale).sum()


def get_dynamic_bounds(data, covariate):
    data_min, data_max = np.min(data), np.max(data)
    data_range = data_max - data_min

    # More appropriate bounds based on data characteristics
    shape_bounds = (-1.0, 1.0)  # Common range for climate data
    loc_intcp_bounds = (data_min / 2.0, data_max)
    loc_trend_bounds = (-data_range / 2.0, data_range / 2.0)
    log_scale_intcp_bounds = (np.log(0.01), np.log(data_range))
    log_scale_trend_bounds = (-np.log(data_range / 2.0), np.log(data_range / 2.0))

    return (
        shape_bounds,
        loc_intcp_bounds,
        loc_trend_bounds,
        log_scale_intcp_bounds,
        log_scale_trend_bounds,
    )


def nonstationary_optimizer(data, covariate, initial_params, bounds=True):
    if bounds:
        result = minimize(
            negative_log_likelihood_numba,
            initial_params,
            args=(data, covariate),
            method="Nelder-Mead",
            bounds=get_dynamic_bounds(data, covariate),
        )
    else:
        result = minimize(
            negative_log_likelihood_numba,
            initial_params,
            args=(data, covariate),
        )
    if result.success:
        shape, loc_intcp, loc_trend, log_scale_intcp, log_scale_trend = result.x
        return (loc_intcp, loc_trend, log_scale_intcp, log_scale_trend, shape)
    else:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)


def _fit_gev_1d_nonstationary(
    data, expected_length, fit_method="mle", initial_params=None
):
    """
    Fit non-stationary GEV to 1-dimensional data with both location and scale trends.
    Note we follow scipy convention of negating the shape parameter relative to other sources.

    Returns:
    --------
    tuple: (loc_intercept, loc_trend, scale_intercept, scale_trend, shape)
    """

    # Return NaN if all Nans or zeros
    if np.isnan(data).all() or (data == 0.0).all():
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    # Check length
    assert len(data) == expected_length, (
        f"data length is {len(data)}, expected {expected_length}"
    )

    # Fit
    result = False
    if fit_method == "mle":
        # Initial params from L-moments
        if initial_params is None:
            loc, scale, shape = pargev_numba(samlmom3_numba(data))
            log_scale = np.log(scale)
            initial_params = [shape, loc, 0.0, log_scale, 0.0]  # 5 parameters now
        # Fit
        try:
            result = nonstationary_optimizer(
                data=data,
                covariate=np.arange(len(data)),
                initial_params=initial_params,
            )
            if np.isnan(result).any():
                result = False
            else:
                return result
        except Exception:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)

    # SDFC fallback does not work for scale trends
    if not result:
        law_ns = sd.GEV()
        for i in range(100):
            law_ns.fit(
                data,
                c_loc=np.arange(len(data)),
                c_scale=np.arange(len(data)),
                l_scale=sd.ULExponential(),
            )
            # if the first coefficient is not zero, we stop fitting
            if law_ns.coef_[0] != 0:
                loc_intcp, loc_trend, log_scale_intcp, log_scale_trend, shape = tuple(
                    law_ns.coef_
                )
                return (loc_intcp, loc_trend, log_scale_intcp, log_scale_trend, -shape)
        return (np.nan, np.nan, np.nan, np.nan, np.nan)


def _gev_fit_parametric_bootstrap_1d_nonstationary(
    params_in,
    expected_length,
    n_boot,
    fit_method,
    return_samples=True,
):
    """
    Generate parametric bootstrap samples for GEV with both location and scale trends.
    """
    loc_intcp = params_in[0]
    loc_trend = params_in[1]
    log_scale_intcp = params_in[2]
    log_scale_trend = params_in[3]
    shape = params_in[4]
    initial_params = [shape, loc_intcp, loc_trend, log_scale_intcp, log_scale_trend]

    # Return NaN if all Nans or zeros
    if np.isnan(params_in).any():
        if return_samples:
            return np.nan * np.ones((n_boot, 5))
        else:
            return np.nan * np.ones((2, 5))

    params_out = np.zeros((n_boot, 5))

    # Bootstrap sampling - scale must be positive at all time points
    covariate = np.arange(expected_length)
    scale_values = np.exp(log_scale_intcp + log_scale_trend * covariate)

    # Check if scale becomes non-positive
    if np.any(scale_values <= 0):
        # Return NaN if scale becomes non-positive
        if return_samples:
            return np.nan * np.ones((n_boot, 5))
        else:
            return np.nan * np.ones((2, 5))

    boot_sample = gev.rvs(
        shape,
        loc=loc_intcp + loc_trend * covariate,
        scale=scale_values,
        size=(n_boot, expected_length),
    )

    for i in range(n_boot):
        # Do the fit
        params_out[i, :] = _fit_gev_1d_nonstationary(
            boot_sample[i],
            expected_length,
            fit_method=fit_method,
            initial_params=initial_params,
        )

    # Return samples or 95% intervals
    if return_samples:
        return params_out
    else:
        return np.nanpercentile(params_out, [2.5, 97.5], axis=0)


def _gev_parametric_bootstrap_1d_nonstationary(
    params,
    expected_length,
    starting_year,
    n_data,
    n_boot,
    fit_method,
    periods_for_level,
    return_period_years,
    return_period_diffs,
    return_samples=False,
):
    """
    Generate parametric bootstrap samples for GEV with both location and scale trends.
    """
    loc_intcp, loc_trend, log_scale_intcp, log_scale_trend, shape = params
    initial_params = [shape, loc_intcp, loc_trend, log_scale_intcp, log_scale_trend]

    params_out = np.zeros((n_boot, 5))
    return_levels_out = np.zeros(
        (n_boot, len(periods_for_level) * len(return_period_years))
    )
    return_level_diffs_out = np.zeros(
        (n_boot, len(periods_for_level) * len(return_period_diffs))
    )
    return_level_chfcs_out = np.zeros(
        (n_boot, len(periods_for_level) * len(return_period_diffs))
    )

    # Check if scale becomes non-positive at any time point
    covariate = np.arange(n_data)
    scale_values = np.exp(log_scale_intcp + log_scale_trend * covariate)

    if np.any(scale_values <= 0):
        # Return NaN arrays if scale becomes non-positive
        params_out[:] = np.nan
        return_levels_out[:] = np.nan
        return_level_diffs_out[:] = np.nan
        return_level_chfcs_out[:] = np.nan
    else:
        # Bootstrap sampling
        boot_sample = gev.rvs(
            shape,
            loc=loc_intcp + loc_trend * covariate,
            scale=scale_values,
            size=(n_boot, n_data),
        )

        for i in range(n_boot):
            # Do the fit
            params_out[i, :] = _fit_gev_1d_nonstationary(
                boot_sample[i],
                expected_length,
                fit_method=fit_method,
                initial_params=initial_params,
            )

            # Return levels
            (
                loc_intcp_tmp,
                loc_trend_tmp,
                log_scale_intcp_tmp,
                log_scale_trend_tmp,
                shape_tmp,
            ) = params_out[i, :]
            return_levels_out[i, :] = [
                estimate_return_level(
                    period,
                    loc_intcp_tmp
                    + loc_trend_tmp * (return_period_year - starting_year),
                    np.exp(
                        log_scale_intcp_tmp
                        + log_scale_trend_tmp * (return_period_year - starting_year)
                    ),
                    shape_tmp,
                )
                for period in periods_for_level
                for return_period_year in return_period_years
            ]

            # Return level differences
            return_level_diffs_out[i, :] = [
                estimate_return_level(
                    period,
                    loc_intcp_tmp
                    + loc_trend_tmp * (return_period_diff[1] - starting_year),
                    np.exp(
                        log_scale_intcp_tmp
                        + log_scale_trend_tmp * (return_period_diff[1] - starting_year)
                    ),
                    shape_tmp,
                )
                - estimate_return_level(
                    period,
                    loc_intcp_tmp
                    + loc_trend_tmp * (return_period_diff[0] - starting_year),
                    np.exp(
                        log_scale_intcp_tmp
                        + log_scale_trend_tmp * (return_period_diff[0] - starting_year)
                    ),
                    shape_tmp,
                )
                for period in periods_for_level
                for return_period_diff in return_period_diffs
            ]

            # Return level change factors
            return_level_chfcs_out[i, :] = [
                estimate_return_level(
                    period,
                    loc_intcp_tmp
                    + loc_trend_tmp * (return_period_diff[1] - starting_year),
                    np.exp(
                        log_scale_intcp_tmp
                        + log_scale_trend_tmp * (return_period_diff[1] - starting_year)
                    ),
                    shape_tmp,
                )
                / estimate_return_level(
                    period,
                    loc_intcp_tmp
                    + loc_trend_tmp * (return_period_diff[0] - starting_year),
                    np.exp(
                        log_scale_intcp_tmp
                        + log_scale_trend_tmp * (return_period_diff[0] - starting_year)
                    ),
                    shape_tmp,
                )
                for period in periods_for_level
                for return_period_diff in return_period_diffs
            ]

    # Return samples or 95% intervals
    if return_samples:
        return (
            params_out,
            return_levels_out,
            return_level_diffs_out,
            return_level_chfcs_out,
        )
    else:
        return (
            np.nanpercentile(params_out, [2.5, 97.5], axis=0),
            np.nanpercentile(return_levels_out, [2.5, 97.5], axis=0),
            np.nanpercentile(return_level_diffs_out, [2.5, 97.5], axis=0),
            np.nanpercentile(return_level_chfcs_out, [2.5, 97.5], axis=0),
        )


def fit_ns_gev_xr(
    ds,
    metric_id,
    expected_length,
    starting_year,
    fit_method="mle",
    periods_for_level=None,
    return_period_years=None,
):
    """
    Fit GEV to xarray data. Note we follow scipy convention
    of negating the shape parameter relative to other sources.
    """
    var_id = metric_id.split("_")[1]
    agg_id = metric_id.split("_")[0]

    # Changes for minima
    if agg_id == "min":
        scalar = -1.0
    else:
        scalar = 1.0

    # Prepare dask data (chunking over space)
    dask_data = da.from_array(
        ds[var_id].to_numpy() * scalar, chunks=(expected_length, 50, 50)
    )

    # Prepare fit function
    fit_gev_1d = partial(
        _fit_gev_1d_nonstationary,
        expected_length=expected_length,
        fit_method=fit_method,
    )

    # Do the fit
    params = da.map_blocks(
        lambda block: np.apply_along_axis(fit_gev_1d, axis=0, arr=block),
        dask_data,
        dtype=np.float64,
        drop_axis=0,
        new_axis=[0],
    ).compute()

    # Gather results
    lat_name = "latitude" if "latitude" in ds.dims else "lat"
    lon_name = "longitude" if "longitude" in ds.dims else "lon"
    ds_out = xr.Dataset(
        data_vars={
            "loc_intcp": ([lat_name, lon_name], params[0, :, :]),
            "loc_trend": ([lat_name, lon_name], params[1, :, :]),
            "log_scale_intcp": ([lat_name, lon_name], params[2, :, :]),
            "log_scale_trend": ([lat_name, lon_name], params[3, :, :]),
            "shape": ([lat_name, lon_name], params[4, :, :]),
        },
        coords={
            lat_name: ([lat_name], ds[lat_name].to_numpy()),
            lon_name: ([lon_name], ds[lon_name].to_numpy()),
        },
    )

    # Return level calculations
    for return_period in periods_for_level:
        return_levels = []
        for return_period_year in return_period_years:
            return_level = xr_estimate_return_level(
                return_period,
                ds_out,
                scalar,
                loc_stationary=False,
                scale_stationary=False,
                return_period_year=return_period_year,
                starting_year=starting_year,
                return_params=False,
            )
            return_levels.append(
                return_level.assign_coords({"time": return_period_year})
            )
        return_levels = xr.concat(return_levels, dim="time")
        ds_out = xr.merge([ds_out, return_levels])

    return ds_out


def fit_ns_gev_xr_bootstrap(
    params_in,
    metric_id,
    expected_length,
    starting_year,
    fit_method="mle",
    periods_for_level=[10, 25, 50, 100],
    return_period_years=[1950, 1975, 2000, 2025, 2050, 2075, 2100],
    return_period_diffs=[(1975, 2075)],
    n_boot=100,
    return_samples=False,
):
    """
    Fit GEV to xarray data. Note we follow scipy convention
    of negating the shape parameter relative to other sources.
    """
    var_id = metric_id.split("_")[1]
    agg_id = metric_id.split("_")[0]

    # Changes for minima
    if agg_id == "min":
        scalar = -1.0
    else:
        scalar = 1.0

    # For testing
    # params_in = params_in.isel(lat=slice(200, 210), lon=slice(400, 410))

    # Prepare dask data (chunking over space)
    dask_data = da.from_array(
        np.squeeze(
            params_in[
                [
                    "loc_intcp",
                    "loc_trend",
                    "log_scale_intcp",
                    "log_scale_trend",
                    "shape",
                ]
            ]
            .to_array()
            .to_numpy()
        ),
        chunks=(4, 10, 10),
    )

    # Prepare fit function
    fit_gev_1d = partial(
        _gev_fit_parametric_bootstrap_1d_nonstationary,
        expected_length=expected_length,
        n_boot=n_boot,
        fit_method=fit_method,
    )

    # Do the fit
    out = da.map_blocks(
        lambda block: np.apply_along_axis(fit_gev_1d, axis=0, arr=block),
        dask_data,
        dtype=np.float64,
        drop_axis=0,
        new_axis=[0, 1],
    ).compute()

    # Gather results
    lat_name = "latitude" if "latitude" in params_in.dims else "lat"
    lon_name = "longitude" if "longitude" in params_in.dims else "lon"
    ds_out = xr.Dataset(
        data_vars={
            "loc_intcp": (["n_boot", lat_name, lon_name], out[:, 0, :, :]),
            "loc_trend": (["n_boot", lat_name, lon_name], out[:, 1, :, :]),
            "log_scale_intcp": (["n_boot", lat_name, lon_name], out[:, 2, :, :]),
            "log_scale_trend": (["n_boot", lat_name, lon_name], out[:, 3, :, :]),
            "shape": (["n_boot", lat_name, lon_name], out[:, 4, :, :]),
        },
        coords={
            "n_boot": (["n_boot"], np.arange(n_boot)),
            lat_name: ([lat_name], params_in[lat_name].to_numpy()),
            lon_name: ([lon_name], params_in[lon_name].to_numpy()),
        },
    )

    # Return level calculations
    for return_period in periods_for_level:
        return_levels = []
        for return_period_year in return_period_years:
            return_level = xr_estimate_return_level(
                return_period,
                ds_out,
                scalar,
                loc_stationary=False,
                scale_stationary=False,
                return_period_year=return_period_year,
                starting_year=starting_year,
                return_params=False,
            )
            return_levels.append(
                return_level.assign_coords({"time": return_period_year})
            )
        return_levels = xr.concat(return_levels, dim="time")
        ds_out = xr.merge([ds_out, return_levels])

    # Calculate return level diffs/chfcs
    for return_period_diff in return_period_diffs:
        # Differences
        rl_diff = ds_out.sel(time=return_period_diff[1]) - ds_out.sel(
            time=return_period_diff[0]
        )
        rl_diff = rl_diff[
            [f"{return_period}yr_return_level" for return_period in periods_for_level]
        ]
        rl_diff = rl_diff.rename(
            {
                f"{return_period}yr_return_level": f"{return_period}yr_return_level_diff"
                for return_period in periods_for_level
            }
        )
        rl_diff = rl_diff.expand_dims(
            {"time_diff": [f"{return_period_diff[1]}-{return_period_diff[0]}"]}
        )
        # Change factors
        rl_chfcs = ds_out.sel(time=return_period_diff[1]) / ds_out.sel(
            time=return_period_diff[0]
        )
        rl_chfcs = rl_chfcs[
            [f"{return_period}yr_return_level" for return_period in periods_for_level]
        ]
        rl_chfcs = rl_chfcs.rename(
            {
                f"{return_period}yr_return_level": f"{return_period}yr_return_level_chfc"
                for return_period in periods_for_level
            }
        )
        rl_chfcs = rl_chfcs.expand_dims(
            {"time_diff": [f"{return_period_diff[1]}-{return_period_diff[0]}"]}
        )
        # Append
        ds_out = xr.merge([ds_out, rl_diff, rl_chfcs])

    # Return
    if return_samples:
        return ds_out
    else:
        ds_out = ds_out.quantile([0.025, 0.975], dim="n_boot")
        ds_out["quantile"] = ["q025", "q975"]
        return ds_out


def fit_ns_gev_single(
    ensemble,
    gcm,
    member,
    ssp,
    metric_id,
    fit_method="mle",
    years=[1950, 2100],
    bootstrap=False,
    n_boot=100,
    periods_for_level=[10, 25, 50, 100],
    return_period_years=[1950, 1975, 2000, 2025, 2050, 2075, 2100],
    return_period_diffs=[(1975, 2075)],
    project_data_path=project_data_path,
    project_code_path=project_code_path,
):
    """
    Read a single metric file and fit the GEV.
    """
    try:
        # Check if done
        time_name = f"{years[0]}-{years[1]}"
        boot_name = f"nboot{n_boot}" if bootstrap else "main"
        store_name = f"{ensemble}_{gcm}_{member}_{ssp}_{time_name}_nonstat_scale_{fit_method}_{boot_name}.nc"
        store_path = f"{project_data_path}/extreme_value/original_grid/{metric_id}"

        if os.path.exists(f"{store_path}/{store_name}"):
            print(f"Skipping {store_name} because it already exists")
            return None

        ## Read data file
        if not bootstrap:
            # LOCA2
            if ensemble == "LOCA2":
                # Projection
                proj_files = glob(
                    f"{project_data_path}/metrics/LOCA2/{metric_id}_{gcm}_{member}_{ssp}_*.nc"
                )
                ds = xr.concat(
                    [xr.open_dataset(file) for file in proj_files], dim="time"
                )
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

            # Check length is as expected
            ds["time"] = ds["time"].dt.year
            ds = ds.sel(time=slice(years[0], years[1]))
            expected_length = check_data_length(ds["time"], ensemble, gcm, ssp, years)
            starting_year = int(ds["time"].min())

        else:
            # Read parameter file
            ds = xr.open_dataset(
                f"{store_path}/{ensemble}_{gcm}_{member}_{ssp}_{time_name}_nonstat_scale_{fit_method}_main.nc"
            )
            expected_length = check_data_length(None, ensemble, gcm, ssp, years)
            starting_year = get_starting_year(ensemble, gcm, ssp, years)

        # Fit GEV
        if bootstrap:
            ds_out = fit_ns_gev_xr_bootstrap(
                params_in=ds,
                metric_id=metric_id,
                fit_method=fit_method,
                expected_length=expected_length,
                starting_year=starting_year,
                n_boot=n_boot,
                periods_for_level=periods_for_level,
                return_period_years=return_period_years,
                return_period_diffs=return_period_diffs,
            )
        else:
            ds_out = fit_ns_gev_xr(
                ds=ds,
                metric_id=metric_id,
                fit_method=fit_method,
                expected_length=expected_length,
                starting_year=starting_year,
                periods_for_level=periods_for_level,
                return_period_years=return_period_years,
            )
        ## Store
        gcm_name, member_name = map_store_names(ensemble, gcm, member)
        ds_out = ds_out.expand_dims(
            {
                "gcm": [gcm_name],
                "member": [member_name],
                "ssp": [ssp],
                "ensemble": [ensemble],
            }
        )
        if not bootstrap:
            ds_out = ds_out.assign_coords({"quantile": ["main"]})

        # Make sure not all NaNs
        assert np.count_nonzero(~np.isnan(ds_out["shape"])), "all NaNs in fit"
        ds_out.to_netcdf(f"{store_path}/{store_name}")

    # Log if error
    except Exception as e:
        except_path = f"{project_code_path}/scripts/logs/gev_freq/"
        with open(
            f"{except_path}/{ensemble}_{gcm}_{member}_{ssp}_{metric_id}_nonstat_{fit_method}_{boot_name}.txt",
            "w",
        ) as f:
            f.write(str(e))
