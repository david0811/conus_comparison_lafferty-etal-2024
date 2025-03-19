import os
from functools import partial
from glob import glob

import dask
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import genextreme as gev

import sdfc_classes as sd
from lmom_utils import (
    pargev_bootstrap_numba,
    pargev_numba,
    pargev_numpy,
    samlmom3_bootstrap_numba,
    samlmom3_numba,
    samlmom3_numpy,
)
from utils import check_data_length, get_unique_loca_metrics, map_store_names
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path


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


def xr_estimate_return_level(return_period, ds, scalar, return_params=False):
    """
    Calculate the return level given GEV parameters.
    """
    return_level = xr.apply_ufunc(
        estimate_return_level,
        return_period,
        ds["loc"],
        ds["scale"],
        ds["shape"],
        input_core_dims=[[], ds["loc"].dims, ds["loc"].dims, ds["loc"].dims],
        output_core_dims=[ds["loc"].dims],
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


##############################################
# GEV fitting with xarray
##############################################
def optimizer(func, x0, args, disp):
    """Define the optimization method to use when fitting the GEV"""
    res = minimize(func, x0, args, method="Nelder-Mead")
    return res.x


def _fit_gev_1d_stationary(
    data, expected_length=None, fit_method="lmom", optimizer=optimizer, numba=True
):
    """
    Fit GEV to 1-dimensional data. Note we follow scipy convention
    of negating the shape parameter relative to other sources.
    """
    # Return NaN if all Nans
    if np.isnan(data).all():
        return (np.nan, np.nan, np.nan)

    # Check length of non-NaNs
    if expected_length is not None:
        non_nans = np.count_nonzero(~np.isnan(data))
        assert non_nans == expected_length, (
            f"data length is {non_nans}, expected {expected_length}"
        )

    # Some GARD-LENS outputs have all values zero
    if (data == 0.0).all():
        return (np.nan, np.nan, np.nan)

    # Fit
    if numba:
        samlmom3 = samlmom3_numba
        pargev = pargev_numba
    else:
        samlmom3 = samlmom3_numpy
        pargev = pargev_numpy
    if fit_method == "lmom":
        loc, scale, shape = pargev(samlmom3(data))
        return (loc, scale, shape)
    elif fit_method == "mle":
        shape, location, scale = gev.fit(data, optimizer=optimizer)
        return (location, scale, shape)


def _gev_parametric_bootstrap_1d_stationary(
    loc,
    scale,
    shape,
    n_data,
    n_boot,
    fit_method,
    periods_for_level,
    return_samples=False,
    numba=True,
):
    """
    Generate parametric bootstrap samples for GEV.
    """
    params_out = np.full((n_boot, 3), np.nan)
    return_levels_out = np.full((n_boot, len(periods_for_level)), np.nan)

    # Bootstrap sampling
    if not np.isnan([loc, scale, shape]).any():
        boot_samples = gev.rvs(shape, loc=loc, scale=scale, size=(n_boot, n_data))
        for i in range(n_boot):
            # Do the fit
            params_out[i, :] = _fit_gev_1d_stationary(
                boot_samples[i], n_data, fit_method=fit_method, numba=numba
            )
            # Return levels
            return_levels_out[i, :] = estimate_return_level(
                np.array(periods_for_level), *params_out[i]
            )

    # Return 95% intervals
    if return_samples:
        return params_out, return_levels_out
    else:
        return (
            np.nanpercentile(params_out, [2.5, 97.5], axis=0),
            np.nanpercentile(return_levels_out, [2.5, 97.5], axis=0),
        )


def negative_log_likelihood(params, data, covariate):
    shape, loc_intcp, loc_trend, scale = params
    loc = loc_intcp + loc_trend * covariate
    return -gev.logpdf(data, shape, loc, scale).sum()


def nonstationary_optimizer(data, covariate, initial_params):
    result = minimize(
        negative_log_likelihood,
        initial_params,
        args=(data, covariate),
        method="Nelder-Mead",
        bounds=((-1, 1), (0, 500), (-10, 10), (0, 100)),
    )
    if result.success:
        shape, loc_intcp, loc_trend, scale = result.x
        return (loc_intcp, loc_trend, scale, shape)
    else:
        return (np.nan, np.nan, np.nan, np.nan)


def _fit_gev_1d_nonstationary(data, years, fit_method="mle"):
    """
    Fit non-stationary GEV to 1-dimensional data. Note we follow scipy convention
    of negating the shape parameter relative to other sources.
    """

    # Return NaN if all Nans
    if np.isnan(data).all():
        return (np.nan, np.nan, np.nan, np.nan)

    # Check length
    expected_length = years[1] - years[0] + 1
    assert len(data) == expected_length, (
        f"data length is {len(data)}, expected {expected_length}"
    )

    # Fit
    if fit_method == "mle":
        # Initial params from L-moments
        loc, scale, shape = pargev_numba(samlmom3_numba(data))
        initial_params = [shape, loc, 0.0, scale]
        # Fit
        return nonstationary_optimizer(
            data=data,
            covariate=np.arange(len(data)),
            initial_params=initial_params,
        )
    elif fit_method == "sdfc":
        law_ns = sd.GEV()
        for i in range(100):
            law_ns.fit(data, c_loc=np.arange(len(data)))
            # if the first coefficient is not zero, we stop fitting
            if law_ns.coef_[0] != 0:
                loc_intcp, loc_trend, scale, shape = tuple(law_ns.coef_)
                return (loc_intcp, loc_trend, scale, -shape)
        return (np.nan, np.nan, np.nan, np.nan)


def _gev_parametric_bootstrap_1d_nonstationary(
    params,
    years,
    n_data,
    n_boot,
    fit_method,
    periods_for_level,
    return_period_years,
    return_period_diffs,
    return_samples=False,
):
    """
    Generate parametric bootstrap samples for GEV.
    """
    loc_intcp, loc_trend, scale, shape = params

    params_out = np.zeros((n_boot, 4))
    return_levels_out = np.zeros(
        (n_boot, len(periods_for_level) * len(return_period_years))
    )
    return_level_diffs_out = np.zeros(
        (n_boot, len(periods_for_level) * len(return_period_diffs))
    )
    return_level_chfcs_out = np.zeros(
        (n_boot, len(periods_for_level) * len(return_period_diffs))
    )

    # Bootstrap sampling
    boot_sample = gev.rvs(
        shape,
        loc=loc_intcp + loc_trend * np.arange(n_data),
        scale=scale,
        size=(n_boot, n_data),
    )
    for i in range(n_boot):
        # Do the fit
        params_out[i, :] = _fit_gev_1d_nonstationary(
            boot_sample[i], years, fit_method=fit_method
        )
        # Return levels
        loc_intcp_tmp, loc_trend_tmp, scale_tmp, shape_tmp = params_out[i, :]
        return_levels_out[i, :] = [
            estimate_return_level(
                period,
                loc_intcp_tmp + loc_trend_tmp * (return_period_year - years[0]),
                scale_tmp,
                shape_tmp,
            )
            for period in periods_for_level
            for return_period_year in return_period_years
        ]
        # Return level differences
        return_level_diffs_out[i, :] = [
            estimate_return_level(
                period,
                loc_intcp_tmp + loc_trend_tmp * (return_period_diff[1] - years[0]),
                scale_tmp,
                shape_tmp,
            )
            - estimate_return_level(
                period,
                loc_intcp_tmp + loc_trend_tmp * (return_period_diff[0] - years[0]),
                scale_tmp,
                shape_tmp,
            )
            for period in periods_for_level
            for return_period_diff in return_period_diffs
        ]
        # Return level change factors
        return_level_chfcs_out[i, :] = [
            estimate_return_level(
                period,
                loc_intcp_tmp + loc_trend_tmp * (return_period_diff[1] - years[0]),
                scale_tmp,
                shape_tmp,
            )
            / estimate_return_level(
                period,
                loc_intcp_tmp + loc_trend_tmp * (return_period_diff[0] - years[0]),
                scale_tmp,
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


def fit_gev_xr(
    ds,
    metric_id,
    stationary,
    years,
    expected_length,
    fit_method,
    periods_for_level=None,
    levels_for_period=None,
    numba=True,
):
    """
    Fit GEV to xarray data. Note we follow scipy convention
    of negating the shape parameter relative to other sources.
    """
    var_id = metric_id.split("_")[1]
    agg_id = metric_id.split("_")[0]
    unit = ds[var_id].attrs["units"] if "units" in ds[var_id].attrs else ""

    # Changes for minima
    if agg_id == "min":
        scalar = -1.0
    else:
        scalar = 1.0

    # Do the fit
    if stationary:
        fit_gev_1d = partial(
            _fit_gev_1d_stationary,
            fit_method=fit_method,
            expected_length=expected_length,
            numba=numba,
        )
        output_dtypes = [float] * 3
        output_core_dims = [[], [], []]
    else:
        fit_gev_1d = partial(
            _fit_gev_1d_nonstationary,
            years=years,
            fit_method=fit_method,
        )
        output_dtypes = [float] * 4
        output_core_dims = [[], [], [], []]

    fit_params = xr.apply_ufunc(
        fit_gev_1d,
        scalar * ds,
        input_core_dims=[["time"]],
        output_core_dims=output_core_dims,
        vectorize=True,
        dask="forbidden",
        output_dtypes=output_dtypes,
    )

    # Create a dataset with the output parameters
    if stationary:
        ds_out = xr.merge(
            [
                fit_params[0][var_id].rename("loc"),
                fit_params[1][var_id].rename("scale"),
                fit_params[2][var_id].rename("shape"),
            ]
        )
    else:
        ds_out = xr.merge(
            [
                fit_params[0][var_id].rename("loc_intcp"),
                fit_params[1][var_id].rename("loc_trend"),
                fit_params[2][var_id].rename("scale"),
                fit_params[3][var_id].rename("shape"),
            ]
        )

    # Return level calculations (for set periods)
    if periods_for_level is not None:
        for period in periods_for_level:
            ds_out = xr_estimate_return_level(
                period, ds_out, scalar, return_params=True
            )

    # Return period calculations (for set levels)
    if levels_for_period is not None:
        for level in levels_for_period:
            ds_out[f"{level}{unit}_return_period"] = xr_estimate_return_period(
                scalar * level, ds_out["loc"], ds_out["scale"], ds_out["shape"]
            )

    return ds_out


def fit_gev_xr_bootstrap(
    ensemble,
    gcm,
    member,
    ssp,
    years,
    fit_method,
    store_path,
    return_samples=True,
    n_boot=100,
    stationary=True,
    periods_for_level=None,
    levels_for_period=None,
):
    """
    Fit GEV to xarray data. Note we follow scipy convention
    of negating the shape parameter relative to other sources.
    """
    # Read main fit results
    if years == [1950, 2014]:
        ssp_name = "historical"
    else:
        ssp_name = ssp
    time_name = f"{years[0]}-{years[1]}" if years is not None else "all"
    stat_name = "stat" if stationary else "nonstat"
    store_name = f"{ensemble}_{gcm}_{member}_{ssp_name}_{time_name}_{stat_name}_{fit_method}_main.nc"

    assert os.path.exists(f"{store_path}/{store_name}"), (
        f"main fit {store_path}/{store_name} not found"
    )
    ds_fit_main = xr.open_dataset(f"{store_path}/{store_name}")

    # Generate bootstrap sample (careful memory requirements!)
    # Get dimensions for latitude and longitude, accounting for different naming conventions
    lat_name = "latitude" if "latitude" in ds_fit_main.dims else "lat"
    lon_name = "longitude" if "longitude" in ds_fit_main.dims else "lon"
    n_lat = len(ds_fit_main[lat_name])
    n_lon = len(ds_fit_main[lon_name])
    n_time = years[1] - years[0] + 1

    shape = np.nan_to_num(ds_fit_main["shape"].to_numpy().squeeze())
    loc = np.nan_to_num(ds_fit_main["loc"].to_numpy().squeeze())
    scale = np.nan_to_num(ds_fit_main["scale"].to_numpy().squeeze())

    # Slightly faster to generate per bootstrap iteration?
    boot_samples = np.stack(
        [
            gev.rvs(shape, loc=loc, scale=scale, size=(n_time, n_lat, n_lon))
            for _ in range(n_boot)
        ]
    )
    boot_samples[boot_samples == 0] = np.nan

    # Calculate GEV parameters for bootstrap replicates
    lmoments = samlmom3_bootstrap_numba(boot_samples, bootstrap_dim=0)
    gev_params = pargev_bootstrap_numba(lmoments)

    ds = xr.Dataset(
        data_vars=dict(
            loc=(["n_boot", lat_name, lon_name], gev_params[:, 0, :, :]),
            scale=(["n_boot", lat_name, lon_name], gev_params[:, 1, :, :]),
            shape=(["n_boot", lat_name, lon_name], gev_params[:, 2, :, :]),
        ),
        coords={
            "n_boot": np.arange(n_boot),
            lat_name: ds_fit_main[lat_name],
            lon_name: ds_fit_main[lon_name],
        },
    )

    # Return level calculations (for set periods)
    if periods_for_level is not None:
        for period in periods_for_level:
            ds = xr_estimate_return_level(period, ds, 1.0, return_params=True)

    # Add coordinate labels
    if return_samples:
        return ds
    else:
        return ds.quantile([0.025, 0.975], dim="n_boot")


def gev_fit_single(
    ensemble,
    gcm,
    member,
    ssp,
    metric_id,
    years,
    stationary,
    fit_method,
    periods_for_level,
    levels_for_period,
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
        stat_name = "stat" if stationary else "nonstat"
        store_name = f"{ensemble}_{gcm}_{member}_{ssp_name}_{time_name}_{stat_name}_{fit_method}_main.nc"
        store_path = f"{project_data_path}/extreme_value/original_grid/{metric_id}"

        if os.path.exists(f"{store_path}/{store_name}"):
            return None

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

        # Apply time slice if needed
        ds["time"] = ds["time"].dt.year
        if years is not None:
            ds = ds.sel(time=slice(years[0], years[1]))

        # Check length is as expected
        expected_length = check_data_length(ds["time"], ensemble, gcm, ssp_name, years)

        # Fit GEV
        ds_out = fit_gev_xr(
            ds=ds,
            metric_id=metric_id,
            stationary=stationary,
            fit_method=fit_method,
            years=years,
            expected_length=expected_length,
            periods_for_level=periods_for_level,
            levels_for_period=levels_for_period,
        )
        ## Store
        gcm_name, member_name = map_store_names(ensemble, gcm, member)
        ds_out = ds_out.expand_dims(
            {
                "gcm": [gcm_name],
                "member": [member_name],
                "ssp": [ssp_name],
                "ensemble": [ensemble],
                "quantile": ["main"],
            }
        )
        # Make sure not all NaNs
        assert np.count_nonzero(~np.isnan(ds_out["loc"])), "all NaNs in fit"
        ds_out.to_netcdf(f"{store_path}/{store_name}")

    # Log if error
    except Exception as e:
        except_path = f"{project_code_path}/scripts/logs/gev_freq/"
        with open(
            f"{except_path}/{ensemble}_{gcm}_{member}_{ssp}_{metric_id}_{stat_name}_main.txt",
            "w",
        ) as f:
            f.write(str(e))


def gev_fit_single_bootstrap(
    ensemble,
    gcm,
    member,
    ssp,
    metric_id,
    proj_slice,
    hist_slice,
    periods_for_level,
    stationary=True,
    fit_method="lmom",
    n_boot=100,
    levels_for_period=None,
    project_data_path=project_data_path,
    project_code_path=project_code_path,
    years=None,  # dummy variable
):
    """
    Read a single metric file and fit the GEV.
    """
    try:
        # Check if done
        time_name = f"{proj_slice[0]}-{proj_slice[1]}_{hist_slice[0]}-{hist_slice[1]}"
        stat_name = "stat" if stationary else "nonstat"
        store_name = f"{ensemble}_{gcm}_{member}_{ssp}_{time_name}_{stat_name}_{fit_method}_bootstrap.nc"
        store_path = f"{project_data_path}/extreme_value/original_grid/{metric_id}"

        if os.path.exists(f"{store_path}/{store_name}"):
            return None

        # Fit GEV: proj
        ds_proj = fit_gev_xr_bootstrap(
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            years=proj_slice,
            fit_method=fit_method,
            n_boot=n_boot,
            store_path=store_path,
            periods_for_level=periods_for_level,
            levels_for_period=levels_for_period,
        )
        # Fit GEV: hist
        ds_hist = fit_gev_xr_bootstrap(
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            years=hist_slice,
            fit_method=fit_method,
            n_boot=n_boot,
            store_path=store_path,
            periods_for_level=periods_for_level,
            levels_for_period=levels_for_period,
        )

        # Take differences
        ds_diff = ds_proj - ds_hist

        # Take quantiles
        ds_hist = ds_hist.quantile([0.025, 0.975], dim="n_boot")
        ds_proj = ds_proj.quantile([0.025, 0.975], dim="n_boot")
        ds_diff = ds_diff.quantile([0.025, 0.975], dim="n_boot")

        # Merge
        ds_out = xr.concat(
            [
                ds_proj.assign_coords(time="proj"),
                ds_hist.assign_coords(time="hist"),
                ds_diff.assign_coords(time="diff"),
            ],
            dim="time",
        )

        # Store
        gcm_name, member_name = map_store_names(ensemble, gcm, member)
        ds_out = ds_out.expand_dims(
            {
                "gcm": [gcm_name],
                "member": [member_name],
                "ssp": [ssp],
                "ensemble": [ensemble],
            }
        )
        ds_out["quantile"] = ["q025", "q975"]
        ds_out.attrs["historical_slice"] = f"{hist_slice[0]}-{hist_slice[1]}"
        ds_out.attrs["projection_slice"] = f"{proj_slice[0]}-{proj_slice[1]}"
        ds_out.to_netcdf(f"{store_path}/{store_name}")

    # Log if error
    except Exception as e:
        except_path = f"{project_code_path}/scripts/logs/gev_freq/"
        with open(
            f"{except_path}/{ensemble}_{gcm}_{member}_{ssp}_{metric_id}_{stat_name}_bootstrap.txt",
            "w",
        ) as f:
            f.write(str(e))


###############################
# GEV fit across whole ensemble
###############################
def gev_fit_all(
    metric_id,
    stationary,
    fit_method,
    periods_for_level,
    levels_for_period,
    proj_years,
    hist_years,
    bootstrap,
    include_STAR_ESDM=True,
):
    """
    Fits a GEV distribution to the entire meta-ensemble of outputs.
    Set hist_years to None to skip historical.
    """
    if bootstrap:
        gev_fit_func = partial(
            gev_fit_single_bootstrap,
            proj_slice=proj_years,
            hist_slice=hist_years,
            metric_id=metric_id,
            periods_for_level=periods_for_level,
            stationary=stationary,
            fit_method=fit_method,
            levels_for_period=levels_for_period,
        )
    else:
        gev_fit_func = partial(
            gev_fit_single,
            stationary=stationary,
            fit_method=fit_method,
            periods_for_level=periods_for_level,
            levels_for_period=levels_for_period,
            metric_id=metric_id,
        )

    #### LOCA2
    ensemble = "LOCA2"
    df_loca = get_unique_loca_metrics(metric_id)

    # Loop through
    delayed = []
    for index, row in df_loca.iterrows():
        # Get info
        gcm, member, ssp = row["gcm"], row["member"], row["ssp"]
        years = hist_years if ssp == "historical" else proj_years
        if bootstrap and ssp == "historical":
            continue
        if years is not None:
            out = dask.delayed(gev_fit_func)(
                ensemble=ensemble,
                gcm=gcm,
                member=member,
                ssp=ssp,
                years=years,
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
            years = hist_years if ssp_id == "historical" else proj_years
            if bootstrap and ssp_id == "historical":
                continue
            if not include_STAR_ESDM:
                continue
            if years is not None:
                out = dask.delayed(gev_fit_func)(
                    ensemble=ensemble,
                    gcm=gcm,
                    member=member,
                    ssp=ssp,
                    years=years,
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
            years = hist_years if ssp_id == "historical" else proj_years
            if bootstrap and ssp_id == "historical":
                continue

            if years is not None:
                out = dask.delayed(gev_fit_func)(
                    ensemble=ensemble,
                    gcm=gcm,
                    member=member,
                    ssp=ssp,
                    years=years,
                )
            delayed.append(out)

    # Compute all
    _ = dask.compute(*delayed)
