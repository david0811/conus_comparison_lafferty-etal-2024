import os
from functools import partial
from glob import glob

import dask
import numpy as np
import xarray as xr
from scipy.optimize import minimize
from scipy.stats import genextreme as gev

from lmom_utils import (
    pargev_bootstrap_numba,
    pargev_numba,
    pargev_numpy,
    samlmom3_bootstrap_numba,
    samlmom3_numba,
    samlmom3_numpy,
)
from gev_utils import (
    estimate_return_level,
    xr_estimate_return_level,
    xr_estimate_return_period,
)

from utils import check_data_length, get_unique_loca_metrics, map_store_names
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path


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

    # If n_boot is 1, return main results
    if n_boot == 1:
        params_out[0, :] = loc, scale, shape
        return_levels_out[0, :] = estimate_return_level(
            np.array(periods_for_level), *params_out[0]
        )
    else:
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


def fit_gev_xr(
    ds,
    metric_id,
    years,
    expected_length,
    fit_method,
    periods_for_level=None,
    levels_for_period=None,
    stationary=True,
    numba=True,
    dask="allowed",
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
        raise NotImplementedError("Non-stationary GEV fitting not implemented")

    fit_params = xr.apply_ufunc(
        fit_gev_1d,
        ds * scalar,
        input_core_dims=[["time"]],
        output_core_dims=output_core_dims,
        vectorize=True,
        dask=dask,
        output_dtypes=output_dtypes,
    )

    # Create a dataset with the output parameters
    ds_out = xr.merge(
        [
            fit_params[0][var_id].rename("loc"),
            fit_params[1][var_id].rename("scale"),
            fit_params[2][var_id].rename("shape"),
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
    if years == [1950, 2014] and ssp.startswith("ssp"):
        ssp_name = "historical"
    else:
        ssp_name = ssp
    time_name = f"{years[0]}-{years[1]}" if years is not None else "all"
    stat_name = "stat" if stationary else "nonstat"
    store_name = f"{ensemble}_{gcm}_{member}_{ssp_name}_{time_name}_{stat_name}_{fit_method}_main.nc"

    # Get scalar
    metric_id = store_path.split("/")[-1]
    if metric_id == "min_tasmin":
        scalar = -1.0
    else:
        scalar = 1.0

    assert os.path.exists(f"{store_path}/{store_name}"), (
        f"main fit {store_path}/{store_name} not found"
    )
    ds_fit_main = xr.open_dataset(f"{store_path}/{store_name}")

    if n_boot > 1:
        # Generate bootstrap sample (careful memory requirements!)
        # Get dimensions for latitude and longitude, accounting for different naming conventions
        if ensemble == "TGW":
            lat_name = "south_north"
            lon_name = "west_east"
        else:
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
    else:
        ds = ds_fit_main

    # Return level calculations (for set periods)
    if periods_for_level is not None:
        for period in periods_for_level:
            if f"{period}yr_return_level" not in ds.data_vars:
                ds = xr_estimate_return_level(period, ds, scalar, return_params=True)

    # Return
    if return_samples or n_boot == 1:
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
    periods_for_level=[10, 25, 50, 100],
    levels_for_period=None,
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
    periods_for_level=[10, 25, 50, 100],
    stationary=True,
    fit_method="lmom",
    n_boot_proj=100,
    n_boot_hist=1,
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
        store_name = f"{ensemble}_{gcm}_{member}_{ssp}_{time_name}_{stat_name}_{fit_method}_nbootproj{n_boot_proj}_nboothist{n_boot_hist}.nc"
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
            n_boot=n_boot_proj,
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
            n_boot=n_boot_hist,
            store_path=store_path,
            periods_for_level=periods_for_level,
            levels_for_period=levels_for_period,
        )
        # Take differences
        if n_boot_hist > 1:
            ds_diff = ds_proj - ds_hist
            ds_chfc = ds_proj / ds_hist
        else:
            ds_diff = ds_proj - ds_hist.sel(quantile="main")
            ds_chfc = ds_proj / ds_hist.sel(quantile="main")

        # Take quantiles
        if n_boot_hist > 1:
            ds_hist = ds_hist.quantile([0.025, 0.975], dim="n_boot")
            ds_hist["quantile"] = ["q025", "q975"]

        ds_proj = ds_proj.quantile([0.025, 0.975], dim="n_boot")
        ds_diff = ds_diff.quantile([0.025, 0.975], dim="n_boot")
        ds_chfc = ds_chfc.quantile([0.025, 0.975], dim="n_boot")

        # Merge
        ds_out = xr.concat(
            [
                ds_proj.assign_coords(time="proj"),
                ds_diff.assign_coords(time="diff"),
                ds_chfc.assign_coords(time="chfc"),
            ],
            dim="time",
        )
        ds_out["quantile"] = ["q025", "q975"]

        # Add historical only if n_boot_hist > 1
        if n_boot_hist > 1:
            ds_out = xr.concat([ds_out, ds_hist.assign_coords(time="hist")], dim="time")

        # Store
        if n_boot_hist > 1:
            gcm_name, member_name = map_store_names(ensemble, gcm, member)
            ds_out = ds_out.expand_dims(
                {
                    "gcm": [gcm_name],
                    "member": [member_name],
                    "ssp": [ssp],
                    "ensemble": [ensemble],
                }
            )
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
    n_boot_proj=100,
    n_boot_hist=1,
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
            n_boot_proj=n_boot_proj,
            n_boot_hist=n_boot_hist,
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
