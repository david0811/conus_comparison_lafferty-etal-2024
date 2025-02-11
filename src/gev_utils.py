import math
import os
from functools import partial
from glob import glob

import dask
import numpy as np
import xarray as xr
from scipy import special
from scipy.optimize import fsolve, minimize
from scipy.stats import genextreme as gev

import sdfc_classes as sd
from utils import get_unique_loca_metrics, loca_gard_mapping
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path

###########################################################
# L-moments GEV fitting
# https://github.com/xiaoganghe/python-climate-visuals
###########################################################


# Calculate samples L-moments
def samlmom3(sample):
    """
    samlmom3 returns the first three L-moments of samples
    sample is the 1-d array
    n is the total number of the samples, j is the j_th sample
    """
    n = len(sample)
    sample = np.sort(sample.reshape(n))[::-1]
    b0 = np.mean(sample)
    b1 = np.array(
        [(n - j - 1) * sample[j] / n / (n - 1) for j in range(n)]
    ).sum()
    b2 = np.array(
        [
            (n - j - 1) * (n - j - 2) * sample[j] / n / (n - 1) / (n - 2)
            for j in range(n - 1)
        ]
    ).sum()
    lmom1 = b0
    lmom2 = 2 * b1 - b0
    lmom3 = 6 * (b2 - b1) + b0

    return lmom1, lmom2, lmom3


# Estimate GEV parameters using the function solver
def pargev_fsolve(lmom):
    """
    pargev_fsolve estimates the parameters of the Generalized Extreme Value
    distribution given the L-moments of samples
    """
    lmom_ratios = [lmom[0], lmom[1], lmom[2] / lmom[1]]
    f = lambda x, t: 2 * (1 - 3 ** (-x)) / (1 - 2 ** (-x)) - 3 - t
    G = fsolve(f, 0.01, lmom_ratios[2])[0]
    para3 = G
    GAM = math.gamma(1 + G)
    para2 = lmom_ratios[1] * G / (GAM * (1 - 2**-G))
    para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
    return para1, para2, para3


# Estimate GEV parameters using numerical approximations
def pargev(lmom):
    """
    pargev returns the parameters of the Generalized Extreme Value
    distribution given the L-moments of samples
    """
    lmom_ratios = [lmom[0], lmom[1], lmom[2] / lmom[1]]

    SMALL = 1e-5
    eps = 1e-6
    maxit = 20

    # EU IS EULER'S CONSTANT
    EU = 0.57721566
    DL2 = math.log(2)
    DL3 = math.log(3)

    # COEFFICIENTS OF RATIONAL-FUNCTION APPROXIMATIONS FOR XI
    A0 = 0.28377530
    A1 = -1.21096399
    A2 = -2.50728214
    A3 = -1.13455566
    A4 = -0.07138022
    B1 = 2.06189696
    B2 = 1.31912239
    B3 = 0.25077104
    C1 = 1.59921491
    C2 = -0.48832213
    C3 = 0.01573152
    D1 = -0.64363929
    D2 = 0.08985247

    T3 = lmom_ratios[2]
    if lmom_ratios[1] <= 0 or abs(T3) >= 1:
        raise ValueError("Invalid L-Moments")

    if T3 <= 0:
        G = (A0 + T3 * (A1 + T3 * (A2 + T3 * (A3 + T3 * A4)))) / (
            1 + T3 * (B1 + T3 * (B2 + T3 * B3))
        )

        if T3 >= -0.8:
            para3 = G
            GAM = math.exp(special.gammaln(1 + G))
            para2 = lmom_ratios[1] * G / (GAM * (1 - 2**-G))
            para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
            return para1, para2, para3
        elif T3 <= -0.97:
            G = 1 - math.log(1 + T3) / DL2

        T0 = (T3 + 3) * 0.5
        for IT in range(1, maxit):
            X2 = 2**-G
            X3 = 3**-G
            XX2 = 1 - X2
            XX3 = 1 - X3
            T = XX3 / XX2
            DERIV = (XX2 * X3 * DL3 - XX3 * X2 * DL2) / (XX2**2)
            GOLD = G
            G -= (T - T0) / DERIV

            if abs(G - GOLD) <= eps * G:
                para3 = G
                GAM = math.exp(special.gammaln(1 + G))
                para2 = lmom_ratios[1] * G / (GAM * (1 - 2**-G))
                para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
                return para1, para2, para3
        raise Exception("Iteration has not converged")
    else:
        Z = 1 - T3
        G = (-1 + Z * (C1 + Z * (C2 + Z * C3))) / (1 + Z * (D1 + Z * D2))
        if abs(G) < SMALL:
            para2 = lmom_ratios[1] / DL2
            para1 = lmom_ratios[0] - EU * para2
            para3 = 0
        else:
            para3 = G
            GAM = math.exp(special.gammaln(1 + G))
            para2 = lmom_ratios[1] * G / (GAM * (1 - 2**-G))
            para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
        return para1, para2, para3


##############################################
# Return level utils
# https://github.com/jdossgollin/2021-TXtreme
##############################################


def estimate_return_level(return_period, loc, scale, shape):
    """
    Calculate the return level given GEV parameters.
    """
    quantile = 1 - 1 / return_period
    level = loc + scale / shape * (1 - (-np.log(quantile)) ** (shape))
    return level


def xr_estimate_return_level(return_period, ds, return_params=False):
    """
    Calculate the return level given GEV parameters.
    """
    ds[f"{int(return_period)}yr_return_level"] = estimate_return_level(
        return_period, ds["loc"], ds["scale"], ds["shape"]
    )
    if not return_params:
        return ds.drop(["loc", "scale", "shape"])
    else:
        return ds


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
    data, expected_length, fit_method="lmom", optimizer=optimizer
):
    """
    Fit GEV to 1-dimensional data. Note we follow scipy convention
    of negating the shape parameter relative to other sources.
    """

    # Check if all finite
    if not np.isfinite(data).all():
        return (np.nan, np.nan, np.nan)

    # Round data
    data = np.round(data, 4)

    # # There should be no more than 1 consecutive duplicate
    # assert (np.diff(data) == 0.0).sum() <= 1, (
    #     "consecutive duplicate values in data"
    # )
    # Check length
    assert len(data) == expected_length, (
        f"data length is {len(data)}, expected {expected_length}"
    )
    # Some GARD-LENS outputs have all values zero
    if (data == 0.0).all():
        return (np.nan, np.nan, np.nan)
    # Fit
    if fit_method == "lmom":
        lmom = samlmom3(data)
        return pargev(lmom)
    elif fit_method == "mle":
        shape, location, scale = gev.fit(data, optimizer=optimizer)
        return location, scale, shape


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

    # Check if all finite
    if not np.isfinite(data).all():
        return (np.nan, np.nan, np.nan, np.nan)

    # Round data
    data = np.round(data, 4)

    # # There should be no more than 1 consecutive duplicate
    # assert (np.diff(data) == 0.0).sum() <= 1, (
    #     "consecutive duplicate values in data"
    # )

    # Check length
    expected_length = years[1] - years[0] + 1
    assert len(data) == expected_length, (
        f"data length is {len(data)}, expected {expected_length}"
    )
    # Fit
    if fit_method == "mle":
        # Initial params from L-moments
        loc, scale, shape = pargev(samlmom3(data))
        initial_params = [shape, loc, 0.0, scale]
        # Fit
        return nonstationary_optimizer(
            data, np.arange(len(data)), initial_params=initial_params
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


def fit_gev_xr(
    ds,
    metric_id,
    stationary,
    years,
    expected_length,
    fit_method,
    periods_for_level=None,
    levels_for_period=None,
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
        dask="allowed",
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
            ds_out[f"{period}yr_return_level"] = (
                scalar
                * estimate_return_level(
                    period, ds_out["loc"], ds_out["scale"], ds_out["shape"]
                )
            )

    # Return period calculations (for set levels)
    if levels_for_period is not None:
        for level in levels_for_period:
            ds_out[f"{level}{unit}_return_period"] = xr_estimate_return_period(
                scalar * level, ds_out["loc"], ds_out["scale"], ds_out["shape"]
            )

    return ds_out


def gev_fit_single(
    ensemble,
    gcm,
    member,
    ssp,
    metric_id,
    years,
    stationary,
    fit_method,
    store_path,
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
        store_name = f"{ensemble}_{gcm}_{member}_{ssp_name}_{time_name}_{stat_name}_{fit_method}.nc"

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

        # Check length is as expected
        if (
            ensemble == "GARD-LENS"
            and gcm == "ecearth3"
            and ssp_name == "historical"
        ):
            expected_length = 2014 - 1970 + 1  # GARD-LENS EC-Earth3
            assert len(ds["time"]) == expected_length, (
                f"ds length is {len(ds['time'])}, expected {expected_length}"
            )
        else:
            expected_length = years[1] - years[0] + 1
            assert len(ds["time"]) == expected_length, (
                f"ds length is {len(ds['time'])}, expected {expected_length}"
            )

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
        except_path = f"{project_code_path}/scripts/logs/gev_freq/"
        with open(
            f"{except_path}/{ensemble}_{gcm}_{member}_{ssp}_{metric_id}_{stat_name}.txt",
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
    future_years,
    hist_years,
):
    """
    Fits a GEV distribution to the entire meta-ensemble of outputs.
    Set hist_years to None to skip historical.
    """
    # Store results location
    store_path = f"{project_data_path}/extreme_value/original_grid/{metric_id}"

    #### LOCA2
    ensemble = "LOCA2"
    df_loca = get_unique_loca_metrics(metric_id)

    # Loop through
    delayed = []
    for index, row in df_loca.iterrows():
        # Get info
        gcm, member, ssp = row["gcm"], row["member"], row["ssp"]
        years = hist_years if ssp == "historical" else future_years

        if years is not None:
            out = dask.delayed(gev_fit_single)(
                ensemble=ensemble,
                gcm=gcm,
                member=member,
                ssp=ssp,
                metric_id=metric_id,
                years=years,
                stationary=stationary,
                fit_method=fit_method,
                periods_for_level=periods_for_level,
                levels_for_period=levels_for_period,
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
            if years is not None:
                out = dask.delayed(gev_fit_single)(
                    ensemble=ensemble,
                    gcm=gcm,
                    member=member,
                    ssp=ssp,
                    metric_id=metric_id,
                    years=years,
                    stationary=stationary,
                    fit_method=fit_method,
                    periods_for_level=periods_for_level,
                    levels_for_period=levels_for_period,
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
            if years is not None:
                out = dask.delayed(gev_fit_single)(
                    ensemble=ensemble,
                    gcm=gcm,
                    member=member,
                    ssp=ssp,
                    metric_id=metric_id,
                    years=years,
                    stationary=stationary,
                    fit_method=fit_method,
                    periods_for_level=periods_for_level,
                    levels_for_period=levels_for_period,
                    store_path=store_path,
                )
            delayed.append(out)

    # Compute all
    _ = dask.compute(*delayed)
