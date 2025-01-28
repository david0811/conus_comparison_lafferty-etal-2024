import os
from glob import glob

import arviz as az
import dask
import numpy as np
import pandas as pd
import pymc as pm
import pymc_extras.distributions as pmx
import xarray as xr

from utils import city_list, get_unique_loca_metrics, loca_gard_mapping
from utils import roar_data_path as project_data_path

#####################################
# Calculating city timeseries metrics
#####################################


def get_city_timeseries(
    city,
    ensemble,
    gcm,
    member,
    ssp,
    metric_id,
    project_data_path=project_data_path,
):
    """
    Reads and returns the annual max timeseries for a
    selected city, ensemble, GCMs, SSPs.
    """
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

    # Add all info
    ds = ds.expand_dims(
        {
            "gcm": [gcm_name],
            "member": [member_name],
            "ssp": [ssp],
            "ensemble": [ensemble],
        }
    )
    ds["time"] = ds["time"].dt.year

    # Extract city data
    lat, lon = city_list[city]
    if ensemble == "LOCA2":
        df_loc = (
            ds.sel(lat=lat, lon=360 + lon, method="nearest")
            .to_dataframe()
            .drop(columns=["lat", "lon"])
            .dropna()
        )
    elif ensemble == "STAR-ESDM":
        df_loc = (
            ds.sel(latitude=lat, longitude=360 + lon, method="nearest")
            .to_dataframe()
            .drop(columns=["latitude", "longitude"])
            .dropna()
        )
    else:
        df_loc = (
            ds.sel(lat=lat, lon=lon, method="nearest")
            .to_dataframe()
            .dropna()
            .drop(columns=["lat", "lon"])
        )

    # Return
    return df_loc.reset_index()


def get_city_timeseries_all(
    city,
    metric_id,
    project_data_path=project_data_path,
):
    """
    Loop through all meta-ensemble members and calculate the city timeseries.
    """
    # Check if done
    if os.path.exists(
        f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv"
    ):
        return None

    #### LOCA2
    ensemble = "LOCA2"
    df_loca = get_unique_loca_metrics(metric_id)

    # Loop through
    delayed = []
    for index, row in df_loca.iterrows():
        # Get info
        gcm, member, ssp = row["gcm"], row["member"], row["ssp"]

        out = dask.delayed(get_city_timeseries)(
            city=city,
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
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
        out = dask.delayed(get_city_timeseries)(
            city=city,
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
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
        out = dask.delayed(get_city_timeseries)(
            city=city,
            ensemble=ensemble,
            gcm=gcm,
            member=member,
            ssp=ssp,
            metric_id=metric_id,
        )
        delayed.append(out)

    # Compute all
    df = pd.concat(dask.compute(*delayed))

    # Store
    df.to_csv(
        f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv",
        index=False,
    )


###########################
# Fitting Bayesian GEV
###########################
def _fit_bayesian_gev(
    data,
    loc_mu,
    loc_sigma,
    scale_sigma,
    store_results,
    years,
    stationary,
    return_periods,
    shape_sigma=1,
    trend_mu=0,
    trend_sigma=5,
    n_draws=5000,
    n_tune=2000,
):
    # Define model
    if stationary:
        # Stationary model
        with pm.Model() as model:
            # Priors
            mu = pm.Normal("loc", mu=loc_mu, sigma=loc_sigma)
            sigma = pm.HalfNormal("scale", sigma=scale_sigma)
            xi = pm.Normal("shape", mu=0.0, sigma=shape_sigma)

            # Estimation
            gev = pmx.GenExtreme(
                "gev", mu=mu, sigma=sigma, xi=xi, observed=data
            )

            # Return level
            for p in return_periods:
                z_p = pm.Deterministic(
                    f"{p}yr_return_level",
                    mu - sigma / xi * (1 - (-np.log(1 - 1.0 / p)) ** (-xi)),
                )
    else:
        # Non-stationary model
        time = np.arange(years[0], years[1] + 1)
        time_zeroed = time - time[0]
        coords = {"time": time}
        with pm.Model(coords=coords) as model:
            # Priors
            intcp = pm.Normal("intcp", mu=loc_mu, sigma=loc_sigma)
            sigma = pm.HalfNormal("scale", sigma=scale_sigma)
            xi = pm.Normal("shape", mu=0.0, sigma=shape_sigma)
            trend = pm.Normal("trend", mu=trend_mu, sigma=trend_sigma)

            # Calculate mu as a deterministic variable
            mu = pm.Deterministic(
                "mu", intcp + trend * time_zeroed, dims="time"
            )

            # Estimation
            gev = pmx.GenExtreme(
                "gev",
                mu=mu,
                sigma=sigma,
                xi=xi,
                observed=data,
            )

            # Return level
            for p in return_periods:
                z_p = pm.Deterministic(
                    f"{p}yr_return_level",
                    mu - sigma / xi * (1 - (-np.log(1 - 1.0 / p)) ** (-xi)),
                    dims="time",
                )

    # Sample
    with model:
        trace = pm.sample(
            draws=n_draws,
            cores=3,
            chains=3,
            tune=n_tune,
            target_accept=0.98,
            nuts_sampler="blackjax",
            progressbar=False,
        )

    return trace


def fit_bayesian_gev_single(
    city,
    metric_id,
    ensemble,
    gcm,
    member,
    ssp,
    years,
    stationary,
    return_periods,
    store_results=True,
    project_data_path=project_data_path,
    **kwargs,
):
    """
    Fits the Bayesian GEV model to a selected city, ensemble, GCM, member, SSP, and years.
    """

    # Skip if done
    stationary_string = "stat" if stationary else "nonstat"
    store_name = f"{city}_{metric_id}_{ensemble}_{gcm}_{member}_{ssp}_{years[0]}-{years[1]}_{stationary_string}"
    if os.path.exists(
        f"{project_data_path}/extreme_value/cities/original_grid/bayes/{store_name}.nc",
    ):
        return None

    # Read and select data
    df = pd.read_csv(
        f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv"
    )

    df_sel = df[
        (df["ensemble"] == ensemble)
        & (df["gcm"] == gcm)
        & (df["member"] == member)
        & (df["ssp"] == ssp)
        & (df["time"] >= years[0])
        & (df["time"] <= years[1])
    ]

    # Skip invalid SSP-years combinations
    if len(df_sel) == 0:
        return None

    _, var_id = metric_id.split("_")
    data = df_sel[var_id].to_numpy()

    # Set priors based on variable
    if var_id == "tasmax":
        loc_mu = 30.0
        loc_sigma = 10.0
        scale_sigma = 10.0
    elif var_id == "tasmin":
        loc_mu = -20.0
        loc_sigma = 10.0
        scale_sigma = 10.0
    elif var_id == "pr":
        loc_mu = 100.0
        loc_sigma = 50.0
        scale_sigma = 50.0

    # Do the fit
    trace = _fit_bayesian_gev(
        data,
        loc_mu,
        loc_sigma,
        scale_sigma,
        store_results,
        years,
        stationary,
        return_periods,
        **kwargs,
    )

    # Re-run if convergence issues
    count = 0
    while (az.summary(trace)["r_hat"] > 1.01).all() and count <= 5:
        # Re-run with perturbed priors
        trace = _fit_bayesian_gev(
            data,
            loc_mu * np.random.uniform(0.9, 1.1),
            loc_sigma * np.random.uniform(0.9, 1.1),
            scale_sigma * np.random.uniform(0.9, 1.1),
            store_results,
            years,
            stationary,
            return_periods,
            **kwargs,
        )
        count += 1

    # Assign coords to attrs
    trace.attrs["ensemble"] = ensemble
    trace.attrs["gcm"] = gcm
    trace.attrs["member"] = member
    trace.attrs["ssp"] = ssp

    # Store if desired
    if store_results:
        trace.to_netcdf(
            f"{project_data_path}/extreme_value/cities/original_grid/bayes/{store_name}.nc",
        )
    else:
        return trace


def fit_bayesian_gev_ensemble(
    city,
    metric_id,
    years,
    stationary,
    return_periods,
    store_results=True,
    project_data_path=project_data_path,
    dask=True,
    **kwargs,
):
    """
    Fits the Bayesian GEV model to a selected city, all ensemble members, GCMs, SSPs
    """
    # Get unique combos
    df = pd.read_csv(
        f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv"
    )
    df = df.set_index(["ensemble", "gcm", "member", "ssp"]).sort_index()
    combos = df.index.unique()

    # Loop through
    delayed = []
    for combo in combos:
        ensemble, gcm, member, ssp = combo
        if dask:
            tmp = dask.delayed(fit_bayesian_gev_single)(
                city=city,
                metric_id=metric_id,
                ensemble=ensemble,
                gcm=gcm,
                member=member,
                ssp=ssp,
                years=years,
                stationary=stationary,
                return_periods=return_periods,
                store_results=store_results,
                project_data_path=project_data_path,
                **kwargs,
            )
            delayed.append(tmp)
        else:
            print(f"{ensemble} {gcm} {member} {ssp}")
            fit_bayesian_gev_single(
                city=city,
                metric_id=metric_id,
                ensemble=ensemble,
                gcm=gcm,
                member=member,
                ssp=ssp,
                years=years,
                stationary=stationary,
                return_periods=return_periods,
                store_results=store_results,
                project_data_path=project_data_path,
                **kwargs,
            )

    if dask:
        _ = dask.compute(*delayed)


def gather_bayesian_gev_results(
    city,
    metric_id,
    return_periods,
    project_data_path=project_data_path,
):
    """
    Gathers all Bayesian GEV results for a selected city and metric, stores return levels in a dataframe.
    """

    # Get all fits
    files = glob(
        f"{project_data_path}/extreme_value/cities/original_grid/bayes/{city}_{metric_id}_*.nc"
    )

    # Loop through all
    df_out = []
    for file in files:
        # Read trace
        trace = az.from_netcdf(file)
        ensemble = trace.attrs["ensemble"]
        gcm = trace.attrs["gcm"]
        member = trace.attrs["member"]
        ssp = trace.attrs["ssp"]

        conv_flag = (az.summary(trace)["r_hat"] < 1.01).all()

        # Return CIs for return levels
        return_level_columns = [f"{p}yr_return_level" for p in return_periods]
        rl_quantiles = (
            trace["posterior"][return_level_columns]
            .quantile([0.025, 0.5, 0.957], dim=["chain", "draw"])
            .to_dataframe()
            .reset_index()
        )
        rl_mean = (
            trace["posterior"][return_level_columns]
            .mean(dim=["chain", "draw"])
            .expand_dims(quantile=["mean"])
            .to_dataframe()
            .reset_index()
        )

        # Concat
        df = pd.concat([rl_quantiles, rl_mean])
        df["ensemble"] = ensemble
        df["gcm"] = gcm
        df["member"] = member
        df["ssp"] = ssp
        df["rhat_good"] = conv_flag

        df_out.append(df)

    return pd.concat(df_out)
