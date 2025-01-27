import os
from glob import glob

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
    """ """
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
def fit_bayesian_gev_single(
    data,
    info,
    loc_mu,
    loc_sigma,
    scale_sigma,
    store_results,
    years,
    stationary,
    shape_sigma=1,
    trend_mu=0,
    trend_sigma=5,
    n_draws=5000,
    n_tune=2000,
    return_periods=[100],
    project_data_path=project_data_path,
):
    """ """

    # Get data info
    city, metric_id, ensemble, gcm, member, ssp = info

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
            cores=1,
            chains=3,
            tune=n_tune,
            target_accept=0.98,
            nuts_sampler="blackjax",
            progressbar=False,
        )
    # Store if desired
    stationary_string = "stat" if stationary else "nonstat"
    store_name = f"{city}_{metric_id}_{ensemble}_{gcm}_{member}_{ssp}_{years[0]}-{years[1]}_{stationary_string}"
    if store_results:
        trace.to_netcdf(
            f"{project_data_path}/extreme_value/cities/original_grid/bayes/{store_name}.nc",
        )

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

    return df
