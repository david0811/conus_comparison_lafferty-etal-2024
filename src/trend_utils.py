import os
from functools import partial
from glob import glob

import dask
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import xarray as xr

from utils import check_data_length, loca_gard_mapping
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path


# Linear regression function
def linear_regression(X, y, expected_length=None):
    """
    Fit linear regression to data.
    """
    if np.isnan(y).all():
        return np.array([np.nan, np.nan])

    # Check length of non-NaNs
    if expected_length is not None:
        non_nans = np.count_nonzero(~np.isnan(y))
        assert non_nans == expected_length, (
            f"data length is {non_nans}, expected {expected_length}"
        )
    # Should be no zeros in data, this was happening with some CDD, HDD instead of NaN
    assert np.sum(y == 0.0) < 5, "At least 5 zeros in data"

    return np.polyfit(X, y, 1)


def linear_regression_bootstrap(X, y, n_boot, expected_length=None, return_samples=False):
    # Check 
    if np.isnan(y).all():
        return np.array([[np.nan, np.nan], [np.nan, np.nan]])

    # Check length of non-NaNs
    if expected_length is not None:
        non_nans = np.count_nonzero(~np.isnan(y))
        assert non_nans == expected_length, (
            f"data length is {non_nans}, expected {expected_length}"
        )
    # Should be no zeros in data, this was happening with some CDD, HDD instead of NaN
    assert np.sum(y == 0.0) < 5, "At least 5 zeros in data"

    # Fit original model
    slope, intercept = np.polyfit(X, y, 1)

    # Bootstrap the residuals
    boot_slope = np.zeros(n_boot)
    boot_intercept = np.zeros(n_boot)
    predictions = (intercept + slope * X)
    residuals = y - predictions
    boot_sample = np.random.choice(residuals, size=(len(residuals), n_boot), replace=True)
    # Vectorize regression with sklearn
    y_matrix = np.tile(y, (n_boot, 1)).T + boot_sample
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X.reshape(-1, 1), y_matrix)
    boot_slope, boot_intercept = lr.coef_.T[0], lr.intercept_
    
    # Return
    if return_samples:
        return boot_slope, boot_intercept
    else:
        return np.array([np.quantile(boot_slope, [0.025, 0.975]), np.quantile(boot_intercept, [0.025, 0.975])])


# Fit trend for single output
def trend_fit_single(
    ensemble,
    gcm,
    member,
    ssp,
    metric_id,
    years,
    store_path,
    n_boot=None,
    return_samples=False,
    project_data_path=project_data_path,
    project_code_path=project_code_path,
):
    """
    Read a single metric file and fit the trend.
    """
    try:
        # Check if done
        if years == [1950, 2014]:
            ssp_name = "historical"
        else:
            ssp_name = ssp
        time_name = f"{years[0]}-{years[1]}" if years is not None else "all"
        boot_name = f"bootstrap{n_boot}" if n_boot is not None else "main"
        store_name = f"{ensemble}_{gcm}_{member}_{ssp_name}_{time_name}_{boot_name}.nc"

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

        # Fit trend
        var_id = metric_id.split("_")[1]
        expected_length = years[1] - years[0] + 1

        # Main
        linear_regression_func = partial(
            linear_regression,
            expected_length=expected_length,
        )
        main_result = xr.apply_ufunc(
            linear_regression_func,
            ds["time"],
            ds[var_id],
            input_core_dims=[
                ["time"],
                ["time"],
            ],
            output_core_dims=[["coef"]],
            vectorize=True,
            dask="forbidden",
            output_dtypes=[float],
            dask_gufunc_kwargs={"output_sizes": {"coef": 2}},
        )
        # Bootstrap
        if n_boot is not None:
            linear_regression_func = partial(
                linear_regression_bootstrap,
                n_boot=n_boot,
                expected_length=expected_length,
                return_samples=return_samples,
            )
            if return_samples:
                dask_gufunc_kwargs = {"output_sizes": {"coef": 2, "n_boot": n_boot}}
                output_core_dims = [["coef", "n_boot"]]
            else:
                dask_gufunc_kwargs = {"output_sizes": {"coef": 2, "quantile": 2}}
                output_core_dims = [["coef", "quantile"]]
            bootstrap_result = xr.apply_ufunc(
                linear_regression_func,
                ds["time"],
                ds[var_id],
                input_core_dims=[["time"], ["time"]],
                output_core_dims=output_core_dims,
                vectorize=True,
                dask="forbidden",
                output_dtypes=[float],
                dask_gufunc_kwargs=dask_gufunc_kwargs,
            )
        
        # Gather results
        ds_main = xr.Dataset({"intcp": main_result.sel(coef=1), "slope": main_result.sel(coef=0)})
        if return_samples:
            ds_main = ds_main.assign_coords({"n_boot": np.arange(n_boot)})
        else:
            ds_main = ds_main.assign_coords({"quantile": ["main"]})

        if n_boot is not None:
            if return_samples:
                ds_boot = xr.Dataset({"intcp": bootstrap_result.sel(coef=1), "slope": bootstrap_result.sel(coef=0)})
                ds_boot = ds_boot.assign_coords({"n_boot": np.arange(n_boot)})
            else:
                ds_boot = xr.Dataset({"intcp": bootstrap_result.sel(coef=1), "slope": bootstrap_result.sel(coef=0)})
                ds_boot = ds_boot.assign_coords({"quantile": ["q025", "q975"]})

        # Merge
        if n_boot is not None:
            if return_samples:
                ds_out = xr.concat([ds_main, ds_boot], dim="n_boot")
            else:
                ds_out = xr.concat([ds_main, ds_boot], dim="quantile")
        else:
            ds_out = ds_main

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
        # Make sure not all NaNs
        assert np.count_nonzero(~np.isnan(ds_out["slope"])), "all NaNs in fit"
        # Or alternatively, check number of unique values
        assert len(np.unique(ds_out["slope"])) > 1, "all slopes are identical"
        # Store
        ds_out.to_netcdf(f"{store_path}/{store_name}")

    # Log if error
    except Exception as e:
        except_path = f"{project_code_path}/scripts/logs/trend"
        with open(
            f"{except_path}/{ensemble}_{gcm}_{member}_{ssp}_{metric_id}.txt",
            "w",
        ) as f:
            f.write(str(e))

###############################
# Trend fit across whole ensemble
###############################
def get_unique_loca_metrics(metric_id):
    """
    Return unique LOCA2 combinations for given metric_id.
    """
    # Read all
    files = glob(f"{project_data_path}/metrics/LOCA2/{metric_id}_*")

    # Extract all info
    df = pd.DataFrame(columns=["gcm", "member", "ssp"])
    for file in files:
        _, _, gcm, member, ssp, _ = file.split("/")[-1].split("_")
        df = pd.concat(
            [
                df,
                pd.DataFrame({"gcm": gcm, "member": member, "ssp": ssp}, index=[0]),
            ]
        )

    # Return unique
    return df.drop_duplicates().reset_index()


def trend_fit_all(metric_id, n_boot=None, future_years=[2015, 2100], hist_years=None):
    """
    Fits a trend to the entire meta-ensemble of outputs.
    """
    # Store results location
    store_path = f"{project_data_path}/trends/original_grid/{metric_id}"

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
            out = dask.delayed(trend_fit_single)(
                ensemble=ensemble,
                gcm=gcm,
                member=member,
                ssp=ssp,
                metric_id=metric_id,
                years=years,
                store_path=store_path,
                n_boot=n_boot,
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
                out = dask.delayed(trend_fit_single)(
                    ensemble=ensemble,
                    gcm=gcm,
                    member=member,
                    ssp=ssp,
                    metric_id=metric_id,
                    years=years,
                    store_path=store_path,
                    n_boot=n_boot,
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
                out = dask.delayed(trend_fit_single)(
                    ensemble=ensemble,
                    gcm=gcm,
                    member=member,
                    ssp=ssp,
                    metric_id=metric_id,
                    years=years,
                    store_path=store_path,
                    n_boot=n_boot,
                )
                delayed.append(out)

    # Compute all
    _ = dask.compute(*delayed)


#########################
# Fit trend to city
#########################
def trend_fit_city(metric_id, city, years=[2015, 2100]):
    """
    Fits a trend to the entire meta-ensemble of outputs.
    """
    # Store results location
    store_path = f"{project_data_path}/trends/cities/original_grid/"

    # Check if done
    year_str = f"{years[0]}-{years[1]}"
    if os.path.exists(f"{store_path}/{city}_{metric_id}_{year_str}.csv"):
        return None

    # Read city data
    df = pd.read_csv(f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv")
    var_id = metric_id.split("_")[1]

    # Select time period
    df = df[df["time"].isin(range(years[0], years[1] + 1))]

    # Apply over all groups
    groups = df.groupby(["gcm", "member", "ssp", "ensemble"])

    # Loop through
    results = []

    for name, group in groups:
        # Sort by time for regression
        group = group.sort_values("time")

        # Get X and y
        X = group["time"].values
        y = group[var_id].values

        # Skip TaiESM1 from STAR (too hot!)
        if name[3] == "STAR-ESDM" and name[0] == "TaiESM1":
            continue

        # Apply linear regression
        try:
            expected_length = years[1] - years[0] + 1
            slope, intercept = linear_regression(X, y, expected_length)
            results.append(
                {
                    "gcm": name[0],
                    "member": name[1],
                    "ssp": name[2],
                    "ensemble": name[3],
                    "slope": slope,
                    "intercept": intercept,
                }
            )
        except Exception as e:
            print(f"Error for {city} {metric_id} {name}: {e}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Store
    results_df.to_csv(f"{store_path}/{city}_{metric_id}_{year_str}.csv", index=False)
