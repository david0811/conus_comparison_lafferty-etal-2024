import os
import numpy as np
import xarray as xr
from glob import glob
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path
from utils import map_store_names
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble", type=str, required=True)
    parser.add_argument("--gcm", type=str, required=True)
    parser.add_argument("--member", type=str, required=True)
    parser.add_argument("--ssp", type=str, required=True)
    parser.add_argument("--metric_id", type=str, required=True)
    return parser.parse_args()


def transform_r_to_xr(
    return_periods,
    return_period_years,
    param_results_main,
    param_results_lower,
    param_results_upper,
    return_levels_main,
    return_levels_lower,
    return_levels_upper,
    lats,
    lat_name,
    lons,
    lon_name,
):
    # Gather
    quantiles = ["main", "q025", "q975"]

    # Create a dictionary to store all parameters and return levels
    data_vars = {}

    # Add parameter variables with quantiles
    param_names = ["loc_intcp", "loc_trend", "scale", "shape"]
    for i, param_name in enumerate(param_names):
        # Stack the three quantiles into a new dimension
        param_data = np.stack(
            [
                param_results_main[:, :, i],
                param_results_lower[:, :, i],
                param_results_upper[:, :, i],
            ],
            axis=0,
        )

        data_vars[param_name] = xr.DataArray(
            data=param_data,
            dims=["quantile", lat_name, lon_name],
            coords={"quantile": quantiles, lat_name: lats, lon_name: lons},
        )

    # Add return levels with quantiles and time dimension
    for i, period in enumerate(return_periods):
        var_name = f"{period}yr_return_level"

        # Reshape to include time dimension
        # Original shape: [quantile, lat, lon, time]
        # This reorganizes the data to put time as a dimension
        level_data = np.stack(
            [
                return_levels_main[:, :, i, :],  # [lat, lon, time]
                return_levels_lower[:, :, i, :],  # [lat, lon, time]
                return_levels_upper[:, :, i, :],  # [lat, lon, time]
            ],
            axis=0,
        )  # Result: [quantile, lat, lon, time]

        # Transpose to get [quantile, time, lat, lon] for better organization
        level_data = np.transpose(level_data, (0, 3, 1, 2))

        data_vars[var_name] = xr.DataArray(
            data=level_data,
            dims=["quantile", "time", lat_name, lon_name],
            coords={
                "quantile": quantiles,
                "time": return_period_years,
                lat_name: lats,
                lon_name: lons,
            },
            attrs={
                "long_name": f"{period}-year return level",
            },
        )

    # Create the dataset
    ds_res = xr.Dataset(data_vars=data_vars)

    # Filter NaNs
    ds_res = ds_res.where(ds_res != -1234.0)

    return ds_res


def fit_mle_nonstat_R(
    metric_id,
    ensemble,
    gcm,
    member,
    ssp,
    periods_for_level=[10, 25, 50, 100],
    return_period_years=[1975, 2000, 2025, 2050, 2075, 2100],
):
    # Check if done
    agg_id, var_id = metric_id.split("_")
    gcm_name, member_name = map_store_names(ensemble, gcm, member)

    store_name = f"{ensemble}_{gcm}_{member}_{ssp}_all_nonstat_mle_main.nc"
    store_path = f"{project_data_path}/extreme_value/original_grid/{metric_id}"

    if os.path.exists(f"{store_path}/{store_name}"):
        return None

    try:
        # Explicitly deactivate any previous conversion
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri

        try:
            numpy2ri.deactivate()
        except Exception:
            pass

        # Activate numpy to R conversion
        numpy2ri.activate()

        # Get fitting function
        r_code = f"source('{project_code_path}/src/gev_utils.R')"
        robjects.r(r_code)
        fit_nonstat_gev_mle = robjects.r["fit_nonstat_gev_mle_parallel"]

        ## Read data file
        # LOCA2
        if ensemble == "LOCA2":
            # Projection
            proj_files = glob(
                f"{project_data_path}/metrics/LOCA2/{metric_id}_{gcm}_{member}_{ssp}_*.nc"
            )
            ds = xr.concat([xr.open_dataset(file) for file in proj_files], dim="time")
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

        # Get data to fit
        starting_year = int(ds["time"].min().dt.year)
        ds_np = ds[var_id].to_numpy()

        if agg_id == "min":
            scalar = -1.0
        else:
            scalar = 1.0

        # Fit
        result = fit_nonstat_gev_mle(
            ds_np * scalar,
            starting_year=robjects.IntVector([starting_year]),
            periods_for_level=robjects.IntVector(periods_for_level),
            return_period_years=robjects.IntVector(return_period_years),
        )

        # Access the results
        param_results_main = np.array(result.rx2("param_results_main"))
        param_results_lower = np.array(result.rx2("param_results_lower"))
        param_results_upper = np.array(result.rx2("param_results_upper"))
        return_levels_main = np.array(result.rx2("return_levels_main"))
        return_levels_lower = np.array(result.rx2("return_levels_lower"))
        return_levels_upper = np.array(result.rx2("return_levels_upper"))

        # Transform to xr
        lat_name = "latitude" if "latitude" in ds.dims else "lat"
        lon_name = "longitude" if "longitude" in ds.dims else "lon"

        ds_res = transform_r_to_xr(
            periods_for_level,
            return_period_years,
            param_results_main,
            param_results_lower,
            param_results_upper,
            return_levels_main,
            return_levels_lower,
            return_levels_upper,
            ds[lat_name].to_numpy(),
            lat_name,
            ds[lon_name].to_numpy(),
            lon_name,
        )

        # Add attributes
        ds_res = ds_res.expand_dims(
            {
                "gcm": [gcm_name],
                "member": [member_name],
                "ssp": [ssp],
                "ensemble": [ensemble],
            }
        )

        # Store
        ds_res.to_netcdf(f"{store_path}/{store_name}")

    except Exception as e:
        except_path = f"{project_code_path}/scripts/logs/gev_freq/"
        with open(
            f"{except_path}/{ensemble}_{gcm}_{member}_{ssp}_{metric_id}_nonstat_mle_R.txt",
            "w",
        ) as f:
            f.write(str(e))


if __name__ == "__main__":
    args = parse_args()
    fit_mle_nonstat_R(**vars(args))
