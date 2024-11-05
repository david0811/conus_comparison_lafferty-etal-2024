import os
from glob import glob

import dask
import xarray as xr

from dask.distributed import LocalCluster, Client

import metric_funcs as mf


#### Preliminaries
# NOTE: this is run on a different system from other datasets
# Update these for reproduction
project_data_path = "/home/fs01/dcl257/projects/data/conus_comparison_lafferty-etal-2024"
project_code_path = "/home/fs01/dcl257/projects/conus_comparison_lafferty-etal-2024"
gard_path = "/home/shared/vs498_0001/GARD-LENS"  # GARD-LENS raw
gard_gcms = ["canesm5", "cesm2", "ecearth3"]


# Metric calculation function
def calculate_metric(metric_func, var_id, model_member, needed_vars, gard_path, out_path):
    """
    Inputs: selected model, member, variable, and metric to calculate (from GARD-LENS)
    Outputs: calculated (annual) metric
    """
    try:
        # Check if done
        if os.path.isfile(out_path):
            print(f"{model_member} already done.")
            return None

        # Read
        if model_member.split("_")[0] == "ecearth3":
            time_range = "1970_2100"
        else:
            time_range = "1950_2100"

        # Read
        ds_tmp = xr.merge(
            [
                xr.open_dataset(
                    f"{gard_path}/{var}/GARDLENS_{model_member}_{var}_{time_range}_CONUS.nc",
                    chunks="auto",
                )
                for var in needed_vars
            ]
        )

        # Calculate metric
        ds_out = metric_func(ds_tmp, var_id)

        # Store
        ds_out.to_netcdf(out_path)
        print(f"{model_member}")

    # Log if error
    except Exception as e:
        except_path = f"{project_code_path}/code/logs"
        with open(f"{except_path}/{model_member}_{var_id}_GARDLENS.txt", "w") as f:
            f.write(str(e))


def main():
    cluster = LocalCluster(n_workers=10)

    # Connect to the Dask client
    client = Client(cluster)
    dask.config.set(scheduler="distributed")

    # Check all same
    for gcm in gard_gcms:
        t_mean_files = glob(f"{gard_path}/t_mean/GARDLENS_{gcm}_*.nc")
        t_range_files = glob(f"{gard_path}/t_range/GARDLENS_{gcm}_*.nc")
        pcp_files = glob(f"{gard_path}/pcp/GARDLENS_{gcm}_*.nc")
        assert len(t_mean_files) == len(t_range_files)
        assert len(t_mean_files) == len(pcp_files)

    # Get all model members
    models_members = glob(f"{gard_path}/t_mean/GARDLENS_*.nc")
    models_members = [file.split("GARDLENS")[1].split("t_")[0][1:-1] for file in models_members]

    ### Cooling degree days: max
    for model_member in models_members:
        calculate_metric(
            metric_func=mf.calculate_dd_max,
            var_id="cdd",
            model_member=model_member,
            needed_vars=["t_mean", "t_range"],
            gard_path=gard_path,
            out_path=f"{project_data_path}/metrics/GARD-LENS/max_cdd_{model_member}_ssp370.nc",
        )

    ### Cooling degree days: sum
    for model_member in models_members:
        calculate_metric(
            metric_func=mf.calculate_dd_sum,
            var_id="cdd",
            model_member=model_member,
            needed_vars=["t_mean", "t_range"],
            gard_path=gard_path,
            out_path=f"{project_data_path}/metrics/GARD-LENS/sum_cdd_{model_member}_ssp370.nc",
        )

    ### Heating degree days: max
    for model_member in models_members:
        calculate_metric(
            metric_func=mf.calculate_dd_max,
            var_id="hdd",
            model_member=model_member,
            needed_vars=["t_mean", "t_range"],
            gard_path=gard_path,
            out_path=f"{project_data_path}/metrics/GARD-LENS/max_hdd_{model_member}_ssp370.nc",
        )

    ### Heating degree days: sum
    for model_member in models_members:
        calculate_metric(
            metric_func=mf.calculate_dd_sum,
            var_id="hdd",
            model_member=model_member,
            needed_vars=["t_mean", "t_range"],
            gard_path=gard_path,
            out_path=f"{project_data_path}/metrics/GARD-LENS/sum_hdd_{model_member}_ssp370.nc",
        )


if __name__ == "__main__":
    main()
