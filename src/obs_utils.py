import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from utils import roar_data_path as project_data_path


def _preprocess_gmet(ds):
    member = ds.encoding["source"].split("_")[-7]
    return ds.expand_dims(member=[member])


def read_all_obs(metric_id, project_data_path=project_data_path):
    """
    Read all obs datasets for a given metric.
    """
    # Load all
    ds_gmet = xr.open_mfdataset(
        f"{project_data_path}/extreme_value/loca_grid/{metric_id}/GARD-LENS_gmet_*_obs_1980-2016_stat_lmom_main_nearest.nc",
        preprocess=_preprocess_gmet,
    )
    ds_nclim = xr.open_dataset(
        f"{project_data_path}/extreme_value/loca_grid/{metric_id}/STAR-ESDM_nclimgrid_None_obs_1951-2014_stat_lmom_main_nearest.nc"
    )

    ds_livneh = xr.open_dataset(
        f"{project_data_path}/extreme_value/original_grid/{metric_id}/LOCA2_livneh-unsplit_None_obs_1950-2014_stat_lmom_main.nc"
    )
    ds_livneh["lon"] = 360.0 + ds_livneh["lon"]

    ds_gmet_mean = ds_gmet.mean(dim="member")
    return ds_livneh, ds_gmet_mean, ds_nclim


def plot_dataset_comparison(
    ds_livneh,
    ds_gmet_mean,
    ds_nclim,
    data_var,
    unit,
    figsize=(10, 9),
    cmap="viridis",
    diff_cmap="RdBu_r",
    diff_lim=20,
):
    """
    Create a 3x3 grid comparing three xarray datasets.

    Parameters:
    -----------
    ds_livneh : xarray.Dataset
        Livneh
    ds_gmet_mean : xarray.Dataset
        GMET mean
    ds_nclim : xarray.Dataset
        NClimGrid
    data_var : str
        Name of the data variable to plot
    figsize : tuple
        Figure size (width, height)
    cmap : str
        Colormap for actual values
    diff_cmap : str
        Colormap for difference plots (should be diverging)
    """

    # Extract the data variables
    data1 = ds_livneh[data_var]
    data2 = ds_gmet_mean[data_var]
    data3 = ds_nclim[data_var]

    # Dataset names for labels
    names = ["Livneh-unsplit", "GMET mean", "NClimGrid-Daily"]
    datasets = [data1, data2, data3]

    # Create figure and subplots
    fig, axes = plt.subplots(
        3,
        3,
        figsize=figsize,
        layout="constrained",
        subplot_kw=dict(projection=ccrs.LambertConformal()),
    )

    # Find global min/max for consistent scaling of actual values
    global_min = np.round(np.min([data.quantile(0.05).values for data in datasets]))
    global_max = np.round(np.max([data.quantile(0.95).values for data in datasets]))

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]

            if i == j:
                # Diagonal: plot actual values
                im_main = datasets[i].plot(
                    ax=ax,
                    cmap=cmap,
                    vmin=global_min,
                    vmax=global_max,
                    transform=ccrs.PlateCarree(),
                    levels=11,
                    add_colorbar=False,
                )
                ax.set_title(f"{names[i]}", fontweight="bold")
                ax.coastlines()
                gl = ax.gridlines(
                    draw_labels=False, x_inline=False, rotate_labels=False, alpha=0.2
                )
                ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
                ax.set_extent([-120, -73, 22, 53], ccrs.Geodetic())

            elif i > j:
                # Below diagonal: plot percentage differences
                diff = (datasets[i] - datasets[j]) / datasets[j] * 100

                im_diff = diff.plot(
                    ax=ax,
                    cmap=diff_cmap,
                    vmin=-diff_lim,
                    vmax=diff_lim,
                    transform=ccrs.PlateCarree(),
                    add_colorbar=False,
                    levels=11,
                )
                ax.set_title(f"({names[i]} - {names[j]}) / {names[j]}", fontsize=10)
                ax.coastlines()
                gl = ax.gridlines(
                    draw_labels=False, x_inline=False, rotate_labels=False, alpha=0.2
                )
                ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
                ax.set_extent([-120, -73, 22, 53], ccrs.Geodetic())

            else:
                # Above diagonal: hide these plots
                ax.axis("off")

            # Clean up axis labels for cleaner look
            if i != 2:  # Not bottom row
                ax.set_xlabel("")
            if j != 0:  # Not leftmost column
                ax.set_ylabel("")

    # Add colorbars
    fig.colorbar(
        im_main,
        label=f"100-year return level [{unit}]",
        ax=axes[-1, :],
        pad=0.05,
        shrink=0.3,
        location="bottom",
    )

    fig.colorbar(
        im_diff,
        label="Difference [%]",
        ax=axes[-1, :],
        pad=0.05,
        shrink=0.3,
        location="bottom",
    )

    return fig, axes
