import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

import sa_city_utils as sacu
from utils import (
    gard_gcms,
    gev_metric_ids,
)
from utils import (
    roar_code_path as project_code_path,
)
from utils import (
    roar_data_path as project_data_path,
)

# Load color maps
cm_data = np.loadtxt(f"{project_code_path}/data/bamako.txt")[::-1]
bamako_map = LinearSegmentedColormap.from_list("bamako", cm_data)
uc_cmap = bamako_map

cm_data = np.loadtxt(f"{project_code_path}/data/hawaii.txt")[::-1]
hawaii_map = LinearSegmentedColormap.from_list("hawaii", cm_data)

cm_data = np.loadtxt(f"{project_code_path}/data/imola.txt")
imola_map = LinearSegmentedColormap.from_list("imola", cm_data)

cm_data = np.loadtxt(f"{project_code_path}/data/lajolla.txt")[::-1]
n_colors = len(cm_data)
end_idx = int(0.8 * n_colors)
lajolla_map = LinearSegmentedColormap.from_list("lajolla", cm_data[:end_idx])

cm_data = np.loadtxt(f"{project_code_path}/data/devon.txt")[::-1]
n_colors = len(cm_data)
start_idx = int(0.3 * n_colors)
devon_map = LinearSegmentedColormap.from_list("devon", cm_data[start_idx:])

# Truncated for total uncertainty
cmap_data = plt.get_cmap("RdPu")
colors = cmap_data(np.linspace(0.2, 1.0, 256))
RdPu_truncated = LinearSegmentedColormap.from_list("truncated_viridis", colors)
total_uc_cmap = RdPu_truncated

ssp_colors = {
    "ssp245": "#1b9e77",
    "ssp370": "#7570b3",
    "ssp585": "#d95f02",
}
ssp_labels = {
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
gev_labels = {
    "max_tasmax": "[°C]",
    "max_cdd": "[degree days]",
    "max_hdd": "[degree days]",
    "max_pr": "[mm]",
    "min_tasmin": "[°C]",
}
rel_labels = {
    "avg_tas": "[]",
    "avg_tasmin": "[]",
    "avg_tasmax": "[]",
    "sum_pr": "[]",
    "sum_cdd": "[]",
    "sum_hdd": "[]",
    "max_tasmax": "[]",
    "max_cdd": "[]",
    "max_hdd": "[]",
    "max_pr": "[]",
    "min_tasmin": "[]",
}
norm_labels = {
    "uc_99w": r"99% range",
    "uc_99w_main": r"99% range",
    "uc_95w_main": r"95% range",
    "uc_range_main": r"Total range",
}
trend_labels_abs = {
    "avg_tas": "[°C/decade]",
    "avg_tasmin": "[°C/decade]",
    "avg_tasmax": "[°C/decade]",
    "sum_pr": "[mm/decade]",
    "sum_cdd": "[degree days/decade]",
    "sum_hdd": "[degree days/decade]",
}
trend_labels_rel = {
    "avg_tas": "[%/decade]",
    "avg_tasmin": "[%/decade]",
    "avg_tasmax": "[%/decade]",
    "sum_pr": "[%/decade]",
    "sum_cdd": "[%/decade]",
    "sum_hdd": "[%/decade]",
}
avg_labels = {
    "avg_tas": "[°C]",
    "avg_tasmin": "[°C]",
    "avg_tasmax": "[°C]",
    "sum_pr": "[mm]",
    "sum_cdd": "[degree days]",
    "sum_hdd": "[degree days]",
}
title_labels = {
    "max_tasmax": "Annual maximum temperature",
    "max_cdd": "Annual maximum 1-day CDD",
    "max_hdd": "Annual maximum 1-day HDD",
    "max_pr": "Annual maximum 1-day precipitation",
    "min_tasmin": "Annual minimum temperature",
    "avg_tas": "Annual average temperature",
    "avg_tasmin": "Annual average daily minimum temperature",
    "avg_tasmax": "Annual average daily maximum temperature",
    "sum_pr": "Annual total precipitation",
    "sum_cdd": "Annual total cooling degree days",
    "sum_hdd": "Annual total heating degree days",
}
city_names = {
    "chicago": "Chicago",
    "seattle": "Seattle",
    "houston": "Houston",
    "denver": "Denver",
    "nyc": "New York",
    "sanfrancisco": "San Francisco",
    "boston": "Boston",
    "nashville": "Nashville",
}
uc_labels = {
    "ssp_uc": "Scenario uncertainty",
    "gcm_uc": "Response uncertainty",
    "iv_uc": "Internal variability",
    "dsc_uc": "Downscaling uncertainty",
    "fit_uc": "Fit uncertainty",
}
uc_colors = {
    "ssp_uc": "#0077BB",
    "gcm_uc": "#33BBEE",
    "iv_uc": "#009988",
    "dsc_uc": "#EE3377",
    "fit_uc": "#CC3311",
}
uc_markers = {
    "ssp_uc": "v",
    "gcm_uc": "^",
    "iv_uc": "o",
    "dsc_uc": "D",
    "fit_uc": "s",
}
quantile_labels = {
    "mean": "Mean",
    "q01": "1st percentile",
    "q99": "99th percentile",
}

subfigure_labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)"]
subplot_labels = ["i)", "ii)", "iii)", "iv)", "v)", "vi)"]


def tidy_ax(ax):
    ax.coastlines(linewidth=0.5)
    gl = ax.gridlines(draw_labels=False, x_inline=False, rotate_labels=False, alpha=0.2)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.25)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
    ax.set_extent([-120, -73, 22, 51], ccrs.Geodetic())


def tidy_ax_conus(ax):
    shapename = "admin_1_states_provinces_lakes"
    states_shp = shpreader.natural_earth(
        resolution="110m", category="cultural", name=shapename
    )

    # Read and filter for only US states
    for state in shpreader.Reader(states_shp).records():
        if state.attributes["admin"] == "United States of America":
            ax.add_geometries(
                [state.geometry],
                ccrs.PlateCarree(),
                facecolor="none",
                edgecolor="black",
                linewidth=0.5,
            )
    ax.set_extent([-120, -73, 22, 51], ccrs.Geodetic())


def get_vmin_vmax(da, metric_id, decimal_places=1, chfc=False):
    """
    Calculate vmin and vmax for colorbar with nicely formatted tick labels.
    Parameters
    ----------
    da : xarray.DataArray
        Data array to calculate limits for
    metric_id : str
        Metric identifier
    decimal_places : int, optional
        Number of decimal places for rounding
    cmap : str, optional
        Colormap name (default: 'RdBu')
    """

    # Determine rounding precision
    if metric_id in ["sum_pr", "max_pr"]:
        decimal_places_out = 0
    else:
        decimal_places_out = 1

    # Overwrite for change factor
    if chfc:
        decimal_places_out = 2

    if metric_id in ["sum_pr", "max_pr"]:
        cmap = "BrBG"
    else:
        cmap = "RdBu_r"

    vmin = np.round(da.quantile(0.01).to_numpy(), decimals=decimal_places)
    vmax = np.round(da.quantile(0.99).to_numpy(), decimals=decimal_places)

    if vmin < 0 and vmax > 0:
        vmin = -vmax
    else:
        cmap = None

    return (
        np.round(vmin, decimals=decimal_places_out),
        np.round(vmax, decimals=decimal_places_out),
        cmap,
    )


##################
# Map plotting
##################
def plot_uc_map(
    metric_id,
    proj_slice,
    hist_slice,
    plot_col,
    return_period,
    grid,
    fit_method,
    stationary,
    stat_str,
    time_str,
    analysis_type,
    plot_fit_uc=False,
    regrid_method="nearest",
    fig=None,
    axs=None,
    norm="uc_99w_main",
    total_uc_col="uc_99w_main",
    plot_total_uc=True,
    rel_metric_ids=[],
    cbar=False,
    vmax_uc=40,
    title="",
    y_title=0.98,
    x_title=0.05,
    fs=8,
    filter_str="",
):
    """
    Plot uncertainty component maps for climate metrics across CONUS.

    This function creates a multi-panel map visualization showing total uncertainty
    and individual uncertainty components (scenario, response, internal variability,
    downscaling, and fit uncertainty) for climate metrics. The maps are displayed
    on a Lambert Conformal projection covering the continental United States.

    Parameters
    ----------
    metric_id : str
        Climate metric identifier (e.g., 'max_tasmax', 'sum_pr', 'avg_tas').
        Must be one of the supported metrics in the title_labels dictionary.

    proj_slice : str
        Projection time period (e.g., '2050-2100').

    hist_slice : str
        Historical time period (e.g., '1950-2014').

    plot_col : str
        Column name for plotting (typically SSP scenario like 'ssp245').

    return_period : int
        Return period in years for extreme value analysis (e.g., 100).
        Only used when analysis_type is 'extreme_value'.

    grid : str
        Grid resolution identifier (e.g., '0.25').

    fit_method : str
        Fitting method used for extreme value analysis (e.g., 'mle').
        Only used when analysis_type is 'extreme_value'.

    stationary : bool
        Whether the analysis is stationary (True) or non-stationary (False).
        Only used when analysis_type is 'extreme_value'.

    time_str : str or None
        Time string for non-stationary analysis (e.g., '2050-2100').
        Only used when analysis_type is 'extreme_value'.

    analysis_type : str
        Type of analysis: 'trends', 'extreme_value', or 'averages'.
        Determines file path construction and plotting behavior.

    plot_fit_uc : bool, default=False
        Whether to include fit uncertainty in the plot.

    regrid_method : str, default="nearest"
        Regridding method used for data processing.

    fig : matplotlib.figure.Figure, optional
        Existing figure to plot on. If None, creates a new figure.

    axs : array-like of matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, creates new axes.

    norm : str or None, default="uc_99w_main"
        Normalization method for uncertainty components:
        - None: No normalization
        - "relative": Normalize by total uncertainty
        - "uc_99w_main": Normalize by 99% uncertainty range
        - Other column names: Normalize by specified column

    total_uc_col : str, default="uc_99w_main"
        Column name for total uncertainty display.

    rel_metric_ids : list of str, optional
        List of metric IDs that should be treated as relative (percentage).
        If metric_id is in this list, "_rel" is appended to the file path.

    cbar : bool, default=False
        Whether to add a colorbar for individual uncertainty components.

    vmax_uc : float, default=40
        Maximum value for uncertainty component colorbar.

    title : str, default=""
        Title for the figure. If empty, uses metric-specific title.

    y_title : float, default=0.98
        Y-position for the title (0-1 scale).

    filter_str : str, default=""
        Additional filter string to append to file paths.

    Returns
    -------
    matplotlib.collections.QuadMesh
        The last plotted contour object (for potential further customization).

    Notes
    -----
    - The function automatically constructs file paths based on analysis_type and parameters
    - Maps are displayed on a Lambert Conformal projection with CONUS extent
    - Coastlines, state boundaries, and grid lines are automatically added
    - Color schemes are automatically selected based on metric type (Blues for precipitation, Oranges for temperature)
    - The function handles both absolute and relative uncertainty metrics
    - Locations without complete ensemble data are masked out

    Examples
    --------
    >>> plot_uc_map(
    ...     metric_id="max_tasmax",
    ...     proj_slice="2050-2100",
    ...     hist_slice="1950-2014",
    ...     plot_col="ssp245",
    ...     return_period=100,
    ...     grid="0.25",
    ...     fit_method="mle",
    ...     stationary=False,
    ...     time_str="2050-2100",
    ...     analysis_type="extreme_value"
    ... )
    """
    # We can choose to normalize by a specific metric id
    if metric_id in rel_metric_ids:
        rel = True
        rel_str = "_rel"
    else:
        rel = False
        rel_str = ""
    # Read
    if analysis_type == "trends":
        if metric_id == "sum_pr":
            file_path = f"{project_data_path}/results/{metric_id}{rel_str}_{proj_slice}_{hist_slice}_{plot_col}_{grid}grid_{regrid_method}.nc"
        else:
            file_path = f"{project_data_path}/results/{metric_id}{rel_str}_{proj_slice}_{hist_slice}_{plot_col}_{grid}grid_{regrid_method}.nc"
    elif analysis_type == "extreme_value":
        if stationary:
            if time_str is not None:
                file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
            else:
                file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
        else:
            file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
    elif analysis_type == "averages":
        file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{grid}grid_{regrid_method}{filter_str}.nc"

    uc = xr.open_dataset(file_path)

    # Mask out locations without all three ensembles
    mask = uc.to_array().sum(dim="variable", skipna=False) >= 0.0
    uc = uc.where(mask, drop=True)

    # Normalize
    if norm is None:
        pass
    elif norm == "relative":
        uc_tot = uc["ssp_uc"] + uc["gcm_uc"] + uc["iv_uc"] + uc["dsc_uc"] + uc["fit_uc"]
        uc["ssp_uc"] = uc["ssp_uc"] / uc_tot
        uc["gcm_uc"] = uc["gcm_uc"] / uc_tot
        uc["iv_uc"] = uc["iv_uc"] / uc_tot
        uc["dsc_uc"] = uc["dsc_uc"] / uc_tot
        uc["fit_uc"] = uc["fit_uc"] / uc_tot
    else:
        uc["ssp_uc"] = uc["ssp_uc"] / uc[norm]
        uc["gcm_uc"] = uc["gcm_uc"] / uc[norm]
        uc["iv_uc"] = uc["iv_uc"] / uc[norm]
        uc["dsc_uc"] = uc["dsc_uc"] / uc[norm]
        uc["fit_uc"] = uc["fit_uc"] / uc[norm]

    if axs is None:
        ncols = 4 if analysis_type == "averages" else 6
        width = 10 if analysis_type == "averages" else 14
        fig, axs = plt.subplots(
            1,
            ncols,
            figsize=(width, 3),
            layout="constrained",
            subplot_kw=dict(projection=ccrs.LambertConformal()),
        )
    if plot_total_uc:
        ncols += 1

    # Plot details
    if analysis_type == "trends":
        if rel:
            unit_labels = trend_labels_rel
            uc[total_uc_col] = uc[total_uc_col] * 10 * 100  # decadal, pct trends
        else:
            unit_labels = trend_labels_abs
            uc[total_uc_col] = uc[total_uc_col] * 10  # decadal, abs trends
    elif analysis_type == "extreme_value":
        if "chfc" in time_str:
            unit_labels = rel_labels
            uc[total_uc_col] = uc[total_uc_col] * 100  # pct changes
        else:
            unit_labels = gev_labels
    else:
        if rel:
            unit_labels = rel_labels
        else:
            unit_labels = avg_labels

    # Get vmin, vmax to format nicely for 11 levels
    nlevels = 10
    if (
        not rel
        and analysis_type == "trends"
        and metric_id in ["avg_tas", "avg_tasmin", "avg_tasmax"]
    ):  # values are much smaller here
        vmin = np.round(uc[total_uc_col].min().to_numpy(), decimals=1)
        vmax = np.round(uc[total_uc_col].max().to_numpy(), decimals=1)
    else:
        vmin = np.round(uc[total_uc_col].min().to_numpy(), decimals=0)
        raw_range = uc[total_uc_col].quantile(0.95).to_numpy() - vmin
        step_size = raw_range / nlevels
        step_size = np.ceil(step_size * 2) / 2  # Round up to nearest 0.5
        vmax = vmin + (step_size * nlevels)  # 10 steps total

    # UC scale factor: change to pct if needed
    if rel:
        if analysis_type == "trends" and norm is None:
            scale_factor = 100.0 * 10.0  # decadal, pct trends
        else:
            scale_factor = 100.0  # pct changes
    elif norm is not None:
        scale_factor = 100.0
    else:
        scale_factor = 1.0

    # First plot total uncertainty
    if plot_total_uc:
        ax = axs[0]
        p = uc[total_uc_col].plot(
            ax=ax,
            levels=nlevels + 1,
            add_colorbar=True,
            vmin=vmin,
            vmax=vmax,
            cmap=total_uc_cmap,
            transform=ccrs.PlateCarree(),
            cbar_kwargs={
                "orientation": "vertical",
                "location": "left",
                "shrink": 0.6,
                "aspect": 10,
                "label": f"{norm_labels[total_uc_col]} {unit_labels[metric_id]}",
            },
        )
        # Tidy
        tidy_ax_conus(ax)
        ax.set_title("Total uncertainty", fontsize=12)
        axi_start = 1
    else:
        axi_start = 0

    # Loop through uncertainties
    for axi, uc_type in enumerate(list(uc_labels.keys())):
        if not plot_fit_uc and uc_type == "fit_uc":
            continue
        ax = axs[axi + axi_start]
        p = (scale_factor * uc[uc_type]).plot(
            ax=ax,
            levels=nlevels + 1,
            add_colorbar=False,
            vmin=0.0,
            vmax=vmax_uc,
            cmap=uc_cmap,
            transform=ccrs.PlateCarree(),
        )

        # Tidy
        tidy_ax_conus(ax)
        ax.set_extent([-120, -73, 22, 51], ccrs.Geodetic())
        ax.set_title(uc_labels[uc_type], fontsize=12)
        # Add spatial average
        avg = scale_factor * uc[uc_type].mean(dim=["lat", "lon"], skipna=True).item()
        ax.text(
            0.125,
            0.1,
            f"{avg:.1f}%",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fs,
            bbox=dict(boxstyle="round", fc="silver", alpha=0.7),
        )

    # Cbar
    if cbar:
        if norm is None:
            cbar_label = f"Absolute uncertainty {unit_labels[metric_id]}"
        elif norm == "relative":
            cbar_label = "Relative uncertainty [%]"
        else:
            cbar_label = "Fraction of total uncertainty [%]"

        fig.colorbar(
            p,
            orientation="horizontal",
            label=cbar_label,
            ax=axs[axi_start:],
            pad=0.05,
            shrink=0.3,
        )

    if title is not None:
        if title in [
            "",
            "a)",
            "b)",
            "c)",
            "d)",
            "e)",
            "i)",
            "ii)",
            "iii)",
            "iv)",
            "v)",
            "vi)",
        ]:
            fig.suptitle(
                f"{title} {title_labels[metric_id]}",
                style="italic",
                y=y_title,
                x=x_title,
                ha="left",
            )
        else:
            fig.suptitle(title, style="italic", y=y_title, x=x_title, ha="left")

    return p


def plot_ensemble_mean_uncertainty(
    plot_metric_ids,
    proj_slice,
    hist_slice,
    plot_col,
    return_period,
    grid,
    fit_method,
    stationary,
    stat_str,
    time_str,
    analysis_type,
    quantile="mean",
    total_uc_col="uc_99w_main",
    regrid_method="nearest",
    narrow_subfigs=False,
    fig=None,
    rel_metric_ids=[],
    y_title=1.08,
    filter_str="",
):
    # Get mask
    mask = xr.open_dataset(f"{project_data_path}/mask.nc")["mask"]

    # Set up figure
    if fig is None:
        fig = plt.figure(
            figsize=(len(plot_metric_ids) * 4, 2.5),
            layout="constrained",
        )
    # Set up subfigs
    if narrow_subfigs:
        subfigs = fig.subfigures(1, 3, width_ratios=[0.4, 1, 0.4])
        idm_start = 1
    else:
        subfigs = fig.subfigures(1, len(plot_metric_ids))
        idm_start = 0
        if len(plot_metric_ids) == 1:
            subfigs = [subfigs]
    # Loop through metrics
    for idm, metric_id in enumerate(plot_metric_ids):
        axs = subfigs[idm + idm_start].subplots(
            1,
            2,
            subplot_kw=dict(projection=ccrs.LambertConformal()),
        )
        # We can choose to normalize by a specific metric id
        if metric_id in rel_metric_ids:
            rel = True
            rel_str = "_rel"
        else:
            rel = False
            rel_str = ""
        # Read
        if analysis_type == "trends":
            if metric_id == "sum_pr":
                mean_file_path = f"{project_data_path}/results/summary_{metric_id}{rel_str}_{proj_slice}_{hist_slice}_{plot_col}_{grid}grid_{regrid_method}.nc"
                uc_file_path = f"{project_data_path}/results/{metric_id}{rel_str}_{proj_slice}_{hist_slice}_{plot_col}_{grid}grid_{regrid_method}.nc"
            else:
                mean_file_path = f"{project_data_path}/results/summary_{metric_id}{rel_str}_{proj_slice}_{hist_slice}_{plot_col}_{grid}grid_{regrid_method}.nc"
                uc_file_path = f"{project_data_path}/results/{metric_id}{rel_str}_{proj_slice}_{hist_slice}_{plot_col}_{grid}grid_{regrid_method}.nc"
        elif analysis_type == "extreme_value":
            if stationary:
                if time_str is not None:
                    mean_file_path = f"{project_data_path}/results/summary_{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
                    uc_file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
                else:
                    mean_file_path = f"{project_data_path}/results/summary_{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
                    uc_file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
            else:
                mean_file_path = f"{project_data_path}/results/summary_{metric_id}_{proj_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
                uc_file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
        elif analysis_type == "averages":
            mean_file_path = f"{project_data_path}/results/summary_{metric_id}_{proj_slice}_{hist_slice}_{grid}grid_{regrid_method}{filter_str}.nc"
            uc_file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{grid}grid_{regrid_method}{filter_str}.nc"

        ds_mean = xr.open_dataset(mean_file_path)
        ds_uc = xr.open_dataset(uc_file_path)

        # Plot details
        if analysis_type == "trends":
            if rel:
                unit_labels = trend_labels_rel
                ds_mean[plot_col] = ds_mean[plot_col] * 10 * 100  # decadal, pct trends
                ds_uc[total_uc_col] = (
                    ds_uc[total_uc_col] * 10 * 100
                )  # decadal, pct trends
            else:
                unit_labels = trend_labels_abs
                ds_mean[plot_col] = ds_mean[plot_col] * 10  # decadal, abs trends
                ds_uc[total_uc_col] = ds_uc[total_uc_col] * 10  # decadal, abs trends
        elif analysis_type == "extreme_value":
            if "chfc" in time_str:
                unit_labels = rel_labels
                ds_mean[plot_col] = ds_mean[plot_col]
                ds_uc[total_uc_col] = ds_uc[total_uc_col]
            else:
                unit_labels = gev_labels
        else:
            if rel:
                unit_labels = rel_labels
            else:
                unit_labels = avg_labels

        # Plot mean
        da = ds_mean.sel(quantile=quantile).mean(dim=["ensemble", "ssp"])[plot_col]
        da = da.where(mask, drop=True)
        if analysis_type == "extreme_value" and "chfc" in time_str:
            vmin, vmax, cmap = get_vmin_vmax(da, metric_id, decimal_places=2, chfc=True)
        else:
            vmin, vmax, cmap = get_vmin_vmax(da, metric_id)
        # cmap
        if cmap is None:
            if metric_id in ["max_pr", "sum_pr"]:
                cmap = devon_map
            else:
                cmap = "Blues_r" if "min" in metric_id else lajolla_map

        # Cbar label
        if rel and analysis_type == "trends":
            cbar_label = "Trend [%/decade]"
        elif not rel and analysis_type == "trends":
            cbar_label = f"Trend {unit_labels[metric_id]}"

        if analysis_type == "extreme_value":
            if "chfc" in time_str:
                cbar_label = "Change factor []"
            elif "diff" in time_str:
                cbar_label = f"Change {unit_labels[metric_id]}"
            else:
                cbar_label = f"Level {unit_labels[metric_id]}"

        ax = axs[0]
        p = da.where(da != 0.0).plot(  # assume 0.0 is missing value
            ax=ax,
            add_colorbar=True,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            cbar_kwargs={
                "orientation": "horizontal",
                "location": "bottom",
                "label": cbar_label,
                "pad": 0.05,
                "shrink": 0.9,
                "ticks": np.linspace(vmin, vmax, num=3),
            },
        )
        # Tidy
        tidy_ax_conus(ax)
        ax.set_title("")

        ## Plot total uncertainty
        # Mask out locations without all three ensembles
        ds_uc = ds_uc.where(mask, drop=True)
        da = ds_uc[total_uc_col]
        ax = axs[1]
        if analysis_type == "extreme_value" and "chfc" in time_str:
            vmin, vmax, cmap = get_vmin_vmax(da, metric_id, decimal_places=2, chfc=True)
        else:
            vmin, vmax, cmap = get_vmin_vmax(da, metric_id)
        p = da.plot(
            ax=ax,
            add_colorbar=True,
            vmin=vmin,
            vmax=vmax,
            cmap=total_uc_cmap,
            transform=ccrs.PlateCarree(),
            cbar_kwargs={
                "orientation": "horizontal",
                "location": "bottom",
                "label": f"{norm_labels[total_uc_col]} {unit_labels[metric_id]}",
                "pad": 0.05,
                "shrink": 0.9,
                "ticks": np.linspace(vmin, vmax, num=3),
            },
        )
        # Tidy
        tidy_ax_conus(ax)
        ax.set_title("")

        # Title
        subfigs[idm + idm_start].suptitle(
            f"{subplot_labels[idm]} {title_labels[metric_id]}",
            style="italic",
            y=y_title,
        )

    return p


def plot_ensemble_ssp_means(
    metric_id,
    proj_slice,
    hist_slice,
    plot_col,
    return_period,
    fit_method,
    stationary,
    stat_str,
    time_str,
    analysis_type,
    grid="LOCA2",
    regrid_method="nearest",
    fig=None,
    axs=None,
    rel_metric_ids=[],
    title="",
    y_title=1.05,
    filter_str="",
    store_path=None,
):
    # We can choose to normalize by a specific metric id
    if metric_id in rel_metric_ids:
        rel = True
        rel_str = "_rel"
    else:
        rel = False
        rel_str = ""
    # Read
    if analysis_type == "trends":
        if metric_id == "sum_pr":
            file_path = f"{project_data_path}/results/summary_{metric_id}{rel_str}_{proj_slice}_{hist_slice}_{plot_col}_{grid}grid_{regrid_method}.nc"
        else:
            file_path = f"{project_data_path}/results/summary_{metric_id}{rel_str}_{proj_slice}_{hist_slice}_{plot_col}_{grid}grid_{regrid_method}.nc"
    elif analysis_type == "extreme_value":
        if stationary:
            if time_str is not None:
                file_path = f"{project_data_path}/results/summary_{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
            else:
                file_path = f"{project_data_path}/results/summary_{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
        else:
            file_path = f"{project_data_path}/results/summary_{metric_id}_{proj_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}{filter_str}.nc"
    elif analysis_type == "averages":
        file_path = f"{project_data_path}/results/summary_{metric_id}_{proj_slice}_{hist_slice}_{grid}grid_{regrid_method}{filter_str}.nc"

    ds = xr.open_dataset(file_path)

    if axs is None:
        fig, axs = plt.subplots(
            6,
            3,
            figsize=(8, 12),
            layout="constrained",
            subplot_kw=dict(projection=ccrs.LambertConformal()),
        )

    # Plot details
    if analysis_type == "trends":
        if rel:
            unit_labels = trend_labels_rel
            ds[plot_col] = ds[plot_col] * 10 * 100  # decadal, pct trends
        else:
            unit_labels = trend_labels_abs
            ds[plot_col] = ds[plot_col] * 10  # decadal, abs trends
    elif analysis_type == "extreme_value":
        if "chfc" in time_str:
            unit_labels = rel_labels
            ds[plot_col] = ds[plot_col]
        else:
            unit_labels = gev_labels
    else:
        if rel:
            unit_labels = rel_labels
        else:
            unit_labels = avg_labels

    # Get vmin, vmax to format nicely for 11 levels
    if "chfc" in time_str:
        vmin, vmax, cmap = get_vmin_vmax(ds[plot_col], metric_id, decimal_places=3)
    else:
        vmin, vmax, cmap = get_vmin_vmax(ds[plot_col], metric_id, decimal_places=2)

    # cmap
    if cmap is None:
        if metric_id in ["max_pr", "sum_pr"]:
            cmap = devon_map
        else:
            cmap = lajolla_map if "max" in metric_id else imola_map

    # Loop through quantiles
    for idq, quantile in enumerate(["q01", "mean", "q99"]):
        # Loop through combos
        combos = [
            ("LOCA2", "ssp245"),
            ("STAR-ESDM", "ssp245"),
            ("LOCA2", "ssp370"),
            ("GARD-LENS", "ssp370"),
            ("LOCA2", "ssp585"),
            ("STAR-ESDM", "ssp585"),
        ]
        for idx, combo in enumerate(combos):
            ensemble, ssp = combo
            da = ds.sel(quantile=quantile, ensemble=ensemble, ssp=ssp)[plot_col]
            # Plot
            ax = axs[idx, idq]
            p = da.where(da != 0.0).plot(  # assume 0.0 is missing value
                ax=ax,
                add_colorbar=False,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
            )
            # Tidy
            tidy_ax(ax)
            ax.set_title(
                f"{ensemble} {ssp_labels[ssp]} {quantile_labels[quantile]}", fontsize=12
            )
            # Add fraction > 0
            if "diff" in time_str:
                frac_gt0 = 100 * (da > 0.0).sum() / (da.notnull().sum())
                ax.text(
                    0.65,
                    0.08,
                    f"{frac_gt0:.1f}%",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                    bbox=dict(boxstyle="round", fc="silver", alpha=0.7),
                )

    # Cbar
    if rel and analysis_type == "trends":
        cbar_label = "Trend \n[%/decade]"
    elif rel and analysis_type == "extreme_value":
        cbar_label = "Change [%]"
    elif not rel and analysis_type == "extreme_value":
        if "diff" in time_str:
            cbar_label = f"Change {unit_labels[metric_id]}"
        else:
            cbar_label = unit_labels[metric_id]
    elif not rel and analysis_type == "trends":
        cbar_label = f"Trend {unit_labels[metric_id]}"

    if vmin < 0:
        ticks = (
            np.linspace(vmin, 0.0, num=3).tolist()
            + np.linspace(0.0, vmax, num=3).tolist()[1:]
        )
    else:
        ticks = np.linspace(vmin, vmax, num=3)
    fig.colorbar(
        p,
        orientation="horizontal",
        label=cbar_label,
        ax=axs,
        location="bottom",
        shrink=0.6,
        pad=0.02,
        ticks=ticks,
    )

    if title is not None:
        if title in [
            "",
            "a)",
            "b)",
            "c)",
            "d)",
            "e)",
            "i)",
            "ii)",
            "iii)",
            "iv)",
            "v)",
            "vi)",
        ]:
            fig.suptitle(
                f"{title} {title_labels[metric_id]}",
                style="italic",
                y=y_title,
                x=0.05,
                ha="left",
            )
        else:
            fig.suptitle(title, style="italic", y=y_title, x=0.05, ha="left")

    if store_path is not None:
        fig.savefig(store_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def plot_ensemble_mean_uq(
    plot_metric_ids,
    plot_col,
    analysis_type,
    summary_title,
    uncertainty_title="b) Uncertainty decomposition",
    proj_slice="1950-2100",
    hist_slice=None,
    return_period=100,
    fit_method="mle",
    stationary=False,
    stat_str="nonstat_scale",
    time_str=None,
    rel_metric_ids=[],
    grid="LOCA2",
    norm="uc_99w_main",
    vmax_uc=50,
    plot_fit_uc=True,
    figsize=(12, 7.5),
    height_ratios=[1.15, 2],
    narrow_subfigs=False,
    a_y_title=1.2,
    a_y_titles=1.1,
    b_y_title=1.1,
    b_y_titles=1.08,
    x_title=0.05,
    hspace=0.2,
    fs=8,
    cbar_ax=[0.4, 0.01, 0.2, 0.025],
    store_path=None,
):
    fig = plt.figure(figsize=figsize, layout="constrained")
    subfigs = fig.subfigures(2, 1, hspace=hspace, height_ratios=height_ratios)

    ################# a) Summary plot
    subfigs[0].suptitle(summary_title, fontweight="bold", y=a_y_title)

    plot_ensemble_mean_uncertainty(
        plot_metric_ids=plot_metric_ids,
        proj_slice=proj_slice,
        hist_slice=hist_slice,
        plot_col=plot_col,
        return_period=return_period,
        grid=grid,
        fit_method=fit_method,
        stationary=stationary,
        stat_str=stat_str,
        time_str=time_str,
        rel_metric_ids=rel_metric_ids,
        analysis_type=analysis_type,
        fig=subfigs[0],
        y_title=a_y_titles,
        narrow_subfigs=narrow_subfigs,
    )

    ############# b) Uncertainty decomposition
    b_subfigs = subfigs[1].subfigures(len(plot_metric_ids), 1, hspace=0.01)
    # Handle single metric case - wrap in list for consistent indexing
    if len(plot_metric_ids) == 1:
        b_subfigs = [b_subfigs]

    subfigs[1].suptitle(uncertainty_title, fontweight="bold", y=b_y_title)
    for idp, metric_id in enumerate(plot_metric_ids):
        if len(plot_metric_ids) == 3:
            axs = b_subfigs[idp].subplots(
                1,
                7,
                subplot_kw=dict(projection=ccrs.LambertConformal()),
                width_ratios=[0.65, 1, 1, 1, 1, 1, 0.65],
            )
        else:
            axs = b_subfigs[idp].subplots(
                1,
                5 if plot_fit_uc else 4,
                subplot_kw=dict(projection=ccrs.LambertConformal()),
            )
        p = plot_uc_map(
            metric_id=metric_id,
            proj_slice=proj_slice,
            hist_slice=hist_slice,
            plot_col=plot_col,
            return_period=return_period,
            grid=grid,
            fit_method=fit_method,
            stationary=stationary,
            stat_str=stat_str,
            time_str=time_str,
            rel_metric_ids=rel_metric_ids,
            norm=norm,
            analysis_type=analysis_type,
            vmax_uc=vmax_uc,
            y_title=b_y_titles,
            title=subplot_labels[idp],
            fig=b_subfigs[idp],
            axs=axs[1:-1] if len(plot_metric_ids) == 3 else axs,
            plot_fit_uc=plot_fit_uc,
            plot_total_uc=False,
            x_title=x_title,
            fs=fs,
        )
        if len(plot_metric_ids) == 3:
            # Remove first and last axes (for spacing)
            axs[0].remove()
            axs[-1].remove()

    # Create a new axes for the colorbar at the bottom
    cbar_ax = subfigs[1].add_axes(cbar_ax)  # [left, bottom, width, height]
    cbar = subfigs[1].colorbar(p, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Fraction of total uncertainty [%]")

    if store_path is not None:
        fig.savefig(store_path, dpi=400, bbox_inches="tight")
    else:
        plt.show()


# Supplementary figure
def plot_ensemble_ssp_means_uncertainty(
    plot_metric_ids,
    plot_col,
    analysis_type,
    summary_title,
    proj_slice="1950-2100",
    hist_slice=None,
    return_period=100,
    fit_method="mle",
    stationary=False,
    stat_str="nonstat_scale",
    time_str=None,
    rel_metric_ids=[],
    grid="LOCA2",
    norm="uc_99w_main",
    vmax_uc=50,
    y_title=1.08,
    figsize=(12, 6),
    store_path=None,
):
    """
    Plot summary statistics and uncertainty decomposition for given metrics.
    """
    if analysis_type == "trends":
        time_str = None

    fig = plt.figure(figsize=figsize, layout="constrained")
    subfigs = fig.subfigures(len(plot_metric_ids), 1, hspace=0.01)
    fig.suptitle(summary_title, fontweight="bold", y=1.09)

    for idp, metric_id in enumerate(plot_metric_ids):
        axs = subfigs[idp].subplots(
            1, 6, subplot_kw=dict(projection=ccrs.LambertConformal())
        )
        plot_ensemble_ssp_means(
            metric_id=metric_id,
            proj_slice=proj_slice,
            hist_slice=hist_slice,
            plot_col=plot_col,
            return_period=return_period,
            grid=grid,
            fit_method=fit_method,
            stationary=stationary,
            stat_str=stat_str,
            time_str=time_str,
            rel_metric_ids=rel_metric_ids,
            analysis_type=analysis_type,
            fig=subfigs[idp],
            axs=axs,
            title=subplot_labels[idp],
            y_title=y_title,
        )

    if store_path is not None:
        fig.savefig(store_path, dpi=400, bbox_inches="tight")
    else:
        plt.show()


#######################
# City plot
#######################
def plot_jagged_scatter(
    df, plot_col, position, color, ax, jitter_amount=0.1, limits=None, s=20, alpha=0.8
):
    # Filter data below limits if desired
    if limits is not None:
        data = df[(df[plot_col] < limits[1]) & (df[plot_col] > limits[0])]
    else:
        data = df.copy()

    # Take only the central values
    if "n_boot" in data.columns:
        data = data[data["n_boot"] == "main"]
    elif "quantile" in data.columns:
        data = data[data["quantile"] == "main"]

    # Random offsets for y-axis
    y_offsets = np.clip(
        np.random.normal(loc=0.0, scale=jitter_amount, size=len(data)), -0.4, 0.4
    )
    y_values = [position + offset for offset in y_offsets]

    # Create jagged scatter plot
    ax.scatter(
        x=data[plot_col],
        y=y_values,
        c="white",
        # c=color,
        edgecolor=color,
        s=s,
        alpha=alpha,
        zorder=5,
    )


def plot_conf_intvs(
    df, plot_col, positions, color, ax, limits=None, lw=1.5, s=20, alpha=1
):
    # Filter data below limits if desired
    if limits is not None:
        data = df[(df[plot_col] < limits[1]) & (df[plot_col] > limits[0])]
    else:
        data = df.copy()

    # Point for median
    ax.scatter(
        x=[data[data["quantile"] == "main"][plot_col].values[0]],
        y=positions,
        c=color,
        s=s,
        zorder=6,
    )

    # Line for 95% CI
    ax.plot(
        [
            data[data["quantile"] == "q025"][plot_col].values[0],
            data[data["quantile"] == "q975"][plot_col].values[0],
        ],
        [positions, positions],
        color=color,
        linewidth=lw,
        zorder=6,
        alpha=alpha,
    )


# Transform samples into quantile if needed
def transform_samples_to_quantile(df):
    """
    Transforms the raw samples into quantiles. This is the 'true' form which
    calculates quantiles across the entire sample.
    Note this only works for one GCM/SSP combination.
    """
    # Get overall quantiles
    df_quantiles = (
        df.quantile([0.025, 0.975], numeric_only=True)
        .reset_index()
        .rename(columns={"index": "quantile"})
    )
    df_quantiles["quantile"] = df_quantiles["quantile"].map(
        {0.025: "q025", 0.975: "q975"}
    )
    # Get overall mean
    df_mean = pd.DataFrame(df.mean(numeric_only=True)).T
    df_mean["quantile"] = "main"
    return pd.concat([df_mean, df_quantiles], ignore_index=True)


def aggregate_quantiles(df):
    """
    Aggregates the quantiles into a single dataframe. This is the approximate form
    which takes the upper and lower pre-computed quantiles.
    Note this only works for one GCM/SSP combination.
    """
    df_lower = pd.DataFrame(
        df[df["quantile"] == "q025"].quantile(0.005, numeric_only=True)
    ).T
    df_lower["quantile"] = "q025"
    df_upper = pd.DataFrame(
        df[df["quantile"] == "q975"].quantile(0.995, numeric_only=True)
    ).T
    df_upper["quantile"] = "q975"
    df_quantiles = pd.concat([df_lower, df_upper], ignore_index=True)

    df_main = pd.DataFrame(df[df["quantile"] == "main"].mean(numeric_only=True)).T
    df_main["quantile"] = "main"

    return pd.concat([df_main, df_quantiles], ignore_index=True)


def plot_city_results(
    city,
    metric_id,
    plot_col,
    fit_method,
    stationary,
    axs,
    hist_slice="1950-2014",
    proj_slice="2050-2100",
    tgw_hist_slice="1980-2019",
    tgw_proj_slice="2049-2099",
    years="1950-2100",  # non-stationary only
    tgw_years="1980-2099",  # non-stationary only
    read_samples=False,
    n_boot=1000,
    n_min_members=5,
    title=None,
    yticklabels=True,
    legend=True,
    limits=None,
):
    # Read results
    sample_str = "_samples" if read_samples else ""
    if stationary:
        df = pd.read_csv(
            f"{project_data_path}/extreme_value/cities/original_grid/freq/{city}_{metric_id}_{hist_slice}_{proj_slice}_{fit_method}_stat_nbootproj1000_nboothist1{sample_str}.csv"
        )
        df_tgw = pd.read_csv(
            f"{project_data_path}/extreme_value/cities/original_grid/freq/TGW_{city}_{metric_id}_{tgw_hist_slice}_{tgw_proj_slice}_{fit_method}_stat_nbootproj1000_nboothist1{sample_str}.csv"
        ).drop_duplicates()
        df_index = pd.read_csv(
            f"{project_data_path}/extreme_value/cities/original_grid/freq/{city}_max_tasmax_{hist_slice}_{proj_slice}_{fit_method}_stat_nbootproj1000_nboothist1{sample_str}.csv"
        )
    else:
        df = pd.read_csv(
            f"{project_data_path}/extreme_value/cities/original_grid/freq/{city}_{metric_id}_{years}_{fit_method}_nonstat_nboot1000{sample_str}.csv"
        )
        df_tgw = pd.read_csv(
            f"{project_data_path}/extreme_value/cities/original_grid/freq/TGW_{city}_{metric_id}_{tgw_years}_{fit_method}_nonstat_nboot1000{sample_str}.csv"
        ).drop_duplicates()
        df_index = pd.read_csv(
            f"{project_data_path}/extreme_value/cities/original_grid/freq/{city}_max_tasmax_{years}_{fit_method}_nonstat_nboot1000{sample_str}.csv"
        )
    # Update GARD GCMs
    df["gcm"] = (
        df["gcm"]
        .replace("canesm5", "CanESM5")
        .replace("cesm2", "CESM2-LENS")
        .replace("ecearth3", "EC-Earth3")
    )

    # Fix negation for min_tasmin, non-stationry models
    if metric_id == "min_tasmin" and not stationary:
        df.loc[df["n_boot"] != "main", plot_col] = -df.loc[
            df["n_boot"] != "main", plot_col
        ]

    df_uc = sacu.calculate_df_uc(df, plot_col)
    df = df.set_index(["ensemble", "gcm", "member", "ssp"])
    df_tgw = df_tgw.set_index(["ensemble", "gcm", "member", "ssp"])
    df_index = df_index.set_index(["ensemble", "gcm", "member", "ssp"])

    # Make figure if needed
    if axs is None:
        fig, axs = plt.subplots(
            2, 1, figsize=(5, 11), height_ratios=[5, 1], layout="constrained"
        )

    if title is None:
        axs[0].set_title(title_labels[metric_id])
    else:
        axs[0].set_title(title)

    # Get details
    units = gev_labels[metric_id]

    ############################
    # UC
    ############################
    ax = axs[1]

    uc_names = [
        "Scenario \n uncertainty",
        "Response \n uncertainty",
        "Internal \n variability",
        "Downscaling \n uncertainty",
        "Fit \n uncertainty",
    ]

    df_uc[df_uc["uncertainty_type"].isin(uc_labels.keys())].plot.bar(
        x="uncertainty_type", y="mean", yerr="std", ax=ax, legend=False, capsize=3
    )

    # Tidy
    ax.set_xticklabels(uc_names, rotation=45, fontsize=10)
    unit_str = "" if "chfc" in plot_col else f" {units}"
    ax.set_ylabel(f"Range{unit_str}")
    ax.set_xlabel("")
    ax.grid(alpha=0.2, zorder=3)
    ax.set_ylim([0, ax.get_ylim()[1]])

    ############################
    # Boxplots
    ############################
    ax = axs[0]
    trans = transforms.blended_transform_factory(
        ax.transAxes,  # x in axis coordinates (0 to 1)
        ax.transData,  # y in data coordinates
    )

    idy = 0
    label_names = []
    label_idy = []

    ############################
    # TGW
    ############################
    ensemble = "TGW"
    if read_samples:
        df_sel_grouped = transform_samples_to_quantile(df_tgw)
    else:
        df_sel_grouped = aggregate_quantiles(df_tgw)
    plot_conf_intvs(
        df_sel_grouped,
        plot_col,
        [idy],
        "silver",
        ax,
        s=75,
        lw=3,
        limits=limits,
    )
    plot_jagged_scatter(
        df_tgw,
        plot_col,
        [idy],
        "silver",
        ax,
        jitter_amount=0.075,
        limits=limits,
    )
    label_names.append(f"All scenarios ({df_tgw.index.nunique()})")
    label_idy.append(idy)
    idy += 1

    ax.axhline(idy - 0.5, color="black")
    ax.text(
        0.98,
        idy - 0.6,
        "TGW",
        transform=trans,
        fontstyle="italic",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5, "pad": 0.1},
        verticalalignment="top",
        horizontalalignment="right",
        zorder=10,
    )

    ############################
    # GARD-LENS
    ############################
    ensemble = "GARD-LENS"
    for gcm in gard_gcms:
        df_sel = df.loc[ensemble, gcm, :, :]
        if read_samples:
            df_sel_grouped = transform_samples_to_quantile(df_sel)
        else:
            df_sel_grouped = aggregate_quantiles(df_sel)
        plot_conf_intvs(
            df_sel_grouped,
            plot_col,
            [idy],
            ssp_colors["ssp370"],
            ax,
            s=75,
            lw=3,
            limits=limits,
        )
        plot_jagged_scatter(
            df_sel, plot_col, [idy], ssp_colors["ssp370"], ax, limits=limits
        )
        label_names.append(f"{gcm} ({df_sel.index.nunique()})")
        label_idy.append(idy)
        idy += 1

    ax.axhline(idy - 0.5, color="black")
    ax.text(
        0.98,
        idy - 0.6,
        "GARD-LENS",
        transform=trans,
        fontstyle="italic",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5, "pad": 0.1},
        verticalalignment="top",
        horizontalalignment="right",
        zorder=10,
    )

    ############################
    # STAR-ESDM
    ############################
    ensemble = "STAR-ESDM"
    for ssp in df.loc[ensemble].index.unique(level="ssp"):
        df_sel = df.loc[ensemble, :, :, ssp]
        if read_samples:
            df_sel_grouped = transform_samples_to_quantile(df_sel)
        else:
            df_sel_grouped = aggregate_quantiles(df_sel)
        plot_conf_intvs(
            df_sel_grouped,
            plot_col,
            [idy],
            ssp_colors[ssp],
            ax,
            s=75,
            lw=3,
            limits=limits,
        )
        plot_jagged_scatter(
            df_sel,
            plot_col,
            [idy],
            ssp_colors[ssp],
            ax,
            jitter_amount=0.075,
            limits=limits,
        )
        label_names.append(f"All GCMs ({df_sel.index.nunique()})")
        label_idy.append(idy)
        idy += 1

    ax.axhline(idy - 0.5, color="black")
    ax.text(
        0.98,
        idy - 0.6,
        "STAR-ESDM",
        transform=trans,
        fontstyle="italic",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5, "pad": 0.1},
        verticalalignment="top",
        horizontalalignment="right",
        zorder=10,
    )

    ############################
    # LOCA2
    ############################
    ensemble = "LOCA2"
    for gcm in df_index.loc[ensemble].index.unique(level="gcm"):
        for ssp in df_index.loc[ensemble, gcm].index.unique(level="ssp"):
            if (
                len(df_index.loc[ensemble, gcm, :, ssp].index.unique(level="member"))
                >= n_min_members
            ):
                df_sel = df.loc[ensemble, gcm, :, ssp]
                df_sel_index = df_index.loc[ensemble, gcm, :, ssp]
                if read_samples:
                    df_sel_grouped = transform_samples_to_quantile(df_sel)
                else:
                    df_sel_grouped = aggregate_quantiles(df_sel)
                plot_conf_intvs(
                    df_sel_grouped,
                    plot_col,
                    [idy],
                    ssp_colors[ssp],
                    ax,
                    s=75,
                    lw=3,
                    limits=limits,
                )
                plot_jagged_scatter(
                    df_sel,
                    plot_col,
                    [idy],
                    ssp_colors[ssp],
                    ax,
                    jitter_amount=0.05,
                    limits=limits,
                )
                label_names.append(f"{gcm} ({df_sel_index.index.nunique()})")
                label_idy.append(idy)
                idy += 0.5

    ax.text(
        0.98,
        idy - 0.1,
        "LOCA2",
        transform=trans,
        fontstyle="italic",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.5, "pad": 0.1},
        verticalalignment="top",
        horizontalalignment="right",
        zorder=10,
    )

    #### Tidy
    ax.set_ylim([-0.5, idy])
    if yticklabels:
        ax.set_yticks(label_idy, label_names, fontsize=10)
    else:
        ax.set_yticks(label_idy, ["" for _ in label_names], fontsize=10)
    ax.grid(alpha=0.75)

    # Get xlabel
    xlabel_str = (
        "Change in"
        if "diff" in plot_col
        else "Change factor:"
        if "chfc" in plot_col
        else ""
    )
    return_level_str = plot_col.split("yr")[0]
    ax.set_xlabel(f"{xlabel_str} {return_level_str}-year return level{unit_str}")

    # Legend
    if legend:
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=ssp_colors[ssp],
                markerfacecolor=ssp_colors[ssp],
                markersize=8,
                lw=3,
                label=ssp_labels[ssp],
            )
            for ssp in ssp_colors.keys()
        ]
        legend = ax.legend(handles=legend_elements)
        legend.set_zorder(10)


def plot_uc_bars(dfs, ax, labels, legend=False, colors=None):
    # Get uc names
    uc_names = [
        "Scenario \n uncertainty",
        "Response \n uncertainty",
        "Internal \n variability",
        "Downscaling \n uncertainty",
        "Fit \n uncertainty",
    ]

    # Get colors
    if colors is None:
        colors = [f"C{i}" for i in range(len(dfs))]

    n = len(dfs)

    # Normalize by uc_99w_main
    for i in range(n):
        if "uc_99w_main" not in dfs[i].index:
            dfs[i] = dfs[i].set_index("uncertainty_type")
        dfs[i] = dfs[i].apply(lambda x: x / dfs[i].loc["uc_99w_main"]["mean"])

    # Make sure only one SSP type
    for i in range(n):
        dfs[i] = dfs[i].drop(["ssp_uc", "uc_99w_main"])

    # Get bar positioning
    bar_width = 1 / (n * 1.5)
    positions = [np.arange(len(dfs[i])) + i * bar_width for i in range(len(dfs))]
    # print(positions)

    # Create the grouped bar chart
    for i, df in enumerate(dfs):
        bars = ax.bar(
            positions[i],
            df["mean"],
            width=bar_width,
            color=colors[i],
            label=labels[i],
            yerr=df["std"],
            capsize=3,
            align="center",
        )

    ax.set_xticks((positions[0] + positions[-1]) / 2)
    ax.set_xticklabels(uc_names, rotation=0)
    ax.set_yticklabels([])
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_xlabel("")
    ax.grid(alpha=0.2, zorder=3)
    if legend:
        ax.legend(loc="upper right", fontsize=10)


################################
# UC breakdown by return level
################################


def plot_uc_rls(
    coord_or_mean,
    proj_slice,
    hist_slice,
    fit_method,
    stat_str,
    grid="LOCA2",
    regrid_method="nearest",
    total_uc="uc_99w_main",
    plot_total_uc=False,
    return_periods=[10, 25, 50, 100],
    metric_ids=gev_metric_ids[:3],
    ax_title=False,
    fig_title=None,
    store_path=None,
    axs=None,
    fig=None,
    legend=True,
    idm_start=0,
    ylim=None,
    return_legend=False,
    y_title=1.05,
    time_str=None,
    plot_fit_uc=True,
    xlabel=True,
    xticklabels=True,
    fontsize=10,
):
    # Make figure
    if axs is None:
        fig, axs = plt.subplots(
            1,
            len(metric_ids),
            figsize=(3.5 * len(metric_ids), 4.5),
            sharey=True,
            gridspec_kw={"wspace": 0.1},
            layout="constrained",
        )

    # Loop through metrics
    for idm, metric_id in enumerate(metric_ids):
        # Read all return levels
        ds = []
        for return_period in return_periods:
            if stat_str == "stat":
                file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}.nc"
            else:
                file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}.nc"
            ds.append(
                xr.open_dataset(file_path).assign_coords(return_period=return_period)
            )
        ds = xr.concat(ds, dim="return_period", coords="minimal")

        # Mask out locations without all three ensembles
        mask = xr.open_dataset(f"{project_data_path}/mask.nc")["mask"]
        ds = ds.where(mask, drop=True)

        # Ready for plot
        if coord_or_mean == "mean":
            df = ds.mean(dim=["lat", "lon"]).to_dataframe().droplevel("quantile")
        else:
            df = (
                ds.sel(
                    lat=coord_or_mean[0], lon=360 + coord_or_mean[1], method="nearest"
                )
                .to_dataframe()
                .droplevel("quantile")
            )
        df_total_uc = df[total_uc]

        # Plot total UC first with lower alpha
        if plot_total_uc:
            ax1 = axs[idm].twinx()
            # ax1.plot(df.index, df_total_uc, lw=2, color="black", alpha=0.5)
            ax1.scatter(
                df.index, df_total_uc, s=50, marker="X", color="black", alpha=0.8
            )
            ax1.set_ylabel(
                f"Total uncertainty {gev_labels[metric_id]}", rotation=-90, va="bottom"
            )

        # Plot UC components on top with higher alpha
        ax = axs[idm] if isinstance(axs, list) else axs
        for uc_type in uc_labels:
            if not plot_fit_uc and uc_type == "fit_uc":
                continue
            ax.plot(
                df.index,
                df[uc_type] / df[total_uc] * 100,
                lw=2,
                color=uc_colors[uc_type],
                alpha=0.9,
            )
            ax.scatter(
                df.index,
                df[uc_type] / df[total_uc] * 100,
                s=50,
                marker=uc_markers[uc_type],
                color=uc_colors[uc_type],
                alpha=0.9,
            )

        # Tidy
        ax.grid(alpha=0.5)
        ax.set_xticks(return_periods)
        if ylim is not None:
            ax.set_ylim(ylim)
        if ax_title:
            title_str = title_labels[metric_id]
            ax.set_title(
                f"{subplot_labels[idm + idm_start]} {title_str}", fontstyle="italic"
            )
        if xlabel:
            ax.set_xlabel("Return period", fontsize=fontsize)
        if xticklabels:
            ax.set_xticklabels(return_periods, fontsize=fontsize)
        else:
            ax.set_xticklabels([])

        # Tidy
        ax.set_ylabel("Relative contribution [%]", fontsize=fontsize)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Add legend below bottom row
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=uc_markers[uc_type],
            lw=3,
            markersize=10,
            color=uc_colors[uc_type],
            label=uc_labels[uc_type],
        )
        for uc_type in uc_labels
    ]
    if plot_total_uc:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="X",
                lw=3,
                markersize=10,
                color="black",
                label="Total uncertainty",
            )
        )
    if legend:
        fig.legend(
            handles=legend_elements,
            loc="outside lower center",
            fontsize=12,
            ncol=2,
            borderaxespad=0.25,
        )

    # Add title
    if fig_title is not None:
        fig.suptitle(fig_title, fontweight="bold", y=y_title)

    # Store
    if store_path is not None:
        fig.savefig(f"../figs/{store_path}.pdf", bbox_inches="tight")

    if axs is None:
        plt.show()

    if return_legend:
        return legend_elements


def plot_response_rls(
    coord_or_mean,
    proj_slice,
    hist_slice,
    fit_method,
    stat_str,
    grid="LOCA2",
    regrid_method="nearest",
    return_periods=[10, 25, 50, 100],
    metric_ids=gev_metric_ids[:3],
    ax_title=False,
    fig_title=None,
    store_path=None,
    axs=None,
    fig=None,
    legend=True,
    idm_start=0,
    y_title=1.05,
    time_str=None,
    xlabel=True,
    xticklabels=True,
    fontsize=10,
):
    # Make figure
    if axs is None:
        fig, axs = plt.subplots(
            1,
            len(metric_ids),
            figsize=(3.5 * len(metric_ids), 4.5),
            sharey=True,
            gridspec_kw={"wspace": 0.1},
            layout="constrained",
        )

    # Loop through metrics
    for idm, metric_id in enumerate(metric_ids):
        # Read all return levels
        ds = []
        for return_period in return_periods:
            if stat_str == "stat":
                file_path = f"{project_data_path}/results/summary_{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}.nc"
            else:
                file_path = f"{project_data_path}/results/summary_{metric_id}_{proj_slice}_{return_period}yr_return_level_{time_str}_{fit_method}_{stat_str}_{grid}grid_{regrid_method}.nc"
            ds.append(xr.open_dataset(file_path))
        ds = xr.combine_by_coords(ds)

        # Mask out locations without all three ensembles
        mask = xr.open_dataset(f"{project_data_path}/mask.nc")["mask"]
        ds = ds.where(mask, drop=True)

        # Ready for plot
        if coord_or_mean == "mean":
            df = (
                ds.mean(dim=["lat", "lon"]).mean(dim=["ensemble", "ssp"]).to_dataframe()
            )
        else:
            df = (
                ds.sel(
                    lat=coord_or_mean[0], lon=360 + coord_or_mean[1], method="nearest"
                )
                .mean(dim=["ensemble", "ssp"])
                .to_dataframe()
            )

        # Reshape
        df = df[[f"{return_period}yr_return_level" for return_period in return_periods]]
        df.columns = [int(col.split("yr")[0]) for col in df.columns]
        df_plot = pd.DataFrame(
            df.stack().rename("return_level").rename_axis(["quantile", "return_period"])
        ).sort_index()

        # Plot mean and 99% range
        ax = axs[idm] if isinstance(axs, list) else axs
        ax.plot(return_periods, df_plot.loc["mean"], color="white")
        ax.fill_between(
            return_periods,
            y1=df_plot.loc["q01"]["return_level"],
            y2=df_plot.loc["q99"]["return_level"],
            alpha=0.9,
            color="gray",
        )

        # Tidy
        ax.grid(alpha=0.5)
        ax.set_xticks(return_periods)
        title_str = title_labels[metric_id]
        if ax_title:
            ax.set_title(
                f"{subplot_labels[idm + idm_start]} {title_str}",
                fontstyle="italic",
                loc="left",
            )
        if xlabel:
            ax.set_xlabel("Return period", fontsize=fontsize)
        if xticklabels:
            ax.set_xticklabels(return_periods, fontsize=fontsize)
        else:
            ax.set_xticklabels([])

        # Tidy
        ylabel = "Change in return level" if "diff" in time_str else "Return level"
        ax.set_ylabel(f"{ylabel} {gev_labels[metric_id]}", fontsize=fontsize)

    # Add title
    if fig_title is not None:
        fig.suptitle(fig_title, fontweight="bold", y=y_title)

    # Store
    if store_path is not None:
        fig.savefig(f"../figs/{store_path}.pdf", bbox_inches="tight")

    if axs is None:
        plt.show()
