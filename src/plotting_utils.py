import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

import sa_city_utils as sacu
from utils import gard_gcms, gev_metric_ids
from utils import roar_data_path as project_data_path

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
unit_labels = {
    "max_tasmax": "[C]",
    "max_cdd": "[degree days]",
    "max_hdd": "[degree days]",
    "max_pr": "[mm]",
    "min_tasmin": "[C]",
    "avg_tas": "\n[C/decade]",
    "sum_pr": "\n[mm/decade]",
    "sum_cdd": "\n[degree days/decade]",
    "sum_hdd": "\n[degree days/decade]",
}
title_labels = {
    "max_tasmax": "Annual maximum temperature",
    "max_cdd": "Annual maximum 1-day CDD",
    "max_hdd": "Annual maximum 1-day HDD",
    "max_pr": "Annual maximum 1-day precipitation",
    "min_tasmin": "Annual minimum temperature",
    "avg_tas": "Annual average temperature",
    "sum_pr": "Annual total precipitation",
    "sum_cdd": "Annual total cooling degree days",
    "sum_hdd": "Annual total heating degree days",
}

city_names = {
    "chicago": "Chicago",
    "seattle": "Seattle",
    "houston": "Houston",
    "denver": "Denver",
    "nyc": "New York City",
    "sanfrancisco": "San Francisco",
}

uc_labels = {
    "ssp_uc": "Scenario uncertainty",
    "gcm_uc": "Response uncertainty",
    "iv_uc": "Internal variability",
    "dsc_uc": "Downscaling uncertainty",
    "gev_uc": "GEV fit uncertainty",
}

uc_colors = {
    "ssp_uc": "#0077BB",
    "gcm_uc": "#33BBEE",
    "iv_uc": "#009988",
    "dsc_uc": "#EE3377",
    "gev_uc": "#CC3311",
}

uc_markers = {
    "ssp_uc": "v",
    "gcm_uc": "^",
    "iv_uc": "o",
    "dsc_uc": "D",
    "gev_uc": "s",
}

subfigure_labels = ["a)", "b)", "c)", "d)", "e)", "f)", "g)", "h)", "i)"]


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
    analysis_type,
    regrid_method="nearest",
    fig=None,
    axs=None,
    norm=None,
    cbar=False,
    vmax_uc=40,
    title="auto",
    y_title=0.98,
):
    # Read
    if analysis_type == "trend":
        file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{plot_col}_{grid}grid_{regrid_method}.nc"
    elif analysis_type == "extreme_value":
        stat_str = "stat" if stationary else "nonstat"
        file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{fit_method}_{stat_str}_{grid}grid_{regrid_method}.nc"

    uc = xr.open_dataset(file_path)

    # Mask out locations without all three ensembles
    mask = uc.to_array().sum(dim="variable", skipna=False) >= 0.0
    uc = uc.where(mask, drop=True)

    # Normalize
    if norm is None:
        pass
    elif norm == "relative":
        uc_tot = uc["ssp_uc"] + uc["gcm_uc"] + uc["iv_uc"] + uc["dsc_uc"] + uc["gev_uc"]
        uc["ssp_uc"] = uc["ssp_uc"] / uc_tot
        uc["gcm_uc"] = uc["gcm_uc"] / uc_tot
        uc["iv_uc"] = uc["iv_uc"] / uc_tot
        uc["dsc_uc"] = uc["dsc_uc"] / uc_tot
        if analysis_type == "extreme_value":
            uc["gev_uc"] = uc["gev_uc"] / uc_tot
    else:
        uc["ssp_uc"] = uc["ssp_uc"] / uc[norm]
        uc["gcm_uc"] = uc["gcm_uc"] / uc[norm]
        uc["iv_uc"] = uc["iv_uc"] / uc[norm]
        uc["dsc_uc"] = uc["dsc_uc"] / uc[norm]
        if analysis_type == "extreme_value":
            uc["gev_uc"] = uc["gev_uc"] / uc[norm]

    title_labels = {
        "max_tasmax": f"{return_period} year return level: annual maximum temperature",
        "max_cdd": f"{return_period} year return level: annual 1-day maximum CDD",
        "max_hdd": f"{return_period} year return level: annual 1-day maximum HDD",
        "max_pr": f"{return_period} year return level: annual 1-day maximum precipitation",
        "min_tasmin": f"{return_period} year return level: annual minimum temperature",
        "avg_tas": "Annual average temperature",
        "sum_pr": "Annual total precipitation",
        "sum_cdd": "Annual total cooling degree days",
        "sum_hdd": "Annual total heating degree days",
    }

    norm_labels = {
        "uc_99w": "99% range",
        "uc_95w": "95% range",
        "uc_range": "Total range",
    }

    if axs is None:
        ncols = 6 if analysis_type == "extreme_value" else 5
        width = 14 if analysis_type == "extreme_value" else 11
        fig, axs = plt.subplots(
            1,
            ncols,
            figsize=(width, 3),
            layout="constrained",
            subplot_kw=dict(projection=ccrs.LambertConformal()),
        )

    # Plot details
    if analysis_type == "trend":
        uc[norm] = uc[norm] * 10  # decadal trends

    # Get vmin, vmax to format nicely for 11 levels
    nlevels = 10
    if metric_id == "avg_tas":  # values are much smaller here
        vmin = np.round(uc[norm].min().to_numpy(), decimals=0)
        vmax = np.round(uc[norm].max().to_numpy(), decimals=0)
    else:
        vmin = np.round(uc[norm].min().to_numpy(), decimals=0)
        raw_range = uc[norm].quantile(0.95).to_numpy() - vmin
        step_size = raw_range / nlevels
        step_size = np.ceil(step_size * 2) / 2  # Round up to nearest 0.5
        vmax = vmin + (step_size * nlevels)  # 10 steps total

    if metric_id in ["max_pr", "sum_pr", "min_tasmin", "max_hdd", "sum_hdd"]:
        cmap = "Blues"
    else:
        cmap = "Oranges"

    if norm is not None:
        vmin_uc, vmax_uc = 0.0, vmax_uc
        scale_factor = 100.0
        cmap_uc = "YlGn"
    else:
        scale_factor = 1.0
        vmin_uc = vmin
        vmax_uc = vmax
        cmap_uc = cmap

    # First plot total uncertainty
    ax = axs[0]
    p = uc[norm].plot(
        ax=ax,
        levels=nlevels + 1,
        add_colorbar=True,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={
            "orientation": "vertical",
            "location": "left",
            "shrink": 0.6,
            "aspect": 10,
            "label": f"{norm_labels[norm]} {unit_labels[metric_id]}",
        },
    )
    # Tidy
    ax.coastlines()
    gl = ax.gridlines(draw_labels=False, x_inline=False, rotate_labels=False, alpha=0.2)
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
    ax.set_extent([-120, -73, 22, 51], ccrs.Geodetic())
    ax.set_title("Total uncertainty", fontsize=12)

    # Loop through uncertainties
    for axi, uc_type in enumerate(list(uc_labels.keys())):
        if analysis_type == "trend" and uc_type == "gev_uc":
            continue
        ax = axs[axi + 1]
        p = (scale_factor * uc[uc_type]).plot(
            ax=ax,
            levels=nlevels + 1,
            add_colorbar=False,
            vmin=vmin_uc,
            vmax=vmax_uc,
            cmap=cmap_uc,
            transform=ccrs.PlateCarree(),
        )

        # Tidy
        ax.coastlines()
        gl = ax.gridlines(
            draw_labels=False, x_inline=False, rotate_labels=False, alpha=0.2
        )
        ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
        ax.set_extent([-120, -73, 22, 51], ccrs.Geodetic())
        ax.set_title(uc_labels[uc_type], fontsize=12)

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
            ax=axs[1:],
            pad=0.05,
            shrink=0.3,
        )

    if title is not None:
        if title == "auto":
            fig.suptitle(title_labels[metric_id], style="italic", y=y_title)
        else:
            fig.suptitle(title, style="italic", y=y_title)

    return p


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
    hist_slice,
    proj_slice,
    fit_method,
    stationary,
    axs,
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
    df = pd.read_csv(
        f"{project_data_path}/extreme_value/cities/original_grid/freq/{city}_{metric_id}_{hist_slice}_{proj_slice}_{fit_method}_{stationary}_nboot{n_boot}{sample_str}.csv"
    )
    df_uc = sacu.calculate_df_uc(df, plot_col, True)
    df = df.set_index(["ensemble", "gcm", "member", "ssp"])

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
    units = unit_labels[metric_id]

    ############################
    # UC
    ############################
    ax = axs[1]

    uc_names = [
        "Scenario \n uncertainty",
        "Response \n uncertainty",
        "Internal \n variability",
        "Downscaling \n uncertainty",
        "GEV fit \n uncertainty",
    ]

    df_uc[df_uc["uncertainty_type"] != "ssp_uc"].plot.bar(
        x="uncertainty_type", y="mean", yerr="std", ax=ax, legend=False
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

    # GARD-LENS
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
        0.99,
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

    # STAR-ESDM
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
        0.99,
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

    # LOCA2
    ensemble = "LOCA2"
    for gcm in df.loc[ensemble].index.unique(level="gcm"):
        for ssp in df.loc[ensemble, gcm].index.unique(level="ssp"):
            if (
                len(df.loc[ensemble, gcm, :, ssp].index.unique(level="member"))
                >= n_min_members
            ):
                df_sel = df.loc[ensemble, gcm, :, ssp]
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
                label_names.append(f"{gcm} ({df_sel.index.nunique()})")
                label_idy.append(idy)
                idy += 0.5

    ax.text(
        0.99,
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
        legend = ax.legend(handles=legend_elements, loc="lower right")
        legend.set_zorder(10)


def plot_uc_bar(df_trend_uc, df_gev_uc, ax, bar_width=0.35, legend=False):
    uc_names = [
        "Scenario \n uncertainty",
        "Response \n uncertainty",
        "Internal \n variability",
        "Downscaling \n uncertainty",
        "GEV fit \n uncertainty",
    ]

    # Normalize by uc_99w
    df_trend_uc = df_trend_uc.apply(lambda x: x / df_trend_uc.loc["uc_99w"]["mean"])
    df_gev_uc = df_gev_uc.apply(lambda x: x / df_gev_uc.loc["uc_99w"]["mean"])

    # Make sure only one SSP type
    df_trend_uc = df_trend_uc.drop(["ssp_uc", "uc_99w"])
    df_gev_uc = df_gev_uc.drop(["ssp_uc", "uc_99w"])

    # Set positions of the bars on X axis
    r1 = np.arange(len(df_trend_uc))
    r2 = [x + bar_width for x in r1]

    # Create the grouped bar chart
    bars1 = ax.bar(
        r1[: len(df_trend_uc)],
        df_trend_uc["mean"],
        width=bar_width,
        color="C0",
        label="Trend (bulk metric)",
        yerr=df_trend_uc["std"],
        capsize=5,
    )
    bars2 = ax.bar(
        r2[: len(df_gev_uc)],
        df_gev_uc["mean"],
        width=bar_width,
        color="C1",
        label="100-year return level",
        yerr=df_gev_uc["std"],
        capsize=5,
    )

    ax.set_xticks([r + bar_width / 2 for r in range(len(df_trend_uc))])
    ax.set_xticklabels(uc_names, rotation=45, fontsize=10)
    # ax.set_ylabel("Relative contribution")
    ax.set_xlabel("")
    ax.grid(alpha=0.2, zorder=3)
    if legend:
        ax.legend(loc="upper right")


#############################
# UC breakdown by return level
#############################


def plot_uc_rls(
    coord_or_mean,
    proj_slice,
    hist_slice,
    fit_method,
    stat_str,
    grid,
    regrid_method,
    total_uc="uc_99w",
    return_periods=[10, 25, 50, 100],
    title=None,
    store_path=None,
):
    # Make figure
    fig, axs = plt.subplots(
        2,
        3,
        figsize=(11, 7),
        sharey=True,
        gridspec_kw={"wspace": 0.07},
        layout="constrained",
    )
    axs = axs.flatten()

    # Loop through metrics
    for idm, metric_id in enumerate(gev_metric_ids):
        # Read all return levels
        ds = []
        for return_period in return_periods:
            file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{return_period}yr_return_level_{fit_method}_{stat_str}_{grid}grid_{regrid_method}.nc"
            ds.append(
                xr.open_dataset(file_path).assign_coords(return_period=return_period)
            )
        ds = xr.concat(ds, dim="return_period")

        # Ready for plot
        if coord_or_mean == "mean":
            df = (
                ds.mean(dim=["lat", "lon"])
                .to_dataframe()
                .droplevel("quantile")
                .drop(columns=["time", "ensemble"])
            )
        else:
            df = (
                ds.sel(
                    lat=coord_or_mean[0], lon=360 + coord_or_mean[1], method="nearest"
                )
                .to_dataframe()
                .droplevel("quantile")
                .drop(columns=["time", "ensemble"])
            )
        df_total_uc = df[total_uc]
        df = df.apply(lambda x: x / df[total_uc])  # normalize

        # Plot total UC first with lower alpha
        ax1 = axs[idm].twinx()
        ax1.plot(df.index, df_total_uc, lw=3, color="black", alpha=0.5)
        ax1.scatter(df.index, df_total_uc, s=50, marker="H", color="black", alpha=0.5)
        ax1.set_ylabel(
            f"Total uncertainty {unit_labels[metric_id]}", rotation=-90, va="bottom"
        )

        # Plot UC components on top with higher alpha
        ax = axs[idm]
        for uc_type in uc_labels:
            ax.plot(df.index, df[uc_type], lw=2, color=uc_colors[uc_type], alpha=0.9)
            ax.scatter(
                df.index,
                df[uc_type],
                s=50,
                marker=uc_markers[uc_type],
                color=uc_colors[uc_type],
                alpha=0.9,
            )

        # Tidy
        ax.grid(alpha=0.5)
        ax.set_xticks(return_periods)
        ax.set_title(f"{subfigure_labels[idm]} {title_labels[metric_id]}")

    # Tidy
    axs[0].set_ylabel("Relative contribution")
    axs[3].set_ylabel("Relative contribution")

    axs[2].set_xlabel("Return period")
    axs[3].set_xlabel("Return period")
    axs[4].set_xlabel("Return period")

    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])

    axs[2].set_xticklabels(return_periods)
    axs[3].set_xticklabels(return_periods)
    axs[4].set_xticklabels(return_periods)

    # Legend plot
    axs[-1].axis("off")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="H",
            lw=3,
            alpha=0.5,
            markersize=10,
            color="black",
            label="Total uncertainty",
        )
    ] + [
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

    axs[-1].legend(handles=legend_elements, loc="center", fontsize=12)

    # Add title
    if title is not None:
        fig.suptitle(title, fontweight="bold", y=1.05)

    # Store
    if store_path is not None:
        fig.savefig(f"../figs/{store_path}.pdf", bbox_inches="tight")

    plt.show()
