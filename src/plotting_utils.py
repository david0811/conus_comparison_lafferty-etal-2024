import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

from utils import gard_gcms
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
ensemble_markers = {"LOCA2": "D", "GARD-LENS": "X", "STAR-ESDM": "s"}


##################
# Map plotting
##################
def plot_uc_map(
    metric_id,
    proj_slice,
    hist_slice,
    return_period,
    fig=None,
    axs=None,
    regrid_method="nearest",
    norm=None,
    cbar=True,
    title="auto",
):
    # Read
    file_path = f"{project_data_path}/results/{metric_id}_{proj_slice}_{hist_slice}_{return_period}rl_{regrid_method}.nc"
    uc = xr.open_dataset(file_path)

    # Mask out locations without all three ensembles
    mask = uc.to_array().sum(dim="variable", skipna=False) >= 0.0
    uc = uc.where(mask, drop=True)

    # Normalize
    if norm == None:
        pass
    elif norm == "relative":
        uc_tot = uc["ssp_uc"] + uc["gcm_uc"] + uc["iv_uc"] + uc["dsc_uc"]
        uc["ssp_uc"] = uc["ssp_uc"] / uc_tot
        uc["gcm_uc"] = uc["gcm_uc"] / uc_tot
        uc["iv_uc"] = uc["iv_uc"] / uc_tot
        uc["dsc_uc"] = uc["dsc_uc"] / uc_tot
    else:
        uc["ssp_uc"] = uc["ssp_uc"] / uc[norm]
        uc["gcm_uc"] = uc["gcm_uc"] / uc[norm]
        uc["iv_uc"] = uc["iv_uc"] / uc[norm]
        uc["dsc_uc"] = uc["dsc_uc"] / uc[norm]

    # Labels
    uc_labels = {
        "ssp_uc": "Scenario uncertainty",
        "gcm_uc": "Response uncertainty",
        "iv_uc": "Internal variability",
        "dsc_uc": "Downscaling uncertainty",
    }

    cbar_labels = {
        "max_tasmax": "[C]",
        "max_cdd": "[degree days]",
        "max_hdd": "[degree days]",
        "max_pr": "[mm]",
        "min_tasmin": "[C]",
    }

    title_labels = {
        "max_tasmax": f"{return_period} year return level: annual maximum temperature",
        "max_cdd": f"{return_period} year return level: annual 1-day maximum CDD",
        "max_hdd": f"{return_period} year return level: annual 1-day maximum HDD",
        "max_pr": f"{return_period} year return level: annual 1-day maximum precipitation",
        "min_tasmin": f"{return_period} year return level: annual minimum temperature",
    }

    norm_labels = {
        'uc_99w': '99% range',
        'uc_95w': '95% range',
        'uc_range': 'Total range'
    }

    if axs is None:
        fig, axs = plt.subplots(
            1,
            5,
            figsize=(12, 5),
            layout="constrained",
            subplot_kw=dict(projection=ccrs.LambertConformal()),
        )

    # Plot details
    if metric_id == "max_pr":
        cmap = "Blues"
        vmin = np.round(uc[norm].min().to_numpy(), decimals=-1)
        vmax = np.round(uc[norm].quantile(0.95).to_numpy(), decimals=-1)
    else:
        cmap = "Oranges"
        vmin = np.round(uc[norm].min().to_numpy(), decimals=0)
        vmax = np.round(uc[norm].quantile(0.95).to_numpy(), decimals=0)

    if norm is not None:
        vmin_uc, vmax_uc = 0.0, 40
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
        levels=11,
        add_colorbar=True,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={
            "orientation": "horizontal",
            "shrink": 0.9,
            "label": f"{norm_labels[norm]} {cbar_labels[metric_id]}",
        },
    )
    # Tidy
    ax.coastlines()
    gl = ax.gridlines(
        draw_labels=False, x_inline=False, rotate_labels=False, alpha=0.2
    )
    ax.add_feature(cfeature.STATES, edgecolor="black", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.5)
    ax.set_extent([-120, -73, 22, 51], ccrs.Geodetic())
    ax.set_title("Total uncertainty", fontsize=12)

    # Loop through uncertainties
    for axi, uc_type in enumerate(list(uc_labels.keys())):
        ax = axs[axi + 1]
        p = (scale_factor * uc[uc_type]).plot(
            ax=ax,
            levels=11,
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
            cbar_label = f"Absolute uncertainty {cbar_labels[metric_id]}"
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
            fig.suptitle(title_labels[metric_id], fontweight="bold", y=0.95)
        else:
            fig.suptitle(title, fontweight="bold", y=0.95)

    return p


#######################
# 'Qualitative' plot
#######################
def plot_boxplot(df, plot_col, positions, color, ax, limits=None, lw=1.5):
    # Filter data below limits if desired
    if limits is not None:
        data = df[(df[plot_col] < limits[1]) & (df[plot_col] > limits[0])][
            plot_col
        ].to_numpy()
    else:
        data = df[plot_col].to_numpy()

    # Plot
    bp = ax.boxplot(
        x=data,
        positions=positions,
        patch_artist=True,
        capprops=dict(color=color, linewidth=lw),
        boxprops=dict(color=color, linewidth=lw),
        whiskerprops=dict(color=color, linewidth=lw),
        flierprops=dict(
            markerfacecolor=color, linestyle="none", markeredgecolor=color
        ),
        medianprops=dict(color="white", linewidth=lw, zorder=5),
        vert=False,
        showmeans=False,
        widths=0.4,
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(color)


def plot_response_differences(df, plot_col, xlabel, ax=None, min_members=5):
    # Plot info
    ensembles = df["ensemble"].unique()
    ssps = df["ssp"].unique()

    # Create a new figure and axis if none are provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    idx = 0
    ylabels = []

    # Loop through ensembles
    for ensemble in ensembles:
        for ssp in ssps:
            # Filter minimum members
            df_sel = df[(df["ensemble"] == ensemble) & (df["ssp"] == ssp)]
            min_filter = df_sel.groupby("gcm")[plot_col].count() >= min_members
            if min_filter.sum() > 0:
                # Plot forced responses
                gcms = min_filter[min_filter].index
                plot_df = pd.DataFrame(
                    df_sel.groupby("gcm")[plot_col].mean().loc[gcms]
                )
                ax.scatter(
                    y=[idx] * len(plot_df),
                    x=plot_df[plot_col],
                    c=ssp_colors[ssp],
                    s=100,
                )
                ax.plot(
                    plot_df[plot_col],
                    [idx] * len(plot_df),
                    linestyle="-",
                    color=ssp_colors[ssp],
                )
                idx += 1
                ylabels.append(f"{ensemble}\n{ssp} ({len(plot_df)})")

    # Tidy
    ax.set_yticks(np.arange(len(ylabels)), ylabels, fontsize=10)
    ax.set_ylim([-0.5, len(ylabels) - 0.5])
    ax.grid(alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_title("Response uncertainty")


def plot_scenario_differences(df, plot_col, xlabel, ax=None, min_members=5):
    # Plot info
    ensembles = df["ensemble"].unique()
    ssps = df["ssp"].unique()

    # Create a new figure and axis if none are provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    idx = 0
    ylabels = []

    # Loop through ensembles
    for ensemble in ensembles:
        # Skip ensembles with only 1 SSP
        if len(df[df["ensemble"] == ensemble]["ssp"].unique()) > 1:
            # Plot line connecting values
            df_sel = df[df["ensemble"] == ensemble]
            plot_df = pd.DataFrame(df_sel.groupby("ssp")[plot_col].mean())
            ax.plot(
                plot_df[plot_col],
                [idx] * len(plot_df),
                linestyle="-",
                color="darkgray",
                zorder=1,
            )
            # Plot each value
            for ssp in ssps:
                df_sel = df[(df["ensemble"] == ensemble) & (df["ssp"] == ssp)]
                ax.scatter(
                    y=[idx],
                    x=df_sel[plot_col].mean(),
                    c=ssp_colors[ssp],
                    s=100,
                    zorder=2,
                )
            idx += 1
            ylabels.append(f"{ensemble}")

    # Tidy
    ax.set_yticks(np.arange(len(ylabels)), ylabels, fontsize=10)
    ax.set_ylim([-0.5, len(ylabels) - 0.5])
    ax.grid(alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_title("Scenario uncertainty")

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ssp_colors[ssp],
            markersize=15,
            label=ssp_labels[ssp],
        )
        for ssp in ssp_colors.keys()
    ]
    ax.legend(handles=legend_elements)


def plot_scenario_differences_by_gcm(
    df, plot_col, xlabel, ax=None, min_members=5
):
    # Plot info
    ensembles = df["ensemble"].unique()
    ssps = df["ssp"].unique()

    # Create a new figure and axis if none are provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    idx = 0
    ylabels = []

    # Loop through ensembles
    for ensemble in ensembles:
        # Skip ensembles with only 1 SSP
        if len(df[df["ensemble"] == ensemble]["ssp"].unique()) > 1:
            # Loop through GCMs
            gcms = df[df["ensemble"] == ensemble]["gcm"].unique()
            for gcm in gcms:
                # Ensure at least 2 SSPs with minimum members
                df_sel = df[(df["ensemble"] == ensemble) & (df["gcm"] == gcm)]
                min_filter = (
                    df_sel.groupby("ssp")[plot_col].count() >= min_members
                )
                if min_filter.sum() > 1:
                    # Plot forced response for each SSP
                    for ssp in ssps:
                        plot_val = df_sel[df_sel["ssp"] == ssp][
                            plot_col
                        ].mean()
                        ax.scatter(
                            y=[idx], x=plot_val, c=ssp_colors[ssp], s=100
                        )
                    idx += 1
                    ylabels.append(f"{ensemble} {gcm}")

    # Tidy
    ax.set_yticks(np.arange(len(ylabels)), ylabels, fontsize=10)
    ax.set_ylim([-0.5, len(ylabels) - 0.5])
    ax.grid(alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_title("Scenario uncertainty")

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=ssp_colors[ssp],
            markersize=15,
            label=ssp_labels[ssp],
        )
        for ssp in ssp_colors.keys()
    ]
    ax.legend(handles=legend_elements)


def plot_iv_differences(df, plot_col, xlabel, ax=None, min_members=5):
    # Plot info
    ensembles = df["ensemble"].unique()
    ssps = df["ssp"].unique()

    # Create a new figure and axis if none are provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    idx = 0
    ylabels = []

    # Loop through ensembles
    for ensemble in ensembles:
        for ssp in ssps:
            # Filter minimum members
            df_sel = df[(df["ensemble"] == ensemble) & (df["ssp"] == ssp)]
            min_filter = df_sel.groupby("gcm")[plot_col].count() >= min_members
            if min_filter.sum() > 0:
                # Loop through GCMs
                gcms = min_filter[min_filter].index
                for gcm in gcms:
                    data = df_sel[df_sel["gcm"] == gcm]
                    if len(data) > 10:
                        plot_boxplot(
                            data, plot_col, [idx], ssp_colors[ssp], ax
                        )
                    else:
                        ax.scatter(
                            y=[idx] * len(data),
                            x=data[plot_col],
                            c=ssp_colors[ssp],
                            s=25,
                        )
                        ax.plot(
                            data[plot_col],
                            [idx] * len(data),
                            linestyle="-",
                            color=ssp_colors[ssp],
                        )
                    idx += 1
                    ylabels.append(f"{ensemble} {gcm} ({len(data)})")
    # Tidy
    ax.grid(alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_title("Internal variability")
    ax.set_yticks(np.arange(len(ylabels)), ylabels, fontsize=10)


def plot_ds_differences(df, plot_col, xlabel, ax=None):
    gcms = df["gcm"].unique()
    members = df["member"].unique()

    # Create a new figure and axis if none are provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Track the y position for each unique combination of fixed columns
    y_labels = []
    y_pos = 0

    # Loop over and search for combos
    for gcm in gcms:
        for member in members:
            # Select GCM/member combo
            df_sel = df[(df["gcm"] == gcm) & (df["member"] == member)]
            # Get SSPS with more than 1 downdcsaling method
            ssps_to_plot = df_sel.groupby("ssp")[plot_col].count() > 1
            ssps_to_plot = ssps_to_plot.index[ssps_to_plot]
            # Split y positions by model/member
            if len(ssps_to_plot) == 0:
                continue
            elif len(ssps_to_plot) == 1:
                y_incs = [0]
            else:
                y_incs = np.linspace(-0.2, 0.2, len(ssps_to_plot))
            # Loop through SSPs
            for idy, ssp in enumerate(ssps_to_plot):
                df_sel_ssp = df_sel[df_sel["ssp"] == ssp]
                for ensemble in df_sel_ssp["ensemble"].unique():
                    # Plot values
                    ax.scatter(
                        y=[y_pos + y_incs[idy]],
                        x=df_sel_ssp[df_sel_ssp["ensemble"] == ensemble][
                            plot_col
                        ],
                        c=ssp_colors[ssp],
                        s=25,
                        marker=ensemble_markers[ensemble],
                    )
                # Plot line connecting values
                ax.plot(
                    df_sel_ssp[plot_col],
                    [y_pos + y_incs[idy]] * len(df_sel_ssp),
                    linestyle="-",
                    color=ssp_colors[ssp],
                )
            y_pos += 1
            y_labels.append(f"{gcm} {member}")

    # Tidy
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel(plot_col)
    ax.grid(alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_title("Downscaling uncertainty")

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=ensemble_markers[ensemble],
            color="w",
            markerfacecolor="black",
            markersize=7,
            label=ensemble,
        )
        for ensemble in ensemble_markers.keys()
    ]
    ax.legend(handles=legend_elements)


def plot_ds_differences_old(df, plot_col, xlabel, ax=None):
    # Create a new figure and axis if none are provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Group by fixed columns
    group_cols = ["gcm", "member", "ssp"]
    diff_cols = ["ensemble"]
    grouped = df.groupby(group_cols)

    # Track the y position for each unique combination of fixed columns
    y_labels = []
    y_pos = 0

    # Loop over each group of fixed columns
    for fixed_values, group in grouped:
        # Ensure there are at least two unique combinations of diff_cols
        if group[diff_cols].drop_duplicates().shape[0] < 2:
            continue  # Skip if there aren't at least two unique combinations

        # Sort by diff_cols to ensure consistent line plots
        group_sorted = group.sort_values(by=diff_cols)

        # Generate a label for the y-axis with only values, separated by spaces
        gcm, member, ssp = fixed_values
        y_labels.append(f"{gcm} {member}")

        # Plot different ensemble values
        for ensemble in group_sorted["ensemble"].unique():
            ax.scatter(
                group_sorted[group_sorted["ensemble"] == ensemble][plot_col],
                [y_pos],
                c=ssp_colors[ssp],
                s=25,
                marker=ensemble_markers[ensemble],
            )

        # Plot line connecting points that differ by diff_cols
        ax.plot(
            group_sorted[plot_col],
            [y_pos] * len(group_sorted),  # Fixed y-position for this group
            linestyle="-",
            color=ssp_colors[ssp],
        )

        # Increment y position for the next label
        y_pos += 1

    # Tidy
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_xlabel(plot_col)
    ax.grid(alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_title("Downscaling uncertainty")

    # Legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=ensemble_markers[ensemble],
            color="w",
            markerfacecolor="black",
            markersize=7,
            label=ensemble,
        )
        for ensemble in ensemble_markers.keys()
    ]
    ax.legend(handles=legend_elements)


def plot_decomp_qual(
    df,
    plot_col,
    xlabel,
    ssp_by_gcm=False,
    limits=None,
    min_members=5,
    axs=None,
):
    # Make axes
    if axs is None:
        fig, axs = plt.subplots(
            2,
            2,
            figsize=(12, 10),
            sharex=True,
            layout="constrained",
            height_ratios=[1, 2],
        )
        axs = axs.flatten()

    ### Plot 1: Scenario uncertainty
    ax = axs[0]
    if ssp_by_gcm:
        plot_scenario_differences_by_gcm(df, plot_col, xlabel, ax=ax)
    else:
        plot_scenario_differences(df, plot_col, xlabel, ax=ax)

    ## Plot 2: Response uncertainty
    ax = axs[1]
    plot_response_differences(df, plot_col, xlabel, ax=ax)

    ### Plot 3: Internal variability
    ax = axs[2]
    plot_iv_differences(df, plot_col, xlabel, ax=ax)

    ## Plot 4: Downscaling
    ax = axs[3]
    plot_ds_differences(df, plot_col, xlabel, ax=ax)

    plt.show()


##########################
# Simpler qualitative plot
##########################
def plot_all_boxplots(df, plot_col, xlabel, title, legend, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ensembles = df["ensemble"].unique()
    ssps = df["ssp"].unique()

    y_pos = []
    y_counter = 0
    y_labels = []

    # Loop through ensembles
    for ensemble in ensembles:
        y_counter += 1
        # Separate by GCMs for GARD-LENS
        if ensemble == "GARD-LENS":
            ssp = "ssp370"
            for gcm in gard_gcms:
                # Plot
                df_sel = df[(df["ensemble"] == ensemble) & (df["gcm"] == gcm)]
                plot_boxplot(
                    df_sel, plot_col, [y_counter], ssp_colors[ssp], ax, lw=2
                )
                # Labels
                y_labels.append(f"{ensemble}: {gcm}")
                y_pos.append(y_counter)
                y_counter += 1
        else:
            # Get SSPs to plot
            ssps = df[df["ensemble"] == ensemble]["ssp"].unique()
            y_labels.append(ensemble)
            y_pos.append(y_counter + np.median(range(len(ssps))))
            for ssp in ssps:
                # Plot
                df_sel = df[(df["ensemble"] == ensemble) & (df["ssp"] == ssp)]
                plot_boxplot(
                    df_sel, plot_col, [y_counter], ssp_colors[ssp], ax, lw=2
                )
                # Single labels
                y_counter += 1

    # Tidy
    ax.set_yticks(y_pos, y_labels)
    ax.grid(alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")

    # Legend
    if legend:
        legend_elements = [
            Line2D(
                [0], [0], color=ssp_colors[ssp], label=ssp_labels[ssp], lw=2
            )
            for ssp in ssp_colors.keys()
        ]
        ax.legend(handles=legend_elements)
