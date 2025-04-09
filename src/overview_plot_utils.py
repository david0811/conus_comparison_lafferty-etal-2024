import pandas as pd

import plotting_utils as pu

from utils import roar_data_path as project_data_path


## Plot 1: scenario uncertainty
def plot_scenario_uncertainty(
    df, metric_id, unit, ax, ylims=None, xtext=0.99, ytext=0.02, ensemble="STAR-ESDM"
):
    var_id = metric_id.split("_")[1]

    df_ssp245 = df[(df["ssp"] == "ssp245") & (df["ensemble"] == ensemble)]
    df_ssp585 = df[(df["ssp"] == "ssp585") & (df["ensemble"] == ensemble)]

    # Plot all
    for df_plot in [df_ssp245, df_ssp585]:
        for gcm in df_plot["gcm"].unique():
            df_plot_sel = df_plot[df_plot["gcm"] == gcm].sort_values("time")
            ax.plot(df_plot_sel["time"], df_plot_sel[var_id], color="silver", alpha=0.5)

    # Plot means
    df_ssp245.query("time <= 2015").groupby("time").mean(numeric_only=True).plot(
        y="tasmax", legend=None, color="black", lw=2, ax=ax
    )
    df_ssp585.query("time > 2015").groupby("time").mean(numeric_only=True).plot(
        y="tasmax", label="SSP5-8.5", lw=2, ax=ax
    )
    df_ssp245.query("time > 2015").groupby("time").mean(numeric_only=True).plot(
        y="tasmax", label="SSP2-4.5", lw=2, ax=ax
    )

    # Tidy
    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} {unit}")
    ax.set_title("Scenario uncertainty", loc="left", style="italic")
    ax.text(
        xtext,
        ytext,
        f"({ensemble})",
        horizontalalignment="right",
        transform=ax.transAxes,
    )
    ax.spines[["right", "top"]].set_visible(False)
    if ylims is not None:
        ax.set_ylim(ylims)

    # Add index of scenario uncertainty
    y1 = (
        df_ssp245.query("time > 2015")
        .groupby("time")
        .mean(numeric_only=True)[var_id]
        .iloc[-1:]
        .min()
    )
    y2 = (
        df_ssp585.query("time > 2015")
        .groupby("time")
        .mean(numeric_only=True)[var_id]
        .iloc[-1:]
        .max()
    )
    ax.annotate(
        "",
        xy=(2103, y1),
        xytext=(2103, y2),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", lw=1.5, shrinkA=0, shrinkB=0),
    )


## Plot 2: response uncertainty
def plot_response_uncertainty(
    df,
    metric_id,
    unit,
    ax,
    ylims=None,
    xtext=0.99,
    ytext=0.02,
    ensemble="LOCA2",
    ssp="ssp245",
    gcm1="CanESM5",
    gcm2="MPI-ESM1-2-HR",
):
    var_id = metric_id.split("_")[1]
    ssp_name = pu.ssp_labels[ssp]

    df_gcm1 = df[
        (df["ssp"].isin([ssp, "historical"]))
        & (df["ensemble"] == ensemble)
        & (df["gcm"] == gcm1)
    ]
    df_gcm2 = df[
        (df["ssp"].isin([ssp, "historical"]))
        & (df["ensemble"] == ensemble)
        & (df["gcm"] == gcm2)
    ]

    # Plot all
    for df_plot in [df_gcm1, df_gcm2]:
        for member in df_plot["member"].unique():
            df_plot_sel = df_plot[df_plot["member"] == member].sort_values("time")
            ax.plot(df_plot_sel["time"], df_plot_sel[var_id], color="silver", alpha=0.5)

    # Plot means
    df_gcm1.groupby("time").mean(numeric_only=True).plot(
        y="tasmax", label=gcm1, lw=2, ax=ax
    )
    df_gcm2.groupby("time").mean(numeric_only=True).plot(
        y="tasmax", label=gcm2, lw=2, ax=ax
    )

    # Tidy
    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} {unit}")
    ax.set_title("Response uncertainty", loc="left", style="italic")
    ax.text(
        xtext,
        ytext,
        f"({ensemble}, {ssp_name})",
        horizontalalignment="right",
        transform=ax.transAxes,
    )
    ax.spines[["right", "top"]].set_visible(False)
    if ylims is not None:
        ax.set_ylim(ylims)

    # Add index of scenario uncertainty
    y1 = df_gcm1.groupby("time").mean(numeric_only=True)[var_id].iloc[-1:].max()
    y2 = df_gcm2.groupby("time").mean(numeric_only=True)[var_id].iloc[-1:].min()

    ax.annotate(
        "",
        xy=(2103, y1),
        xytext=(2103, y2),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", lw=1.5, shrinkA=0, shrinkB=0),
    )


## Plot 3: internal variability
def plot_internal_variability(
    df,
    metric_id,
    unit,
    ax,
    ensemble="GARD-LENS",
    ssp="ssp370",
    gcm="CESM2-LENS",
    ylims=None,
    xtext=0.99,
    ytext=0.02,
):
    var_id = metric_id.split("_")[1]
    ssp_name = pu.ssp_labels[ssp]

    df_gcm = df[(df["ssp"] == ssp) & (df["ensemble"] == ensemble) & (df["gcm"] == gcm)]

    # Plot all
    for member in df_gcm["member"].unique():
        df_plot_sel = df_gcm[df_gcm["member"] == member].sort_values("time")
        ax.plot(df_plot_sel["time"], df_plot_sel[var_id], color="silver", alpha=0.5)

    # Add mean
    df_gcm.groupby("time").mean(numeric_only=True)[var_id].plot(
        color="C0", label=gcm, lw=2, ax=ax
    )

    # Tidy
    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} {unit}")
    ax.set_title("Internal variability", loc="left", style="italic")
    ax.text(
        xtext,
        ytext,
        f"({ensemble}, {ssp_name})",
        horizontalalignment="right",
        transform=ax.transAxes,
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend()
    if ylims is not None:
        ax.set_ylim(ylims)

    # Add index of scenario uncertainty
    y1 = df_gcm.query("time == 2100")[var_id].max()
    y2 = df_gcm.query("time == 2100")[var_id].min()

    ax.annotate(
        "",
        xy=(2103, y1),
        xytext=(2103, y2),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", lw=1.5, shrinkA=0, shrinkB=0),
    )


## Plot 4: downscaling uncertainty
def plot_downscaling_uncertainty(
    df,
    metric_id,
    unit,
    ax,
    gcm="CanESM5",
    ssp="ssp370",
    ensemble1="GARD-LENS",
    ensemble2="LOCA2",
    member="r3i1p1f1",
    ylims=None,
    xtext=0.99,
    ytext=0.02,
):
    var_id = metric_id.split("_")[1]
    ssp_name = pu.ssp_labels[ssp]

    df_ensemble1 = df[
        (df["ssp"] == ssp)
        & (df["ensemble"] == ensemble1)
        & (df["gcm"] == gcm)
        & (df["member"] == member)
    ]
    df_ensemble2 = df[
        (df["ssp"].isin([ssp, "historical"]))
        & (df["ensemble"] == ensemble2)
        & (df["gcm"] == gcm)
        & (df["member"] == member)
    ]

    # Plot all
    ax.plot(df_ensemble1["time"], df_ensemble1[var_id], label=ensemble1)
    ax.plot(df_ensemble2["time"], df_ensemble2[var_id], label=ensemble2)

    # Tidy
    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} {unit}")
    ax.set_title("Downscaling uncertainty", loc="left", style="italic")
    ax.text(
        xtext,
        ytext,
        f"({gcm}, {member}, {ssp_name})",
        horizontalalignment="right",
        transform=ax.transAxes,
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend()
    if ylims is not None:
        ax.set_ylim(ylims)

    # Add index of scenario uncertainty
    y1 = df_ensemble1.query("time > 2099")[var_id].min()
    y2 = df_ensemble2.query("time > 2099")[var_id].max()

    ax.annotate(
        "",
        xy=(2103, y1),
        xytext=(2103, y2),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", lw=1.5, shrinkA=0, shrinkB=0),
    )


# Plot 5: GEV fit uncertainty
def plot_gev_uncertainty(
    df,
    metric_id,
    unit,
    ax,
    city="chicago",
    ensemble="STAR-ESDM",
    gcm="ACCESS-CM2",
    ssp="ssp585",
    member="r1i1p1f1",
    col_name="100yr_return_level",
    ylims=None,
    xtext=0.99,
    ytext=0.02,
):
    var_id = metric_id.split("_")[1]
    df_sel = df[
        (df["gcm"] == gcm)
        & (df["ssp"] == ssp)
        & (df["ensemble"] == ensemble)
        & (df["member"] == member)
    ]
    ssp_name = pu.ssp_labels[ssp]

    # Read RLs
    stat_n_boot = 1000
    nonstat_n_boot = 250
    sample_str = "_samples"
    stat_fit_method = "lmom"
    nonstat_fit_method = "mle"
    proj_slice = "2050-2100"
    hist_slice = "1950-2014"
    nonstat_slice = "1950-2100"

    # Read both fits
    df_stat = pd.read_csv(
        f"{project_data_path}/extreme_value/cities/original_grid/freq/{city}_{metric_id}_{hist_slice}_{proj_slice}_{stat_fit_method}_stat_nboot{stat_n_boot}{sample_str}.csv"
    )
    df_stat_sel = df_stat[
        (df_stat["gcm"] == gcm)
        & (df_stat["ssp"] == ssp)
        & (df_stat["ensemble"] == ensemble)
        & (df_stat["member"] == member)
    ]

    df_nonstat = pd.read_csv(
        f"{project_data_path}/extreme_value/cities/original_grid/freq/{city}_{metric_id}_{nonstat_slice}_{nonstat_fit_method}_nonstat_nboot{nonstat_n_boot}{sample_str}.csv"
    )
    df_nonstat_sel = df_nonstat[
        (df_nonstat["gcm"] == gcm)
        & (df_nonstat["ssp"] == ssp)
        & (df_nonstat["ensemble"] == ensemble)
        & (df_nonstat["member"] == member)
    ]
    df_nonstat_sel_main = df_nonstat_sel[df_nonstat_sel["n_boot"] == "main"]

    # Plot
    ax.plot(df_sel["time"], df_sel[var_id], color="black")

    # Proj
    ax.fill_between(
        x=[2050, 2100],
        y1=[df_stat_sel[f"{col_name}_proj"].quantile(0.975)],
        y2=[df_stat_sel[f"{col_name}_proj"].quantile(0.025)],
        color="C0",
        alpha=0.5,
    )
    ax.hlines(
        df_stat_sel[f"{col_name}_proj"].median(),
        2050,
        2100,
        colors="C0",
        ls="--",
        label="Stationary (2050-2100)",
    )
    y1 = df_stat_sel[f"{col_name}_proj"].quantile(0.975)
    y2 = df_stat_sel[f"{col_name}_proj"].quantile(0.025)
    ax.annotate(
        "",
        xy=(2103, y1),
        xytext=(2103, y2),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="C0", shrinkA=0, shrinkB=0),
    )

    # Non-stationary
    years = [1950, 1975, 2000, 2050, 2075, 2100]
    ax.plot(
        years,
        df_nonstat_sel_main[[f"{col_name}_{year}" for year in years]]
        .to_numpy()
        .flatten(),
        color="C1",
        ls="dotted",
        label="Non-stationary (1950-2100)",
    )

    ax.fill_between(
        x=years,
        y1=df_nonstat_sel[[f"{col_name}_{year}" for year in years]].quantile(0.975),
        y2=df_nonstat_sel[[f"{col_name}_{year}" for year in years]].quantile(0.025),
        color="C1",
        alpha=0.5,
    )
    y1 = df_nonstat_sel[f"{col_name}_2100"].quantile(0.975)
    y2 = df_nonstat_sel[f"{col_name}_2100"].quantile(0.025)
    ax.annotate(
        "",
        xy=(2106, y1),
        xytext=(2106, y2),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="C1", shrinkA=0, shrinkB=0),
    )

    # Tidy
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} {unit}")
    ax.set_title(
        "GEV fit uncertainty: 100-year return level", loc="left", fontstyle="italic"
    )
    ax.text(
        xtext,
        ytext,
        f"({ensemble}, {gcm}, {member}, {ssp_name})",
        horizontalalignment="right",
        transform=ax.transAxes,
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend()

    # Apply the UC and return for second plot
    return df_stat, df_nonstat
    # uc_df_stat_abs = sacu.calculate_df_uc(df_stat, "")
    # uc_df_nonstat = sacu.calculate_df_uc(
    #     df_nonstat, city, metric_id, ssp, ensemble, gcm, member
    # )
