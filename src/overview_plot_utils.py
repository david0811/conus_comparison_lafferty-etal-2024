import pandas as pd
from matplotlib.lines import Line2D
import plotting_utils as pu

from utils import roar_data_path as project_data_path


## Plot 0: total uncertainty
def plot_total_uncertainty(
    df,
    metric_id,
    unit,
    ax,
    title_index="a)",
    ylims=None,
    xtext=0.99,
    ytext=0.02,
):
    # Get var_id
    var_id = metric_id.split("_")[1]

    # Index
    df_indexed = df.set_index(["gcm", "ssp", "member", "ensemble"]).sort_index()
    unique_combos = df_indexed.index.unique()

    # Plot timeseries
    for combo in unique_combos[::2]:
        # Get entries
        gcm, ssp, member, ensemble = combo
        if ssp == "historical":
            continue
        # Subset
        df_sel = df[
            (df["ensemble"] == ensemble)
            & (df["gcm"] == gcm)
            & (df["member"] == member)
            & (df["ssp"].isin([ssp, "historical"]))
        ]
        df_sel.sort_values("time").plot(
            x="time", y=var_id, color="silver", alpha=0.5, ax=ax, legend=None
        )

    # Plot mean
    df.groupby("time").mean(numeric_only=True).plot(y=var_id, lw=2, color="gray", ax=ax)

    # Plot quantiles
    df.groupby("time").quantile(0.995, numeric_only=True).plot(
        y=var_id, lw=1, color="gray", ax=ax, ls="--"
    )
    df.groupby("time").quantile(0.005, numeric_only=True).plot(
        y=var_id, lw=1, color="gray", ax=ax, ls="--"
    )

    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} anomaly {unit}")
    ax.set_title(f"{title_index} Total uncertainty", loc="left", style="italic")
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(
        xtext,
        ytext,
        f"(All {len(unique_combos)} projections)",
        horizontalalignment="right",
        transform=ax.transAxes,
    )

    # Legend
    legend_elements = [
        Line2D([0], [0], color="gray", lw=1, ls="--", label="99% range"),
        Line2D([0], [0], color="gray", lw=2, label="Ensemble mean"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Add index of uncertainty
    y1 = df.groupby("time").quantile(0.995, numeric_only=True)[var_id].iloc[-1]
    y2 = df.groupby("time").quantile(0.005, numeric_only=True)[var_id].iloc[-1]

    ax.annotate(
        "",
        xy=(2103, y1),
        xytext=(2103, y2),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", lw=1.5, shrinkA=0, shrinkB=0),
    )


## Plot 1: scenario uncertainty
def plot_scenario_uncertainty(
    df,
    metric_id,
    unit,
    ax,
    title_index="b)",
    ylims=None,
    xtext=0.99,
    ytext=0.02,
    ensemble="STAR-ESDM",
):
    # SSP colors
    ssp245_color = "lightsalmon"
    ssp585_color = "sienna"

    # Get variable name
    var_id = metric_id.split("_")[1]

    # Get SSP data
    df_ssp245 = df[(df["ssp"] == "ssp245") & (df["ensemble"] == ensemble)]
    df_ssp585 = df[(df["ssp"] == "ssp585") & (df["ensemble"] == ensemble)]

    # Plot all
    for gcm in df_ssp245["gcm"].unique():
        df_plot = df_ssp245[df_ssp245["gcm"] == gcm]
        # Historical
        df_plot_sel = df_plot.sort_values("time").query("time <= 2015")
        ax.plot(df_plot_sel["time"], df_plot_sel[var_id], color="silver", alpha=0.5)
        # Future
        df_plot_sel = df_plot.sort_values("time").query("time > 2015")
        ax.plot(df_plot_sel["time"], df_plot_sel[var_id], color="silver", alpha=0.5)
    for gcm in df_ssp585["gcm"].unique():
        df_plot = df_ssp585[df_ssp585["gcm"] == gcm]
        df_plot_sel = df_plot.sort_values("time").query("time > 2015")
        ax.plot(df_plot_sel["time"], df_plot_sel[var_id], color="silver", alpha=0.5)

    # Plot means
    df_ssp245.query("time <= 2015").groupby("time").mean(numeric_only=True).plot(
        y=var_id, legend=None, lw=2, ax=ax, color="black"
    )
    df_ssp585.query("time > 2015").groupby("time").mean(numeric_only=True).plot(
        y=var_id, label="SSP5-8.5", lw=2, ax=ax, color=ssp585_color
    )
    df_ssp245.query("time > 2015").groupby("time").mean(numeric_only=True).plot(
        y=var_id, label="SSP2-4.5", lw=2, ax=ax, color=ssp245_color
    )

    # Tidy
    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} anomaly {unit}")
    ax.set_title(f"{title_index} Scenario uncertainty", loc="left", style="italic")
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
    ssp="ssp370",
    gcm1="CanESM5",
    gcm2="MPI-ESM1-2-HR",
    title_index="c)",
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
        y=var_id, label=gcm1, lw=2, ax=ax
    )
    df_gcm2.groupby("time").mean(numeric_only=True).plot(
        y=var_id, label=gcm2, lw=2, ax=ax
    )

    # Tidy
    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} anomaly {unit}")
    ax.set_title(f"{title_index} Response uncertainty", loc="left", style="italic")
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

    # Add index of uncertainty
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
    title_index="d)",
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
        color="gray", label=gcm, lw=2, ax=ax
    )

    # Tidy
    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} anomaly {unit}")
    ax.set_title(f"{title_index} Internal variability", loc="left", style="italic")
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
    title_index="e)",
    gcm="CanESM5",
    ssp="ssp370",
    ensemble1="GARD-LENS",
    ensemble2="LOCA2",
    member="r1i1p1f1",
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
    ax.plot(
        df_ensemble1["time"], df_ensemble1[var_id], label=ensemble1, color="deepskyblue"
    )
    ax.plot(df_ensemble2["time"], df_ensemble2[var_id], label=ensemble2, color="teal")

    # Tidy
    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} anomaly {unit}")
    ax.set_title(f"{title_index} Downscaling uncertainty", loc="left", style="italic")
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
    title_index="f)",
    city="chicago",
    ensemble="STAR-ESDM",
    gcm="ACCESS-CM2",
    ssp="ssp585",
    member="r1i1p1f1",
    col_name="100yr_return_level",
    ylims=None,
    xtext=0.99,
    ytext=0.02,
    include_stat_fit=True,
    anomaly_baseline=0.0
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
    nonstat_n_boot = 100
    sample_str = "_samples"
    stat_fit_method = "lmom"
    nonstat_fit_method = "mle"
    proj_slice = "2050-2100"
    hist_slice = "1950-2014"
    nonstat_slice = "1950-2100"

    # Read both fits
    df_stat = pd.read_csv(
        f"{project_data_path}/extreme_value/cities/original_grid/freq/{city}_{metric_id}_{hist_slice}_{proj_slice}_{stat_fit_method}_stat_nbootproj{stat_n_boot}_nboothist1{sample_str}.csv"
    )
    df_stat_sel = df_stat[
        (df_stat["gcm"] == gcm)
        & (df_stat["ssp"] == ssp)
        & (df_stat["ensemble"] == ensemble)
        & (df_stat["member"] == member)
    ]

    df_nonstat = pd.read_csv(
        f"{project_data_path}/extreme_value/cities/original_grid/freq/{city}_{metric_id}_{nonstat_slice}_{nonstat_fit_method}_nonstat_nboot{nonstat_n_boot}{sample_str}_scale.csv"
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
    if include_stat_fit:
        ax.fill_between(
            x=[2050, 2100],
            y1=[df_stat_sel[f"{col_name}_proj"].quantile(0.975) - anomaly_baseline],
            y2=[df_stat_sel[f"{col_name}_proj"].quantile(0.025) -  anomaly_baseline],
            color="violet",
            alpha=0.5,
        )
        ax.hlines(
            df_stat_sel[f"{col_name}_proj"].median() - anomaly_baseline,
            2050,
            2100,
            colors="violet",
            ls="--",
            label="Stationary (2050-2100)",
        )
        y1 = df_stat_sel[f"{col_name}_proj"].quantile(0.975) - anomaly_baseline
        y2 = df_stat_sel[f"{col_name}_proj"].quantile(0.025) - anomaly_baseline
        ax.annotate(
            "",
            xy=(2103, y1),
            xytext=(2103, y2),
            textcoords="data",
            arrowprops=dict(
                arrowstyle="<->", lw=1.5, color="violet", shrinkA=0, shrinkB=0
            ),
        )

    # Non-stationary
    years = [1950, 1975, 2000, 2050, 2075, 2100]
    ax.plot(
        years,
        df_nonstat_sel_main[[f"{col_name}_{year}" for year in years]].to_numpy().flatten() - anomaly_baseline,
        color="indigo",
        ls="dotted",
        label="Non-stationary (1950-2100)",
    )

    ax.fill_between(
        x=years,
        y1=df_nonstat_sel[[f"{col_name}_{year}" for year in years]].quantile(0.975) - anomaly_baseline,
        y2=df_nonstat_sel[[f"{col_name}_{year}" for year in years]].quantile(0.025) - anomaly_baseline,
        color="indigo",
        alpha=0.5,
    )
    y1 = df_nonstat_sel[f"{col_name}_2100"].quantile(0.975) - anomaly_baseline
    y2 = df_nonstat_sel[f"{col_name}_2100"].quantile(0.025) - anomaly_baseline
    ax.annotate(
        "",
        xy=(2106, y1),
        xytext=(2106, y2),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", lw=1.5, color="indigo", shrinkA=0, shrinkB=0),
    )

    # Tidy
    if ylims is not None:
        ax.set_ylim(ylims)

    ax.set_xlabel("")
    ax.set_ylabel(f"{pu.title_labels[metric_id]} anomaly {unit}")
    ax.set_title(
        f"{title_index} GEV fit uncertainty: 100-year return level",
        loc="left",
        fontstyle="italic",
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
