import numpy as np
from scipy.stats import genextreme
import matplotlib.pyplot as plt
from plotting_utils import gev_labels


def gev_qq_plot(
    df_fit,
    df_obs,
    metric_id,
    gcm,
    ssp,
    member,
    ensemble,
    gev_type,
    confidence_level=0.95,
    make_plot=True,
):
    """
    Plot a GEV QQ plot for a single location with correct confidence intervals.

    The confidence bands are horizontal error bars around theoretical quantiles,
    representing the uncertainty in what theoretical quantile each observed value
    corresponds to, given the uncertainty in fitted parameters.

    Inputs:
    - df_fit: DataFrame with fitted GEV parameters
    - df_obs: DataFrame with observed data
    - metric_id: Metric ID
    - gev_type: Type of GEV fit (stat, nonstat, nonstat-scale)

    Outputs:
    - Dictionary with results
    """
    # Filter for climate data
    if gev_type == "stat":
        if ssp == "historical":
            # Historical stat fits are the same for all SSPs
            df_fit_sel = df_fit[
                (df_fit["gcm"] == gcm)
                & (df_fit["ssp"] == "ssp370")
                & (df_fit["member"] == member)
                & (df_fit["ensemble"] == ensemble)
            ]
            loc_name = "loc_hist"
            scale_name = "scale_hist"
            shape_name = "shape_hist"
        else:
            df_fit_sel = df_fit[
                (df_fit["gcm"] == gcm)
                & (df_fit["ssp"] == ssp)
                & (df_fit["member"] == member)
                & (df_fit["ensemble"] == ensemble)
            ]
            loc_name = "loc_proj"
            scale_name = "scale_proj"
            shape_name = "shape_proj"
    else:
        df_fit_sel = df_fit[
            (df_fit["gcm"] == gcm)
            & (df_fit["ssp"] == ssp)
            & (df_fit["member"] == member)
            & (df_fit["ensemble"] == ensemble)
        ]

    # Filter obs
    df_obs_sel = df_obs[
        (df_obs["gcm"] == gcm)
        & (df_obs["ssp"].isin([ssp, "historical"]))
        & (df_obs["member"] == member)
        & (df_obs["ensemble"] == ensemble)
    ]

    # Get empirical quantiles
    var_id = metric_id.split("_")[-1]
    observed_data = df_obs_sel.sort_values(by="time")[var_id].to_numpy()
    times = np.sort(df_obs_sel["time"].to_numpy() - np.min(df_obs_sel["time"]))

    # Invert for minima
    if metric_id == "min_tasmin":
        observed_data = -observed_data

    observed_sorted = np.sort(observed_data)
    n = len(observed_sorted)

    empirical_probs = (np.arange(n) + 1) / (n + 1)

    # Get main fit and bootstrap fits
    main_fit = df_fit_sel[df_fit_sel["n_boot"] == "main"].iloc[0]
    bootstrap_fits = df_fit_sel[df_fit_sel["n_boot"] != "main"]

    if gev_type == "stat":
        # Main params
        main_loc = main_fit[loc_name]
        main_scale = main_fit[scale_name]
        main_shape = main_fit[shape_name]

        # Bootstrap params
        bootstrap_loc = bootstrap_fits[loc_name]
        bootstrap_scale = bootstrap_fits[scale_name]
        bootstrap_shape = bootstrap_fits[shape_name]

        # Main quantiles
        theoretical_quantiles = genextreme.ppf(
            empirical_probs, main_shape, main_loc, main_scale
        )
        # Bootstrap quantiles
        bootstrap_theoretical_quantiles = np.zeros((len(bootstrap_fits), n))

        for i in range(len(bootstrap_fits)):
            # Convert these probabilities to theoretical quantiles using main fit
            bootstrap_theoretical_quantiles[i, :] = genextreme.ppf(
                empirical_probs,
                bootstrap_shape.iloc[i],
                bootstrap_loc.iloc[i],
                bootstrap_scale.iloc[i],
            )

        # Get fit statistic
        alpha = 1 - confidence_level
        lower = np.nanpercentile(bootstrap_theoretical_quantiles, alpha / 2, axis=0)
        upper = np.nanpercentile(bootstrap_theoretical_quantiles, 1 - alpha / 2, axis=0)
        frac_within_band = np.mean(
            (observed_sorted >= lower) * (observed_sorted <= upper)
        )

    elif gev_type == "nonstat":
        # Standard Gumbel quantiles
        theoretical_quantiles = -np.log(-np.log(empirical_probs))

        # Main params
        main_loc_intp = main_fit["loc_intcp"]
        main_loc_trend = main_fit["loc_trend"]
        main_locs = main_loc_intp + times * main_loc_trend
        main_scale = main_fit["scale"]
        main_shape = main_fit["shape"]

        # Bootstrap params
        bootstrap_loc_intcp = bootstrap_fits["loc_intcp"].to_numpy()
        bootstrap_loc_trend = bootstrap_fits["loc_trend"].to_numpy()
        bootstrap_scale = bootstrap_fits["scale"].to_numpy()
        bootstrap_shape = bootstrap_fits["shape"].to_numpy()

        # Transormed variables
        z_resids = (
            -1
            / main_shape
            * np.log(1 - main_shape * (observed_data - main_locs) / main_scale)
        )
        z_sorted = np.sort(z_resids)

        # Bootstrap transformed variables
        bootstrap_z_resids = np.zeros((len(bootstrap_fits), n))
        for i in range(len(bootstrap_fits)):
            # For each bootstrap iteration b
            bootstrap_locs_b = bootstrap_loc_intcp[i] + times * bootstrap_loc_trend[i]
            bootstrap_scale_b = bootstrap_scale[i]
            bootstrap_shape_b = bootstrap_shape[i]

            # Transform original data with bootstrap parameters
            z_resids_b = (
                -1
                / bootstrap_shape_b
                * np.log(
                    1
                    - bootstrap_shape_b
                    * (observed_data - bootstrap_locs_b)
                    / bootstrap_scale_b
                )
            )
            z_sorted_b = np.sort(z_resids_b)
            bootstrap_z_resids[i, :] = z_sorted_b

        # Get fit statistic
        alpha = 1 - confidence_level
        lower = np.nanpercentile(bootstrap_z_resids, alpha / 2, axis=0)
        upper = np.nanpercentile(bootstrap_z_resids, 1 - alpha / 2, axis=0)
        frac_within_band = np.mean(
            (theoretical_quantiles >= lower) * (theoretical_quantiles <= upper)
        )

    if make_plot:
        fig, ax = plt.subplots()
        if gev_type == "stat":
            ax.scatter(x=observed_sorted, y=theoretical_quantiles, color="C0")
            ax.errorbar(
                x=observed_sorted,
                y=theoretical_quantiles,
                yerr=[theoretical_quantiles - lower, upper - theoretical_quantiles],
                color="C0",
                linestyle="none",
            )
        elif gev_type == "nonstat":
            ax.scatter(x=theoretical_quantiles, y=z_sorted, color="C0")
            ax.errorbar(
                x=theoretical_quantiles,
                y=z_sorted,
                yerr=[z_sorted - lower, upper - z_sorted],
                color="C0",
                linestyle="none",
            )
        ylims = ax.get_ylim()
        xlims = ax.get_xlim()
        ax.axline((1, 1), slope=1, color="C1", ls="--")
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)
        if gev_type == "stat":
            ax.set_ylabel(f"Theoretical quantile {gev_labels[metric_id]}")
            ax.set_xlabel(f"Observed quantile {gev_labels[metric_id]}")
        else:
            ax.set_xlabel("Theoretical Gumbel quantile")
            ax.set_ylabel("Empirical quantile")
        ax.set_title(f"{100 * frac_within_band:.2f}% coverage")
        ax.grid()
        plt.show()

    return frac_within_band
