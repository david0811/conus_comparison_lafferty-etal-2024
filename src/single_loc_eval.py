import numpy as np
from scipy.stats import genextreme

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
    ax=None,
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
    - gev_type: Type of GEV fit (stat, nonstat, nonstat_scale)

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
            & (df_fit["ssp"].isin([ssp, "historical"]))
            & (df_fit["member"] == member)
            & (df_fit["ensemble"] == ensemble)
        ]

    # Filter obs
    if gev_type == "stat":
        if ssp == "historical":
            year_min = 1950
            year_max = 2014
        else:
            year_min = 2050
            year_max = 2100

        df_obs_sel = df_obs[
            (df_obs["gcm"] == gcm)
            & (df_obs["ssp"] == ssp)
            & (df_obs["member"] == member)
            & (df_obs["ensemble"] == ensemble)
            & (df_obs["time"] >= year_min)
            & (df_obs["time"] <= year_max)
        ]
    else:
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

    if gev_type == "stat":
        # Main params
        main_loc = main_fit[loc_name]
        main_scale = main_fit[scale_name]
        main_shape = main_fit[shape_name]

        # Main quantiles
        theoretical_quantiles = genextreme.ppf(
            empirical_probs, main_shape, main_loc, main_scale
        )

        # Quantile RMSE
        qq_rmse = np.sqrt(np.mean((theoretical_quantiles - observed_sorted) ** 2))

        # Log likelihood
        log_lik = np.sum(
            genextreme.logpdf(observed_data, main_shape, main_loc, main_scale)
        )

    elif gev_type in ["nonstat", "nonstat_scale"]:
        # Standard Gumbel quantiles
        theoretical_quantiles = -np.log(-np.log(empirical_probs))

        # Main params
        main_loc_intp = main_fit["loc_intcp"]
        main_loc_trend = main_fit["loc_trend"]
        main_locs = main_loc_intp + times * main_loc_trend
        if gev_type == "nonstat_scale":
            main_scale_intp = main_fit["log_scale_intcp"]
            main_scale_trend = main_fit["log_scale_trend"]
            main_scales = np.exp(main_scale_intp + times * main_scale_trend)
        else:
            main_scales = main_fit["scale"]
        main_shape = main_fit["shape"]

        # Transormed variables
        if np.abs(main_shape) < 1e-6:
            z_resids = (observed_data - main_locs) / main_scales
        else:
            z_resids = (
                -1
                / main_shape
                * np.log(1 - main_shape * (observed_data - main_locs) / main_scales)
            )

        z_sorted = np.sort(z_resids)

        # Quantile RMSE
        qq_rmse = np.sqrt(np.mean((theoretical_quantiles - z_sorted) ** 2))

        # Log likelihood
        log_lik = np.sum(
            genextreme.logpdf(
                observed_data,
                main_shape,
                main_locs,
                main_scales,
            )
        )

    # Plot
    if ax is not None:
        if gev_type == "stat":
            ax.scatter(x=observed_sorted, y=theoretical_quantiles, color="C0")
        else:
            ax.scatter(x=theoretical_quantiles, y=z_sorted, color="C0")
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
        ax.grid()

    # Return stats
    # Add after calculating log_lik
    if gev_type == "stat":
        n_params = 3  # loc, scale, shape
    elif gev_type == "nonstat":
        n_params = 4  # loc_intcp, loc_trend, scale, shape
    else:  # nonstat_scale
        n_params = 5  # loc_intcp, loc_trend, scale_intcp, scale_trend, shape

    aic = 2 * n_params - 2 * log_lik
    bic = n_params * np.log(n) - 2 * log_lik
    return qq_rmse, log_lik, aic, bic
