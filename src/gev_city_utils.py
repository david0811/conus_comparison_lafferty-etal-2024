import os

import numpy as np
import pandas as pd

import gev_utils as gevu
import gev_stat_utils as gevsu
import gev_nonstat_trend_utils as gevnst
import gev_nonstat_scale_utils as gevnss
from utils import check_data_length
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path


########################################################
# Fill missing points in the neighbors dataframe
########################################################
def fill_missing_points(df, var_id, required_points=["south", "north", "east", "west"]):
    """
    Check for missing points and fill them by randomly sampling from existing points
    for the same year, maintaining all other column values.

    Parameters:
    df: DataFrame with climate data
    required_points: List of points that should exist

    Returns:
    DataFrame with missing points filled in
    """

    # Make a copy to avoid modifying original
    df_filled = df.copy()

    # Get unique combinations of non-point columns to maintain structure
    grouping_cols = [col for col in df.columns if col not in ["point", var_id]]

    # Check what points currently exist
    existing_points = set(df["point"].unique())
    missing_points = set(required_points) - existing_points

    if not missing_points:
        return df_filled

    # For each missing point, create rows by sampling from existing points
    new_rows = []

    for missing_point in missing_points:
        # Get all unique combinations of the grouping columns
        unique_combinations = df[grouping_cols].drop_duplicates()

        for _, combo in unique_combinations.iterrows():
            # Find rows that match this combination
            mask = True
            for col in grouping_cols:
                mask &= df[col] == combo[col]

            matching_rows = df[mask]

            if len(matching_rows) > 0:
                # Randomly select one row from the matching rows
                sampled_row = matching_rows.sample(n=1).copy()

                # Update the point to the missing point
                sampled_row["point"] = missing_point

                new_rows.append(sampled_row)

    if new_rows:
        # Combine all new rows
        new_data = pd.concat(new_rows, ignore_index=True)

        # Add to original dataframe
        df_filled = pd.concat([df_filled, new_data], ignore_index=True)

    return df_filled


###########################
# Fit GEV to single city
###########################
def fit_gev_city(
    city,
    metric_id,
    ensemble,
    gcm,
    ssp,
    member,
    fit_method,
    stationary,
    include_neighbors=False,
    periods_for_level=[10, 25, 50, 100],
    hist_slice=[1950, 2014],  # only for stationary fits
    proj_slice=[2050, 2100],  # only for stationary fits
    years=[1950, 2100],  # only for non-stationary fits
    return_period_years=[
        1950,
        1975,
        2000,
        2025,
        2050,
        2075,
        2100,
    ],  # only for non-stationary fits
    return_period_diffs=[[1975, 2075]],  # only for non-stationary fits
    nonstationary_scale=False,  # NEW: whether to fit non-stationary scale parameter
    bootstrap=True,
    n_boot_hist=1,
    n_boot_proj=1000,
    return_samples=False,
    project_data_path=project_data_path,
):
    """
    Fits the GEV model to a selected city, ensemble, GCM, member, SSP, and years.

    Parameters:
    -----------
    nonstationary_scale : bool, default False
        If True and stationary=False, fits scale parameter with linear trend.
        Only applicable for non-stationary fits.
    """

    # Validate parameter combinations
    if stationary and nonstationary_scale:
        raise ValueError("nonstationary_scale=True is only valid when stationary=False")

    if include_neighbors and not stationary:
        raise ValueError("Non-stationary fits with neighbors not implemented")

    # Read and select data
    if include_neighbors:
        df = pd.read_csv(
            f"{project_data_path}/metrics/cities/{city}_{metric_id}_neighbors.csv"
        )
    else:
        df = pd.read_csv(f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv")

    # Determine SSP for historical data
    ssp_hist = "historical" if ensemble in ["LOCA2", "TGW"] else ssp

    # Select data based on stationary/non-stationary approach
    if stationary:
        hist_data, proj_data = _select_stationary_data(
            df,
            ensemble,
            gcm,
            member,
            ssp,
            ssp_hist,
            hist_slice,
            proj_slice,
            metric_id,
            include_neighbors,
        )
        data = None
    else:
        data = _select_nonstationary_data(
            df, ensemble, gcm, member, ssp, ssp_hist, years, metric_id
        )
        hist_data, proj_data = None, None

    # Apply scalar for minimum metrics
    agg, var_id = metric_id.split("_")
    scalar = -1.0 if agg == "min" else 1.0

    # Check data length and get expected length
    if stationary:
        expected_length_hist = check_data_length(
            hist_data,
            ensemble,
            gcm,
            "historical",
            hist_slice,
            include_neighbors=include_neighbors,
        )
        expected_length_proj = check_data_length(
            proj_data,
            ensemble,
            gcm,
            ssp,
            proj_slice,
            include_neighbors=include_neighbors,
        )
        expected_length = None
        starting_year = None
    else:
        expected_length = check_data_length(
            data, ensemble, gcm, ssp, years, include_neighbors=include_neighbors
        )
        starting_year = 1970 if expected_length == 131 else 1950
        expected_length_hist, expected_length_proj = None, None

    # Fit GEV parameters
    if stationary:
        hist_params, proj_params, params = _fit_stationary_gev(
            hist_data,
            proj_data,
            scalar,
            expected_length_hist,
            expected_length_proj,
            fit_method,
        )
    else:
        params = _fit_nonstationary_gev(
            data, scalar, expected_length, fit_method, nonstationary_scale
        )
        hist_params, proj_params = None, None

    # Bootstrap if requested
    bootstrap_results = None
    if bootstrap:
        bootstrap_results = _perform_bootstrap(
            stationary,
            nonstationary_scale,
            hist_params,
            proj_params,
            params,
            hist_data,
            proj_data,
            data,
            expected_length,
            starting_year,
            n_boot_hist,
            n_boot_proj,
            fit_method,
            periods_for_level,
            return_period_years,
            return_period_diffs,
        )

    # Calculate return levels
    return_level_results = _calculate_return_levels(
        stationary,
        nonstationary_scale,
        hist_params,
        proj_params,
        params,
        scalar,
        periods_for_level,
        return_period_years,
        return_period_diffs,
        years,
    )

    # Return results
    return _format_results(
        stationary,
        nonstationary_scale,
        return_samples,
        ensemble,
        gcm,
        member,
        ssp,
        hist_params,
        proj_params,
        params,
        return_level_results,
        bootstrap_results,
        periods_for_level,
        return_period_years,
        return_period_diffs,
        n_boot_proj,
        scalar,
        metric_id,
    )


def _select_stationary_data(
    df,
    ensemble,
    gcm,
    member,
    ssp,
    ssp_hist,
    hist_slice,
    proj_slice,
    metric_id,
    include_neighbors,
):
    """Select data for stationary fits."""
    _, var_id = metric_id.split("_")

    df_hist = df[
        (df["ensemble"] == ensemble)
        & (df["gcm"] == gcm)
        & (df["member"] == member)
        & (df["ssp"] == ssp_hist)
        & (df["time"] >= hist_slice[0])
        & (df["time"] <= hist_slice[1])
    ]

    df_proj = df[
        (df["ensemble"] == ensemble)
        & (df["gcm"] == gcm)
        & (df["member"] == member)
        & (df["ssp"] == ssp)
        & (df["time"] >= proj_slice[0])
        & (df["time"] <= proj_slice[1])
    ]

    if include_neighbors:
        df_hist = fill_missing_points(df_hist, var_id)
        df_proj = fill_missing_points(df_proj, var_id)

    return df_hist[var_id].to_numpy(), df_proj[var_id].to_numpy()


def _select_nonstationary_data(
    df, ensemble, gcm, member, ssp, ssp_hist, years, metric_id
):
    """Select data for non-stationary fits."""
    _, var_id = metric_id.split("_")

    df_sel = df[
        (df["ensemble"] == ensemble)
        & (df["gcm"] == gcm)
        & (df["member"] == member)
        & (df["ssp"].isin([ssp_hist, ssp]))
        & (df["time"] >= years[0])
        & (df["time"] <= years[1])
    ]

    return df_sel[var_id].to_numpy()


def _fit_stationary_gev(
    hist_data, proj_data, scalar, expected_length_hist, expected_length_proj, fit_method
):
    """Fit stationary GEV parameters."""
    hist_params = gevsu._fit_gev_1d_stationary(
        data=scalar * hist_data,
        expected_length=expected_length_hist,
        fit_method=fit_method,
    )
    proj_params = gevsu._fit_gev_1d_stationary(
        data=scalar * proj_data,
        expected_length=expected_length_proj,
        fit_method=fit_method,
    )
    return hist_params, proj_params, None


def _fit_nonstationary_gev(
    data, scalar, expected_length, fit_method, nonstationary_scale
):
    """Fit non-stationary GEV parameters."""
    if nonstationary_scale:
        # Fit with both location and scale non-stationary
        params = gevnss._fit_gev_1d_nonstationary(
            data=scalar * data,
            expected_length=expected_length,
            fit_method=fit_method,
        )
    else:
        # Fit with only location non-stationary (original behavior)
        params = gevnst._fit_gev_1d_nonstationary(
            data=scalar * data,
            expected_length=expected_length,
            fit_method=fit_method,
        )
    return params


def _perform_bootstrap(
    stationary,
    nonstationary_scale,
    hist_params,
    proj_params,
    params,
    hist_data,
    proj_data,
    data,
    expected_length,
    starting_year,
    n_boot_hist,
    n_boot_proj,
    fit_method,
    periods_for_level,
    return_period_years,
    return_period_diffs,
):
    """Perform bootstrap sampling."""
    if stationary:
        bootstrap_params_hist, bootstrap_rls_hist = (
            gevsu._gev_parametric_bootstrap_1d_stationary(
                loc=hist_params[0],
                scale=hist_params[1],
                shape=hist_params[2],
                n_data=len(hist_data),
                n_boot=n_boot_hist,
                fit_method=fit_method,
                periods_for_level=periods_for_level,
                return_samples=True,
            )
        )
        bootstrap_params_proj, bootstrap_rls_proj = (
            gevsu._gev_parametric_bootstrap_1d_stationary(
                loc=proj_params[0],
                scale=proj_params[1],
                shape=proj_params[2],
                n_data=len(proj_data),
                n_boot=n_boot_proj,
                fit_method=fit_method,
                periods_for_level=periods_for_level,
                return_samples=True,
            )
        )
        return {
            "hist": (bootstrap_params_hist, bootstrap_rls_hist),
            "proj": (bootstrap_params_proj, bootstrap_rls_proj),
        }
    else:
        if nonstationary_scale:
            bootstrap_params, bootstrap_rls, bootstrap_rl_diffs, bootstrap_rl_chfcs = (
                gevnss._gev_parametric_bootstrap_1d_nonstationary(
                    params=params,
                    expected_length=expected_length,
                    starting_year=starting_year,
                    n_data=len(data),
                    n_boot=n_boot_proj,
                    fit_method=fit_method,
                    periods_for_level=periods_for_level,
                    return_period_years=return_period_years,
                    return_period_diffs=return_period_diffs,
                    return_samples=True,
                )
            )
        else:
            bootstrap_params, bootstrap_rls, bootstrap_rl_diffs, bootstrap_rl_chfcs = (
                gevnst._gev_parametric_bootstrap_1d_nonstationary(
                    params=params,
                    expected_length=expected_length,
                    starting_year=starting_year,
                    n_data=len(data),
                    n_boot=n_boot_proj,
                    fit_method=fit_method,
                    periods_for_level=periods_for_level,
                    return_period_years=return_period_years,
                    return_period_diffs=return_period_diffs,
                    return_samples=True,
                )
            )
        return {
            "nonstat": (
                bootstrap_params,
                bootstrap_rls,
                bootstrap_rl_diffs,
                bootstrap_rl_chfcs,
            )
        }


def _calculate_return_levels(
    stationary,
    nonstationary_scale,
    hist_params,
    proj_params,
    params,
    scalar,
    periods_for_level,
    return_period_years,
    return_period_diffs,
    years,
):
    """Calculate return levels for main estimates."""
    if stationary:
        return_levels_hist_main = scalar * gevu.estimate_return_level(
            np.array(periods_for_level), *hist_params
        )
        return_levels_proj_main = scalar * gevu.estimate_return_level(
            np.array(periods_for_level), *proj_params
        )
        return {"hist": return_levels_hist_main, "proj": return_levels_proj_main}
    else:
        if nonstationary_scale:
            # params = [loc_intercept, loc_trend, log_scale_intercept, log_scale_trend, shape]
            return_levels_main = [
                scalar
                * gevu.estimate_return_level(
                    period,
                    params[0] + params[1] * (return_period_year - years[0]),
                    np.exp(params[2] + params[3] * (return_period_year - years[0])),
                    params[4],
                )
                for period in periods_for_level
                for return_period_year in return_period_years
            ]

            return_level_diffs_main = [
                scalar
                * (
                    gevu.estimate_return_level(
                        period,
                        params[0] + params[1] * (return_period_diff[1] - years[0]),
                        np.exp(
                            params[2] + params[3] * (return_period_diff[1] - years[0])
                        ),
                        params[4],
                    )
                )
                - scalar
                * (
                    gevu.estimate_return_level(
                        period,
                        params[0] + params[1] * (return_period_diff[0] - years[0]),
                        np.exp(
                            params[2] + params[3] * (return_period_diff[0] - years[0])
                        ),
                        params[4],
                    )
                )
                for period in periods_for_level
                for return_period_diff in return_period_diffs
            ]

            return_level_chfcs_main = [
                (
                    scalar
                    * gevu.estimate_return_level(
                        period,
                        params[0] + params[1] * (return_period_diff[1] - years[0]),
                        np.exp(
                            params[2] + params[3] * (return_period_diff[1] - years[0])
                        ),
                        params[4],
                    )
                )
                / (
                    scalar
                    * gevu.estimate_return_level(
                        period,
                        params[0] + params[1] * (return_period_diff[0] - years[0]),
                        np.exp(
                            params[2] + params[3] * (return_period_diff[0] - years[0])
                        ),
                        params[4],
                    )
                )
                for period in periods_for_level
                for return_period_diff in return_period_diffs
            ]
        else:
            # Original non-stationary location only: params = [loc_intercept, loc_trend, scale, shape]
            return_levels_main = [
                scalar
                * gevu.estimate_return_level(
                    period,
                    params[0] + params[1] * (return_period_year - years[0]),
                    params[2],
                    params[3],
                )
                for period in periods_for_level
                for return_period_year in return_period_years
            ]

            return_level_diffs_main = [
                scalar
                * (
                    gevu.estimate_return_level(
                        period,
                        params[0] + params[1] * (return_period_diff[1] - years[0]),
                        params[2],
                        params[3],
                    )
                )
                - scalar
                * (
                    gevu.estimate_return_level(
                        period,
                        params[0] + params[1] * (return_period_diff[0] - years[0]),
                        params[2],
                        params[3],
                    )
                )
                for period in periods_for_level
                for return_period_diff in return_period_diffs
            ]

            return_level_chfcs_main = [
                (
                    scalar
                    * gevu.estimate_return_level(
                        period,
                        params[0] + params[1] * (return_period_diff[1] - years[0]),
                        params[2],
                        params[3],
                    )
                )
                / (
                    scalar
                    * gevu.estimate_return_level(
                        period,
                        params[0] + params[1] * (return_period_diff[0] - years[0]),
                        params[2],
                        params[3],
                    )
                )
                for period in periods_for_level
                for return_period_diff in return_period_diffs
            ]

        return {
            "main": return_levels_main,
            "diffs": return_level_diffs_main,
            "chfcs": return_level_chfcs_main,
        }


def _format_results(
    stationary,
    nonstationary_scale,
    return_samples,
    ensemble,
    gcm,
    member,
    ssp,
    hist_params,
    proj_params,
    params,
    return_level_results,
    bootstrap_results,
    periods_for_level,
    return_period_years,
    return_period_diffs,
    n_boot_proj,
    scalar,
    metric_id,
):
    """Format and return final results."""
    if return_samples:
        return _format_sample_results(
            stationary,
            nonstationary_scale,
            ensemble,
            gcm,
            member,
            ssp,
            hist_params,
            proj_params,
            params,
            return_level_results,
            bootstrap_results,
            periods_for_level,
            return_period_years,
            return_period_diffs,
            n_boot_proj,
            scalar,
        )
    else:
        return _format_summary_results(
            stationary,
            nonstationary_scale,
            ensemble,
            gcm,
            member,
            ssp,
            hist_params,
            proj_params,
            params,
            return_level_results,
            bootstrap_results,
            periods_for_level,
            return_period_years,
            return_period_diffs,
            scalar,
            metric_id,
        )


def _format_sample_results(
    stationary,
    nonstationary_scale,
    ensemble,
    gcm,
    member,
    ssp,
    hist_params,
    proj_params,
    params,
    return_level_results,
    bootstrap_results,
    periods_for_level,
    return_period_years,
    return_period_diffs,
    n_boot_proj,
    scalar,
):
    """Format results when return_samples=True."""
    if stationary:
        bootstrap_params_hist, bootstrap_rls_hist = bootstrap_results["hist"]
        bootstrap_params_proj, bootstrap_rls_proj = bootstrap_results["proj"]

        # Need to repeat historical results for each bootstrap sample if n_boot_hist = 1
        if bootstrap_params_hist.shape[0] == 1:
            bootstrap_params_hist = np.repeat(
                bootstrap_params_hist, n_boot_proj, axis=0
            )
            bootstrap_rls_hist = np.repeat(bootstrap_rls_hist, n_boot_proj, axis=0)

        df_res = pd.DataFrame(
            {
                "ensemble": [ensemble],
                "gcm": [gcm],
                "member": [member],
                "ssp": [ssp],
                "n_boot": ["main"],
                "loc_hist": [hist_params[0]],
                "scale_hist": [hist_params[1]],
                "shape_hist": [hist_params[2]],
                "loc_proj": [proj_params[0]],
                "scale_proj": [proj_params[1]],
                "shape_proj": [proj_params[2]],
                **{
                    f"{period}yr_return_level_hist": [return_level_results["hist"][i]]
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_proj": [return_level_results["proj"][i]]
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_diff": [
                        return_level_results["proj"][i]
                        - return_level_results["hist"][i]
                    ]
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_chfc": [
                        return_level_results["proj"][i]
                        / return_level_results["hist"][i]
                    ]
                    for i, period in enumerate(periods_for_level)
                },
            }
        )

        df_res_boot = pd.DataFrame(
            {
                "ensemble": ensemble,
                "gcm": gcm,
                "member": member,
                "ssp": ssp,
                "n_boot": np.arange(n_boot_proj),
                "loc_hist": bootstrap_params_hist[:, 0],
                "scale_hist": bootstrap_params_hist[:, 1],
                "shape_hist": bootstrap_params_hist[:, 2],
                "loc_proj": bootstrap_params_proj[:, 0],
                "scale_proj": bootstrap_params_proj[:, 1],
                "shape_proj": bootstrap_params_proj[:, 2],
                **{
                    f"{period}yr_return_level_hist": scalar * bootstrap_rls_hist[:, i]
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_proj": scalar * bootstrap_rls_proj[:, i]
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_diff": (scalar * bootstrap_rls_proj[:, i])
                    - (scalar * bootstrap_rls_hist[:, i])
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_chfc": (scalar * bootstrap_rls_proj[:, i])
                    / (scalar * bootstrap_rls_hist[:, i])
                    for i, period in enumerate(periods_for_level)
                },
            }
        )

        return pd.concat([df_res, df_res_boot], ignore_index=True)

    else:
        # Non-stationary case
        bootstrap_params, bootstrap_rls, bootstrap_rl_diffs, bootstrap_rl_chfcs = (
            bootstrap_results["nonstat"]
        )

        if nonstationary_scale:
            # 5 parameters: loc_intcp, loc_trend, scale_intcp, scale_trend, shape
            df_res = pd.DataFrame(
                {
                    "ensemble": [ensemble],
                    "gcm": [gcm],
                    "member": [member],
                    "ssp": [ssp],
                    "n_boot": ["main"],
                    "loc_intcp": [params[0]],
                    "loc_trend": [params[1]],
                    "log_scale_intcp": [params[2]],
                    "log_scale_trend": [params[3]],
                    "shape": [params[4]],
                    **{
                        f"{period}yr_return_level_{return_period_year}": return_level_results[
                            "main"
                        ][i_period * len(return_period_years) + i_year]
                        for i_period, period in enumerate(periods_for_level)
                        for i_year, return_period_year in enumerate(return_period_years)
                    },
                    **{
                        f"{period}yr_return_level_diff_{return_period_diff[1]}-{return_period_diff[0]}": return_level_results[
                            "diffs"
                        ][i_period * len(return_period_diffs) + i_diff]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                    **{
                        f"{period}yr_return_level_chfc_{return_period_diff[1]}-{return_period_diff[0]}": return_level_results[
                            "chfcs"
                        ][i_period * len(return_period_diffs) + i_diff]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                }
            )

            df_res_boot = pd.DataFrame(
                {
                    "ensemble": ensemble,
                    "gcm": gcm,
                    "member": member,
                    "ssp": ssp,
                    "n_boot": np.arange(n_boot_proj),
                    "loc_intcp": bootstrap_params[:, 0],
                    "loc_trend": bootstrap_params[:, 1],
                    "log_scale_intcp": bootstrap_params[:, 2],
                    "log_scale_trend": bootstrap_params[:, 3],
                    "shape": bootstrap_params[:, 4],
                    **{
                        f"{period}yr_return_level_{return_period_year}": scalar
                        * bootstrap_rls[:, i_period * len(return_period_years) + i_year]
                        for i_period, period in enumerate(periods_for_level)
                        for i_year, return_period_year in enumerate(return_period_years)
                    },
                    **{
                        f"{period}yr_return_level_diff_{return_period_diff[1]}-{return_period_diff[0]}": scalar
                        * bootstrap_rl_diffs[
                            :, i_period * len(return_period_diffs) + i_diff
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                    **{
                        f"{period}yr_return_level_chfc_{return_period_diff[1]}-{return_period_diff[0]}": scalar
                        * bootstrap_rl_chfcs[
                            :, i_period * len(return_period_diffs) + i_diff
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                }
            )
        else:
            # Original 4 parameters: loc_intcp, loc_trend, scale, shape
            df_res = pd.DataFrame(
                {
                    "ensemble": [ensemble],
                    "gcm": [gcm],
                    "member": [member],
                    "ssp": [ssp],
                    "n_boot": ["main"],
                    "loc_intcp": [params[0]],
                    "loc_trend": [params[1]],
                    "scale": [params[2]],
                    "shape": [params[3]],
                    **{
                        f"{period}yr_return_level_{return_period_year}": return_level_results[
                            "main"
                        ][i_period * len(return_period_years) + i_year]
                        for i_period, period in enumerate(periods_for_level)
                        for i_year, return_period_year in enumerate(return_period_years)
                    },
                    **{
                        f"{period}yr_return_level_diff_{return_period_diff[1]}-{return_period_diff[0]}": return_level_results[
                            "diffs"
                        ][i_period * len(return_period_diffs) + i_diff]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                    **{
                        f"{period}yr_return_level_chfc_{return_period_diff[1]}-{return_period_diff[0]}": return_level_results[
                            "chfcs"
                        ][i_period * len(return_period_diffs) + i_diff]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                }
            )

            df_res_boot = pd.DataFrame(
                {
                    "ensemble": ensemble,
                    "gcm": gcm,
                    "member": member,
                    "ssp": ssp,
                    "n_boot": np.arange(n_boot_proj),
                    "loc_intcp": bootstrap_params[:, 0],
                    "loc_trend": bootstrap_params[:, 1],
                    "scale": bootstrap_params[:, 2],
                    "shape": bootstrap_params[:, 3],
                    **{
                        f"{period}yr_return_level_{return_period_year}": scalar
                        * bootstrap_rls[:, i_period * len(return_period_years) + i_year]
                        for i_period, period in enumerate(periods_for_level)
                        for i_year, return_period_year in enumerate(return_period_years)
                    },
                    **{
                        f"{period}yr_return_level_diff_{return_period_diff[1]}-{return_period_diff[0]}": scalar
                        * bootstrap_rl_diffs[
                            :, i_period * len(return_period_diffs) + i_diff
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                    **{
                        f"{period}yr_return_level_chfc_{return_period_diff[1]}-{return_period_diff[0]}": scalar
                        * bootstrap_rl_chfcs[
                            :, i_period * len(return_period_diffs) + i_diff
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                }
            )

        return pd.concat([df_res, df_res_boot], ignore_index=True)


def _format_summary_results(
    stationary,
    nonstationary_scale,
    ensemble,
    gcm,
    member,
    ssp,
    hist_params,
    proj_params,
    params,
    return_level_results,
    bootstrap_results,
    periods_for_level,
    return_period_years,
    return_period_diffs,
    scalar,
    metric_id,
):
    """Format results when return_samples=False."""
    if stationary:
        bootstrap_params_hist, bootstrap_rls_hist = bootstrap_results["hist"]
        bootstrap_params_proj, bootstrap_rls_proj = bootstrap_results["proj"]

        # Calculate percentiles
        hist_params_q025 = np.percentile(bootstrap_params_hist, 2.5, axis=0)
        hist_params_q975 = np.percentile(bootstrap_params_hist, 97.5, axis=0)
        return_levels_hist_q025 = np.percentile(
            scalar * bootstrap_rls_hist, 2.5, axis=0
        )
        return_levels_hist_q975 = np.percentile(
            scalar * bootstrap_rls_hist, 97.5, axis=0
        )

        proj_params_q025 = np.percentile(bootstrap_params_proj, 2.5, axis=0)
        proj_params_q975 = np.percentile(bootstrap_params_proj, 97.5, axis=0)
        return_levels_proj_q025 = np.percentile(
            scalar * bootstrap_rls_proj, 2.5, axis=0
        )
        return_levels_proj_q975 = np.percentile(
            scalar * bootstrap_rls_proj, 97.5, axis=0
        )

        # Differences and change factors
        return_levels_diff_main = (
            return_level_results["proj"] - return_level_results["hist"]
        )
        return_levels_diff_q025 = np.nanpercentile(
            (scalar * bootstrap_rls_proj) - (scalar * bootstrap_rls_hist), 2.5, axis=0
        )
        return_levels_diff_q975 = np.nanpercentile(
            (scalar * bootstrap_rls_proj) - (scalar * bootstrap_rls_hist), 97.5, axis=0
        )
        return_levels_chfc_main = (
            return_level_results["proj"] / return_level_results["hist"]
        )
        return_levels_chfc_q025 = np.nanpercentile(
            (scalar * bootstrap_rls_proj) / (scalar * bootstrap_rls_hist), 2.5, axis=0
        )
        return_levels_chfc_q975 = np.nanpercentile(
            (scalar * bootstrap_rls_proj) / (scalar * bootstrap_rls_hist), 97.5, axis=0
        )

        # Create output DataFrame
        df_out = pd.DataFrame(
            {
                "quantile": ["main", "q025", "q975"],
                "ensemble": [ensemble, ensemble, ensemble],
                "gcm": [gcm, gcm, gcm],
                "member": [member, member, member],
                "ssp": [ssp, ssp, ssp],
                "loc_hist": [hist_params[0], hist_params_q025[0], hist_params_q975[0]],
                "scale_hist": [
                    hist_params[1],
                    hist_params_q025[1],
                    hist_params_q975[1],
                ],
                "shape_hist": [
                    hist_params[2],
                    hist_params_q025[2],
                    hist_params_q975[2],
                ],
                "loc_proj": [proj_params[0], proj_params_q025[0], proj_params_q975[0]],
                "scale_proj": [
                    proj_params[1],
                    proj_params_q025[1],
                    proj_params_q975[1],
                ],
                "shape_proj": [
                    proj_params[2],
                    proj_params_q025[2],
                    proj_params_q975[2],
                ],
                **{
                    f"{period}yr_return_level_hist": [
                        return_level_results["hist"][i],
                        return_levels_hist_q025[i],
                        return_levels_hist_q975[i],
                    ]
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_proj": [
                        return_level_results["proj"][i],
                        return_levels_proj_q025[i],
                        return_levels_proj_q975[i],
                    ]
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_diff": [
                        return_levels_diff_main[i],
                        return_levels_diff_q025[i],
                        return_levels_diff_q975[i],
                    ]
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_chfc": [
                        return_levels_chfc_main[i],
                        return_levels_chfc_q025[i],
                        return_levels_chfc_q975[i],
                    ]
                    for i, period in enumerate(periods_for_level)
                },
            }
        )

        # Drop change factors for tasmin since they are meaningless
        if metric_id == "min_tasmin":
            df_out = df_out.drop(
                columns=[col for col in df_out.columns if "chfc" in col]
            )

        return df_out

    else:
        # Non-stationary case
        bootstrap_params, bootstrap_rls, bootstrap_rl_diffs, bootstrap_rl_chfcs = (
            bootstrap_results["nonstat"]
        )

        # Calculate percentiles for parameters
        params_q025 = np.nanpercentile(bootstrap_params, 2.5, axis=0)
        params_q975 = np.nanpercentile(bootstrap_params, 97.5, axis=0)

        # Calculate percentiles for return levels
        return_levels_q025 = np.nanpercentile(scalar * bootstrap_rls, 2.5, axis=0)
        return_levels_q975 = np.nanpercentile(scalar * bootstrap_rls, 97.5, axis=0)

        # Calculate percentiles for return level differences
        return_level_diffs_q025 = np.nanpercentile(
            scalar * bootstrap_rl_diffs, 2.5, axis=0
        )
        return_level_diffs_q975 = np.nanpercentile(
            scalar * bootstrap_rl_diffs, 97.5, axis=0
        )

        # Calculate percentiles for return level change factors
        return_level_chfcs_q025 = np.nanpercentile(
            scalar * bootstrap_rl_chfcs, 2.5, axis=0
        )
        return_level_chfcs_q975 = np.nanpercentile(
            scalar * bootstrap_rl_chfcs, 97.5, axis=0
        )

        if nonstationary_scale:
            # 5 parameters: loc_intcp, loc_trend, scale_intcp, scale_trend, shape
            df_out = pd.DataFrame(
                {
                    "quantile": ["main", "q025", "q975"],
                    "ensemble": [ensemble, ensemble, ensemble],
                    "gcm": [gcm, gcm, gcm],
                    "member": [member, member, member],
                    "ssp": [ssp, ssp, ssp],
                    "loc_intcp": [params[0], params_q025[0], params_q975[0]],
                    "loc_trend": [params[1], params_q025[1], params_q975[1]],
                    "log_scale_intcp": [params[2], params_q025[2], params_q975[2]],
                    "log_scale_trend": [params[3], params_q025[3], params_q975[3]],
                    "shape": [params[4], params_q025[4], params_q975[4]],
                    **{
                        f"{period}yr_return_level_{return_period_year}": [
                            return_level_results["main"][
                                i_period * len(return_period_years) + i_year
                            ],
                            return_levels_q025[
                                i_period * len(return_period_years) + i_year
                            ],
                            return_levels_q975[
                                i_period * len(return_period_years) + i_year
                            ],
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_year, return_period_year in enumerate(return_period_years)
                    },
                    **{
                        f"{period}yr_return_level_diff_{return_period_diff[1]}-{return_period_diff[0]}": [
                            return_level_results["diffs"][
                                i_period * len(return_period_diffs) + i_diff
                            ],
                            return_level_diffs_q025[
                                i_period * len(return_period_diffs) + i_diff
                            ],
                            return_level_diffs_q975[
                                i_period * len(return_period_diffs) + i_diff
                            ],
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                    **{
                        f"{period}yr_return_level_chfc_{return_period_diff[1]}-{return_period_diff[0]}": [
                            return_level_results["chfcs"][
                                i_period * len(return_period_diffs) + i_diff
                            ],
                            return_level_chfcs_q025[
                                i_period * len(return_period_diffs) + i_diff
                            ],
                            return_level_chfcs_q975[
                                i_period * len(return_period_diffs) + i_diff
                            ],
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                }
            )
        else:
            # Original 4 parameters: loc_intcp, loc_trend, scale, shape
            df_out = pd.DataFrame(
                {
                    "quantile": ["main", "q025", "q975"],
                    "ensemble": [ensemble, ensemble, ensemble],
                    "gcm": [gcm, gcm, gcm],
                    "member": [member, member, member],
                    "ssp": [ssp, ssp, ssp],
                    "loc_intcp": [params[0], params_q025[0], params_q975[0]],
                    "loc_trend": [params[1], params_q025[1], params_q975[1]],
                    "scale": [params[2], params_q025[2], params_q975[2]],
                    "shape": [params[3], params_q025[3], params_q975[3]],
                    **{
                        f"{period}yr_return_level_{return_period_year}": [
                            return_level_results["main"][
                                i_period * len(return_period_years) + i_year
                            ],
                            return_levels_q025[
                                i_period * len(return_period_years) + i_year
                            ],
                            return_levels_q975[
                                i_period * len(return_period_years) + i_year
                            ],
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_year, return_period_year in enumerate(return_period_years)
                    },
                    **{
                        f"{period}yr_return_level_diff_{return_period_diff[1]}-{return_period_diff[0]}": [
                            return_level_results["diffs"][
                                i_period * len(return_period_diffs) + i_diff
                            ],
                            return_level_diffs_q025[
                                i_period * len(return_period_diffs) + i_diff
                            ],
                            return_level_diffs_q975[
                                i_period * len(return_period_diffs) + i_diff
                            ],
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                    **{
                        f"{period}yr_return_level_chfc_{return_period_diff[1]}-{return_period_diff[0]}": [
                            return_level_results["chfcs"][
                                i_period * len(return_period_diffs) + i_diff
                            ],
                            return_level_chfcs_q025[
                                i_period * len(return_period_diffs) + i_diff
                            ],
                            return_level_chfcs_q975[
                                i_period * len(return_period_diffs) + i_diff
                            ],
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                }
            )

        # Drop change factors for tasmin since they are meaningless
        if metric_id == "min_tasmin":
            df_out = df_out.drop(
                columns=[col for col in df_out.columns if "chfc" in col]
            )

        return df_out


def fit_ensemble_gev_city(
    city,
    metric_id,
    stationary,
    fit_method,
    nonstationary_scale=False,
    include_neighbors=False,
    periods_for_level=[10, 25, 50, 100],
    bootstrap="parametric",
    n_boot_proj=1000,
    n_boot_hist=1,
    hist_slice=[1950, 2014],  # only for stationary fits
    proj_slice=[2050, 2100],  # only for stationary fits
    years=[1950, 2100],  # only for non-stationary fits
    return_period_years=[
        1950,
        1975,
        2000,
        2025,
        2050,
        2075,
        2100,
    ],  # only for non-stationary fits
    return_period_diffs=[[1975, 2075]],  # only for non-stationary fits
    store=True,
    return_samples=False,
    project_data_path=project_data_path,
):
    """
    Fits GEV (Generalized Extreme Value) distributions across an entire climate model ensemble for a specific city.

    This function processes all available ensemble members, GCMs, and scenarios for a given city and metric,
    fitting either stationary or non-stationary GEV distributions with optional bootstrapping for uncertainty estimation.

    Parameters
    ----------
    city : str
        Name of the city to analyze
    metric_id : str
        Identifier for the climate metric to analyze (e.g., 'max_tasmax' for maximum temperature)
    stationary : bool
        If True, fits a stationary GEV. If False, fits a non-stationary GEV with temporal trend
    fit_method : str
        Method used for fitting the GEV distribution
    nonstationary_scale : bool, optional
        If True, fits a non-stationary GEV with temporal trend in the scale parameter
    include_neighbors : bool, optional
        If True, include neighbors in the fit
    periods_for_level : list
        Return periods (in years) for which to calculate return levels
    bootstrap : str, optional
        Type of bootstrap method, defaults to "parametric"
    n_boot : int, optional
        Number of bootstrap samples, defaults to 1000
    hist_slice : list, optional
        [start_year, end_year] for historical period, defaults to [1950, 2014]
    proj_slice : list, optional
        [start_year, end_year] for projection period, defaults to [2050, 2100]
    years : list, optional
        Years to use for non-stationary fit
    return_period_years : list, optional
        Years at which to calculate return levels for non-stationary fits
    return_period_diffs : list, optional
        Year pairs for calculating return level differences in non-stationary fits
    store : bool, optional
        If True, saves results to file; if False, returns results DataFrame, defaults to True
    return_samples: bool, optional
        If True, return individual bootstrap samples
    project_data_path : str, optional
        Base path for project data files

    Returns
    -------
    pandas.DataFrame or None
        If store=False, returns DataFrame with GEV fit results for all ensemble members.
        If store=True, saves results to CSV and returns None.
        Returns None if results file already exists.
    """

    # Get unique combos
    if include_neighbors:
        df = pd.read_csv(
            f"{project_data_path}/metrics/cities/{city}_{metric_id}_neighbors.csv"
        )
    else:
        df = pd.read_csv(f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv")
    df = df.set_index(["ensemble", "gcm", "member", "ssp"]).sort_index()
    combos = df.index.unique()

    # Check if done
    stat_str = "stat" if stationary else "nonstat"
    sample_str = "_samples" if return_samples else ""
    neighbor_str = "_neighbors" if include_neighbors else ""
    scale_str = "_scale" if nonstationary_scale else ""
    if stationary:
        file_name = f"{city}_{metric_id}_{hist_slice[0]}-{hist_slice[1]}_{proj_slice[0]}-{proj_slice[1]}_{fit_method}_{stat_str}_nbootproj{n_boot_proj}_nboothist{n_boot_hist}{sample_str}{neighbor_str}.csv"
    else:
        file_name = f"{city}_{metric_id}_{years[0]}-{years[1]}_{fit_method}_{stat_str}_nboot{n_boot_proj}{sample_str}{neighbor_str}{scale_str}.csv"

    if os.path.exists(
        f"{project_data_path}/extreme_value/cities/original_grid/freq/{file_name}"
    ):
        return None

    # Output df
    df_out = []

    # Loop through
    for combo in combos:
        ensemble, gcm, member, ssp = combo
        if ssp == "historical":
            continue
        if ensemble == "STAR-ESDM" and gcm == "TaiESM1":
            continue  # Skip recalled outputs
        try:
            df_tmp = fit_gev_city(
                city=city,
                metric_id=metric_id,
                ensemble=ensemble,
                gcm=gcm,
                ssp=ssp,
                member=member,
                nonstationary_scale=nonstationary_scale,
                hist_slice=hist_slice,
                proj_slice=proj_slice,
                fit_method=fit_method,
                include_neighbors=include_neighbors,
                periods_for_level=periods_for_level,
                stationary=stationary,
                years=years,
                bootstrap=bootstrap,
                n_boot_hist=n_boot_hist,
                n_boot_proj=n_boot_proj,
                return_period_years=return_period_years,
                return_period_diffs=return_period_diffs,
                return_samples=return_samples,
            )
            df_out.append(df_tmp)
        except Exception as e:
            except_path = f"{project_code_path}/scripts/logs/gev_freq/city"
            with open(
                f"{except_path}/{city}_{ensemble}_{gcm}_{member}_{ssp}_{metric_id}_{stat_str}_{fit_method}.txt",
                "w",
            ) as f:
                f.write(str(e))

    # Concat
    df_out = pd.concat(df_out, ignore_index=True)

    # Store or return
    if store:
        df_out.to_csv(
            f"{project_data_path}/extreme_value/cities/original_grid/freq/{file_name}",
            index=False,
        )
    else:
        return df_out
