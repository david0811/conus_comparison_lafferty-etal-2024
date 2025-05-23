import os

import numpy as np
import pandas as pd

import gev_utils as gevu
import gev_stat_utils as gevsu
import gev_nonstat_utils as gevnsu
from utils import check_data_length
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path


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
    bootstrap=True,
    n_boot_hist=1,
    n_boot_proj=1000,
    return_samples=False,
    project_data_path=project_data_path,
):
    """
    Fits the GEV model to a selected city, ensemble, GCM, member, SSP, and years.
    """

    # Read and select data
    df = pd.read_csv(f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv")

    if ensemble in ["LOCA2", "TGW"]:
        ssp_hist = "historical"
    else:
        ssp_hist = ssp

    if stationary:
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
    else:
        df_sel = df[
            (df["ensemble"] == ensemble)
            & (df["gcm"] == gcm)
            & (df["member"] == member)
            & (df["ssp"].isin([ssp_hist, ssp]))
            & (df["time"] >= years[0])
            & (df["time"] <= years[1])
        ]

    # Info
    agg, var_id = metric_id.split("_")

    if stationary:
        hist_data = df_hist[var_id].to_numpy()
        proj_data = df_proj[var_id].to_numpy()
    else:
        data = df_sel[var_id].to_numpy()

    if agg == "min":
        scalar = -1.0
    else:
        scalar = 1.0

    # Check length is as expected
    if stationary:
        expected_length_hist = check_data_length(
            hist_data, ensemble, gcm, "historical", hist_slice
        )
        expected_length_proj = check_data_length(
            proj_data, ensemble, gcm, ssp, proj_slice
        )
    else:
        expected_length = check_data_length(data, ensemble, gcm, ssp, years)
        starting_year = 1970 if expected_length == 131 else 1950

    # Do the fit
    if stationary:
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
    else:
        params = gevnsu._fit_gev_1d_nonstationary(
            data=scalar * data,
            expected_length=expected_length,
            fit_method=fit_method,
        )

    # Do the bootstrap if desried
    if bootstrap:
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
        else:
            bootstrap_params, bootstrap_rls, bootstrap_rl_diffs, bootstrap_rl_chfcs = (
                gevnsu._gev_parametric_bootstrap_1d_nonstationary(
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

    # Calculate return levels
    if stationary:
        return_levels_hist_main = scalar * gevu.estimate_return_level(
            np.array(periods_for_level), *hist_params
        )
        return_levels_proj_main = scalar * gevu.estimate_return_level(
            np.array(periods_for_level), *proj_params
        )
    else:
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
                * (
                    gevu.estimate_return_level(
                        period,
                        params[0] + params[1] * (return_period_diff[1] - years[0]),
                        params[2],
                        params[3],
                    )
                )
            )
            / (
                scalar
                * (
                    gevu.estimate_return_level(
                        period,
                        params[0] + params[1] * (return_period_diff[0] - years[0]),
                        params[2],
                        params[3],
                    )
                )
            )
            for period in periods_for_level
            for return_period_diff in return_period_diffs
        ]

    # Return
    if return_samples:
        if stationary:
            # Need to repeat historical results for each bootstrap sample if n_boot_hist = 1
            if n_boot_hist == 1:
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
                        f"{period}yr_return_level_hist": [return_levels_hist_main[i]]
                        for i, period in enumerate(periods_for_level)
                    },
                    **{
                        f"{period}yr_return_level_proj": [return_levels_proj_main[i]]
                        for i, period in enumerate(periods_for_level)
                    },
                    **{
                        f"{period}yr_return_level_diff": [
                            return_levels_proj_main[i] - return_levels_hist_main[i]
                        ]
                        for i, period in enumerate(periods_for_level)
                    },
                    **{
                        f"{period}yr_return_level_chfc": [
                            return_levels_proj_main[i] / return_levels_hist_main[i]
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
                        f"{period}yr_return_level_hist": scalar
                        * bootstrap_rls_hist[:, i]
                        for i, period in enumerate(periods_for_level)
                    },
                    **{
                        f"{period}yr_return_level_proj": scalar
                        * bootstrap_rls_proj[:, i]
                        for i, period in enumerate(periods_for_level)
                    },
                    **{
                        f"{period}yr_return_level_diff": (
                            scalar * bootstrap_rls_proj[:, i]
                        )
                        - (scalar * bootstrap_rls_hist[:, i])
                        for i, period in enumerate(periods_for_level)
                    },
                    **{
                        f"{period}yr_return_level_chfc": (
                            scalar * bootstrap_rls_proj[:, i]
                        )
                        / (scalar * bootstrap_rls_hist[:, i])
                        for i, period in enumerate(periods_for_level)
                    },
                }
            )
            return pd.concat([df_res, df_res_boot], ignore_index=True)
        else:
            df_res = pd.DataFrame(
                {
                    "ensemble": [ensemble],
                    "gcm": [gcm],
                    "member": [member],
                    "ssp": [ssp],
                    "n_boot": ["main"],
                    "loc_intcp": params[0],
                    "loc_trend": params[1],
                    "scale": params[2],
                    "shape": params[3],
                    **{
                        f"{period}yr_return_level_{return_period_year}": return_levels_main[
                            i_period * len(return_period_years) + i_year
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_year, return_period_year in enumerate(return_period_years)
                    },
                    **{
                        f"{period}yr_return_level_diff_{return_period_diff[1]}-{return_period_diff[0]}": return_level_diffs_main[
                            i_period * len(return_period_diffs) + i_diff
                        ]
                        for i_period, period in enumerate(periods_for_level)
                        for i_diff, return_period_diff in enumerate(return_period_diffs)
                    },
                    **{
                        f"{period}yr_return_level_chfc_{return_period_diff[1]}-{return_period_diff[0]}": return_level_chfcs_main[
                            i_period * len(return_period_diffs) + i_diff
                        ]
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
    else:
        ## Parameter results
        if stationary:
            # Historical
            hist_params_q025 = np.percentile(bootstrap_params_hist, 2.5, axis=0)
            hist_params_q975 = np.percentile(bootstrap_params_hist, 97.5, axis=0)
            return_levels_hist_q025 = np.percentile(
                scalar * bootstrap_rls_hist, 2.5, axis=0
            )
            return_levels_hist_q975 = np.percentile(
                scalar * bootstrap_rls_hist, 97.5, axis=0
            )

            # Projection
            proj_params_q025 = np.percentile(bootstrap_params_proj, 2.5, axis=0)
            proj_params_q975 = np.percentile(bootstrap_params_proj, 97.5, axis=0)
            return_levels_proj_q025 = np.percentile(
                scalar * bootstrap_rls_proj, 2.5, axis=0
            )
            return_levels_proj_q975 = np.percentile(
                scalar * bootstrap_rls_proj, 97.5, axis=0
            )
            # Differences
            return_levels_diff_main = return_levels_proj_main - return_levels_hist_main
            return_levels_diff_q025 = np.nanpercentile(
                (scalar * bootstrap_rls_proj) - (scalar * bootstrap_rls_hist),
                2.5,
                axis=0,
            )
            return_levels_diff_q975 = np.nanpercentile(
                (scalar * bootstrap_rls_proj) - (scalar * bootstrap_rls_hist),
                97.5,
                axis=0,
            )
            # Change factors
            return_levels_chfc_main = return_levels_proj_main / return_levels_hist_main
            return_levels_chfc_q025 = np.nanpercentile(
                (scalar * bootstrap_rls_proj) / (scalar * bootstrap_rls_hist),
                2.5,
                axis=0,
            )
            return_levels_chfc_q975 = np.nanpercentile(
                (scalar * bootstrap_rls_proj) / (scalar * bootstrap_rls_hist),
                97.5,
                axis=0,
            )
            # Return
            df_out = pd.DataFrame(
                {
                    "quantile": ["main", "q025", "q975"],
                    "ensemble": [ensemble, ensemble, ensemble],
                    "gcm": [gcm, gcm, gcm],
                    "member": [member, member, member],
                    "ssp": [ssp, ssp, ssp],
                    "loc_hist": [
                        hist_params[0],
                        hist_params_q025[0],
                        hist_params_q975[0],
                    ],
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
                    "loc_proj": [
                        proj_params[0],
                        proj_params_q025[0],
                        proj_params_q975[0],
                    ],
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
                            return_levels_hist_main[i],
                            return_levels_hist_q025[i],
                            return_levels_hist_q975[i],
                        ]
                        for i, period in enumerate(periods_for_level)
                    },
                    **{
                        f"{period}yr_return_level_proj": [
                            return_levels_proj_main[i],
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
            # Parameters
            params_q025 = np.nanpercentile(bootstrap_params, 2.5, axis=0)
            params_q975 = np.nanpercentile(bootstrap_params, 97.5, axis=0)
            # Return levels
            return_levels_q025 = np.nanpercentile(scalar * bootstrap_rls, 2.5, axis=0)
            return_levels_q975 = np.nanpercentile(scalar * bootstrap_rls, 97.5, axis=0)
            # Return level differences
            return_level_diffs_q025 = np.nanpercentile(
                scalar * bootstrap_rl_diffs, 2.5, axis=0
            )
            return_level_diffs_q975 = np.nanpercentile(
                scalar * bootstrap_rl_diffs, 97.5, axis=0
            )
            # Return level change factors
            return_level_chfcs_q025 = np.nanpercentile(
                scalar * bootstrap_rl_chfcs, 2.5, axis=0
            )
            return_level_chfcs_q975 = np.nanpercentile(
                scalar * bootstrap_rl_chfcs, 97.5, axis=0
            )

            # Return
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
                    "shape": [
                        params[3],
                        params_q025[3],
                        params_q975[3],
                    ],
                    **{
                        f"{period}yr_return_level_{return_period_year}": [
                            return_levels_main[
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
                            return_level_diffs_main[
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
                            return_level_chfcs_main[
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
    df = pd.read_csv(f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv")
    df = df.set_index(["ensemble", "gcm", "member", "ssp"]).sort_index()
    combos = df.index.unique()

    # Check if done
    stat_str = "stat" if stationary else "nonstat"
    sample_str = "_samples" if return_samples else ""
    if stationary:
        file_name = f"{city}_{metric_id}_{hist_slice[0]}-{hist_slice[1]}_{proj_slice[0]}-{proj_slice[1]}_{fit_method}_{stat_str}_nbootproj{n_boot_proj}_nboothist{n_boot_hist}{sample_str}.csv"
    else:
        file_name = f"{city}_{metric_id}_{years[0]}-{years[1]}_{fit_method}_{stat_str}_nboot{n_boot_proj}{sample_str}.csv"

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
                hist_slice=hist_slice,
                proj_slice=proj_slice,
                fit_method=fit_method,
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
