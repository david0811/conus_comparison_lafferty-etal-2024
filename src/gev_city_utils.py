import os

import numpy as np
import pandas as pd

import gev_utils as gevu
from utils import roar_code_path as project_code_path
from utils import roar_data_path as project_data_path
from utils import check_data_length

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
    hist_slice,
    proj_slice,
    fit_method,
    stationary,
    periods_for_level,
    years=None,
    return_period_years=None,
    return_period_diffs=None,
    bootstrap="parametric",
    n_boot=1000,
    project_data_path=project_data_path,
):
    """
    Fits the GEV model to a selected city, ensemble, GCM, member, SSP, and years.
    """

    # Read and select data
    df = pd.read_csv(f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv")

    if ensemble == "LOCA2":
        ssp_sel = "historical"
    else:
        ssp_sel = ssp

    if stationary:
        df_hist = df[
            (df["ensemble"] == ensemble)
            & (df["gcm"] == gcm)
            & (df["member"] == member)
            & (df["ssp"] == ssp_sel)
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
            & (df["ssp"].isin([ssp_sel, ssp]))
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
        expected_length_hist = check_data_length(hist_data, ensemble, gcm, "historical", hist_slice)
        expected_length_proj = check_data_length(proj_data, ensemble, gcm, ssp, proj_slice)
    else:
        _ = check_data_length(data, ensemble, gcm, ssp, years)

    # Do the fit
    if stationary:
        hist_params = gevu._fit_gev_1d_stationary(
            data=scalar * hist_data,
            expected_length=expected_length_hist,
            fit_method=fit_method,
        )
        proj_params = gevu._fit_gev_1d_stationary(
            data=scalar * proj_data,
            expected_length=expected_length_proj,
            fit_method=fit_method,
        )
    else:
        params = gevu._fit_gev_1d_nonstationary(
            data=scalar * data,
            years=years,
            fit_method=fit_method,
        )

    # Do the bootstrap if desried
    if bootstrap == "parametric":
        if stationary:
            bootstrap_params_hist, bootstrap_rls_hist = (
                gevu._gev_parametric_bootstrap_1d_stationary(
                    loc=hist_params[0],
                    scale=hist_params[1],
                    shape=hist_params[2],
                    n_data=len(hist_data),
                    n_boot=n_boot,
                    fit_method=fit_method,
                    periods_for_level=periods_for_level,
                    return_samples=True,
                )
            )
            bootstrap_params_proj, bootstrap_rls_proj = (
                gevu._gev_parametric_bootstrap_1d_stationary(
                    loc=proj_params[0],
                    scale=proj_params[1],
                    shape=proj_params[2],
                    n_data=len(proj_data),
                    n_boot=n_boot,
                    fit_method=fit_method,
                    periods_for_level=periods_for_level,
                    return_samples=True,
                )
            )
        else:
            bootstrap_params, bootstrap_rls, bootstrap_rl_diffs = (
                gevu._gev_parametric_bootstrap_1d_nonstationary(
                    params=params,
                    years=years,
                    n_data=len(data),
                    n_boot=n_boot,
                    fit_method=fit_method,
                    periods_for_level=periods_for_level,
                    return_period_years=return_period_years,
                    return_period_diffs=return_period_diffs,
                )
            )

    ## Parameter results
    if stationary:
        # Historical
        hist_params_q025 = np.percentile(bootstrap_params_hist, 2.5, axis=0)
        hist_params_q975 = np.percentile(bootstrap_params_hist, 97.5, axis=0)

        # Projection
        proj_params_q025 = np.percentile(bootstrap_params_proj, 2.5, axis=0)
        proj_params_q975 = np.percentile(bootstrap_params_proj, 97.5, axis=0)
    else:
        # Parameters
        params_q025 = np.percentile(bootstrap_params, 2.5, axis=0)
        params_q975 = np.percentile(bootstrap_params, 97.5, axis=0)

    # Combine
    if stationary:
        df_res = pd.DataFrame(
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
            }
        )
    else:
        df_res = pd.DataFrame(
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
            }
        )

    ## Return level results
    if stationary:
        # Get return levels
        return_levels_hist_main = scalar * gevu.estimate_return_level(
            np.array(periods_for_level), *hist_params
        )
        return_levels_hist_q025 = np.percentile(
            scalar * bootstrap_rls_hist, 2.5, axis=0
        )
        return_levels_hist_q975 = np.percentile(
            scalar * bootstrap_rls_hist, 97.5, axis=0
        )

        return_levels_proj_main = scalar * gevu.estimate_return_level(
            np.array(periods_for_level), *proj_params
        )
        return_levels_proj_q025 = np.percentile(
            scalar * bootstrap_rls_proj, 2.5, axis=0
        )
        return_levels_proj_q975 = np.percentile(
            scalar * bootstrap_rls_proj, 97.5, axis=0
        )

        # Diffs
        return_levels_diff_main = return_levels_proj_main - return_levels_hist_main
        return_levels_diff_q025 = np.percentile(
            scalar * (bootstrap_rls_proj - bootstrap_rls_hist), 2.5, axis=0
        )
        return_levels_diff_q975 = np.percentile(
            scalar * (bootstrap_rls_proj - bootstrap_rls_hist), 97.5, axis=0
        )
        # Store in dataframe
        df_return_levels = pd.DataFrame(
            {
                "quantile": ["main", "q025", "q975"],
                "ensemble": [ensemble, ensemble, ensemble],
                "gcm": [gcm, gcm, gcm],
                "member": [member, member, member],
                "ssp": [ssp, ssp, ssp],
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
            }
        )
    else:
        # Get return levels
        return_levels_main = [
            scalar * gevu.estimate_return_level(
                period,
                params[0] + params[1] * (return_period_year - years[0]),
                params[2],
                params[3],
            )
            for period in periods_for_level
            for return_period_year in return_period_years
        ]
        return_levels_q025 = np.percentile(scalar * bootstrap_rls, 2.5, axis=0)
        return_levels_q975 = np.percentile(scalar * bootstrap_rls, 97.5, axis=0)

        # Store in dataframe
        df_return_levels = pd.DataFrame(
            {
                "quantile": ["main", "q025", "q975"],
                "ensemble": [ensemble, ensemble, ensemble],
                "gcm": [gcm, gcm, gcm],
                "member": [member, member, member],
                "ssp": [ssp, ssp, ssp],
                **{
                    f"{period}yr_return_level_{return_period_year}": [
                        return_levels_main[i_period * len(return_period_years) + i_year],
                        return_levels_q025[i_period * len(return_period_years) + i_year],
                        return_levels_q975[i_period * len(return_period_years) + i_year],
                    ]
                    for i_period, period in enumerate(periods_for_level)
                    for i_year, return_period_year in enumerate(return_period_years)
                },
            }
        )
        # Get return level differences
        return_level_diffs_main = [
            scalar * (
            gevu.estimate_return_level(
                period,
                params[0] + params[1] * (return_period_diff[1] - years[0]),
                params[2],
                params[3],
            )) -
            scalar * (
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
        return_level_diffs_q025 = np.percentile(scalar * bootstrap_rl_diffs, 2.5, axis=0)
        return_level_diffs_q975 = np.percentile(scalar * bootstrap_rl_diffs, 97.5, axis=0)

        df_return_level_diffs = pd.DataFrame(
            {
            "quantile": ["main", "q025", "q975"],
            "ensemble": [ensemble, ensemble, ensemble],
            "gcm": [gcm, gcm, gcm],
            "member": [member, member, member],
            "ssp": [ssp, ssp, ssp],
            **{
                f"{period}yr_return_level_diff_{return_period_diff[1]}-{return_period_diff[0]}": [
                return_level_diffs_main[i_period * len(return_period_diffs) + i_diff],
                return_level_diffs_q025[i_period * len(return_period_diffs) + i_diff],
                return_level_diffs_q975[i_period * len(return_period_diffs) + i_diff],
                ]
                for i_period, period in enumerate(periods_for_level)
                for i_diff, return_period_diff in enumerate(return_period_diffs)
            },
            }
        )

    if stationary:
        return pd.merge(
            df_res,
            df_return_levels,
            on=["quantile", "ensemble", "gcm", "member", "ssp"],
        )
    else:
        return pd.merge(
            df_res,
            pd.merge(
                df_return_levels,
                df_return_level_diffs,
                on=["quantile", "ensemble", "gcm", "member", "ssp"],
            ),
            on=["quantile", "ensemble", "gcm", "member", "ssp"],
        )


def fit_ensemble_gev_city(
    city,
    metric_id,
    stationary,
    fit_method,
    periods_for_level,
    bootstrap="parametric",
    n_boot=1000,
    hist_slice=[1950, 2014],
    proj_slice=[2050, 2100],
    years=None,
    return_period_years=None,
    return_period_diffs=None,
    store=True,
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
    project_data_path : str, optional
        Base path for project data files
        
    Returns
    -------
    pandas.DataFrame or None
        If store=False, returns DataFrame with GEV fit results for all ensemble members.
        If store=True, saves results to CSV and returns None.
        Returns None if results file already exists.
        
    Notes
    -----
    Results are stored in CSV format with filename pattern:
    {city}_{metric_id}_{hist_start}-{hist_end}_{proj_start}-{proj_end}_{fit_method}_{stat/nonstat}_nboot{n_boot}.csv
    
    Failed fits are logged to individual text files in the project's log directory.
    """
    
    # Get unique combos
    df = pd.read_csv(f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv")
    df = df.set_index(["ensemble", "gcm", "member", "ssp"]).sort_index()
    combos = df.index.unique()

    # Check if done
    stat_str = "stat" if stationary else "nonstat"
    file_name = f"{city}_{metric_id}_{hist_slice[0]}-{hist_slice[1]}_{proj_slice[0]}-{proj_slice[1]}_{fit_method}_{stat_str}_nboot{n_boot}.csv"
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
            continue # Skip recalled outputs
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
                n_boot=n_boot,
                return_period_years=return_period_years,
                return_period_diffs=return_period_diffs,
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
