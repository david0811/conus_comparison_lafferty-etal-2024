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
        check_data_length(hist_data, ensemble, gcm, "historical", hist_slice)
        check_data_length(proj_data, ensemble, gcm, ssp, proj_slice)
    else:
        check_data_length(data, ensemble, gcm, ssp, years)

    # Do the fit
    if stationary:
        hist_params = gevu._fit_gev_1d_stationary(
            data=scalar * hist_data,
            fit_method=fit_method,
        )
        proj_params = gevu._fit_gev_1d_stationary(
            data=scalar * proj_data,
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
            bootstrap_params, bootstrap_rls, bootsrap_rl_diffs = (
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
        hist_params_p025 = np.percentile(bootstrap_params_hist, 2.5, axis=0)
        hist_params_p975 = np.percentile(bootstrap_params_hist, 97.5, axis=0)

        # Projection
        proj_params_p025 = np.percentile(bootstrap_params_proj, 2.5, axis=0)
        proj_params_p975 = np.percentile(bootstrap_params_proj, 97.5, axis=0)
    else:
        params_p025 = np.percentile(bootstrap_params, 2.5, axis=0)
        params_p975 = np.percentile(bootstrap_params, 97.5, axis=0)

    # Combine
    if stationary:
        df_res = pd.DataFrame(
            {
                "quantile": ["main", "p025", "p975"],
                "ensemble": [ensemble, ensemble, ensemble],
                "gcm": [gcm, gcm, gcm],
                "member": [member, member, member],
                "ssp": [ssp, ssp, ssp],
                "loc_hist": [hist_params[0], hist_params_p025[0], hist_params_p975[0]],
                "scale_hist": [
                    hist_params[1],
                    hist_params_p025[1],
                    hist_params_p975[1],
                ],
                "shape_hist": [
                    hist_params[2],
                    hist_params_p025[2],
                    hist_params_p975[2],
                ],
                "loc_proj": [proj_params[0], proj_params_p025[0], proj_params_p975[0]],
                "scale_proj": [
                    proj_params[1],
                    proj_params_p025[1],
                    proj_params_p975[1],
                ],
                "shape_proj": [
                    proj_params[2],
                    proj_params_p025[2],
                    proj_params_p975[2],
                ],
            }
        )
    else:
        df_res = pd.DataFrame(
            {
                "quantile": ["main", "p025", "p975"],
                "ensemble": [ensemble, ensemble, ensemble],
                "gcm": [gcm, gcm, gcm],
                "member": [member, member, member],
                "ssp": [ssp, ssp, ssp],
                "loc_intcp": [params[0], params_p025[0], params_p975[0]],
                "loc_trend": [params[1], params_p025[1], params_p975[1]],
                "scale": [params[2], params_p025[2], params_p975[2]],
                "shape": [params[3], params_p025[3], params_p975[3]],
            }
        )

    ## Return level results
    if stationary:
        # Get return levels
        return_levels_hist_main = scalar * gevu.estimate_return_level(
            np.array(periods_for_level), *hist_params
        )
        return_levels_hist_p025 = np.percentile(
            scalar * bootstrap_rls_hist, 2.5, axis=0
        )
        return_levels_hist_p975 = np.percentile(
            scalar * bootstrap_rls_hist, 97.5, axis=0
        )

        return_levels_proj_main = scalar * gevu.estimate_return_level(
            np.array(periods_for_level), *proj_params
        )
        return_levels_proj_p025 = np.percentile(
            scalar * bootstrap_rls_proj, 2.5, axis=0
        )
        return_levels_proj_p975 = np.percentile(
            scalar * bootstrap_rls_proj, 97.5, axis=0
        )

        # Change
        return_levels_change_main = return_levels_proj_main - return_levels_hist_main
        return_levels_change_p025 = np.percentile(
            scalar * (bootstrap_rls_proj - bootstrap_rls_hist), 2.5, axis=0
        )
        return_levels_change_p975 = np.percentile(
            scalar * (bootstrap_rls_proj - bootstrap_rls_hist), 97.5, axis=0
        )
        # Store in dataframe
        df_return_levels = pd.DataFrame(
            {
                "quantile": ["main", "p025", "p975"],
                "ensemble": [ensemble, ensemble, ensemble],
                "gcm": [gcm, gcm, gcm],
                "member": [member, member, member],
                "ssp": [ssp, ssp, ssp],
                **{
                    f"{period}yr_return_level_hist": [
                        return_levels_hist_main[i],
                        return_levels_hist_p025[i],
                        return_levels_hist_p975[i],
                    ]
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_proj": [
                        return_levels_proj_main[i],
                        return_levels_proj_p025[i],
                        return_levels_proj_p975[i],
                    ]
                    for i, period in enumerate(periods_for_level)
                },
                **{
                    f"{period}yr_return_level_change": [
                        return_levels_change_main[i],
                        return_levels_change_p025[i],
                        return_levels_change_p975[i],
                    ]
                    for i, period in enumerate(periods_for_level)
                },
            }
        )
    else:
        # Get return levels
        return_levels_main = [
            gevu.estimate_return_level(
                period,
                params[0] + params[1] * (return_period_year - years[0]),
                params[2],
                params[3],
            )
            for period in periods_for_level
            for return_period_year in return_period_years
        ]
        return_levels_p025 = np.percentile(scalar * bootstrap_rls, 2.5, axis=0)
        return_levels_p975 = np.percentile(scalar * bootstrap_rls, 97.5, axis=0)

        # Store in dataframe
        df_return_levels = pd.DataFrame(
            {
                "quantile": ["main", "p025", "p975"],
                "ensemble": [ensemble, ensemble, ensemble],
                "gcm": [gcm, gcm, gcm],
                "member": [member, member, member],
                "ssp": [ssp, ssp, ssp],
                **{
                    f"{period}yr_return_level_{return_period_year}": [
                        return_levels_main[i],
                        return_levels_p025[i],
                        return_levels_p975[i],
                    ]
                    for i, period in enumerate(periods_for_level)
                    for return_period_year in return_period_years
                },
            }
        )
        # Get return level differences
        return_level_diffs = [
            gevu.estimate_return_level(
                period,
                params[0] + params[1] * (return_period_diff[1] - return_period_diff[0]),
                params[2],
                params[3],
            )
            for period in periods_for_level
            for return_period_diff in return_period_diffs
        ]
        return_level_diffs_p025 = np.percentile(bootsrap_rl_diffs, 2.5, axis=0)
        return_level_diffs_p975 = np.percentile(bootsrap_rl_diffs, 97.5, axis=0)

        # Store in dataframe
        df_return_level_diffs = pd.DataFrame(
            {
                "quantile": ["main", "p025", "p975"],
                "ensemble": [ensemble, ensemble, ensemble],
                "gcm": [gcm, gcm, gcm],
                "member": [member, member, member],
                "ssp": [ssp, ssp, ssp],
                **{
                    f"{period}yr_return_level_{return_period_diff[1]}-{return_period_diff[0]}": [
                        return_level_diffs[i],
                        return_level_diffs_p025[i],
                        return_level_diffs_p975[i],
                    ]
                    for i, period in enumerate(periods_for_level)
                    for return_period_diff in return_period_diffs
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
    hist_slice=[1950, 2014],
    proj_slice=[2050, 2100],
    years=None,
    return_period_years=None,
    return_period_diffs=None,
    store=True,
    project_data_path=project_data_path,
):
    """
    Fit city GEV across the entire ensemble.
    """
    # Get unique combos
    df = pd.read_csv(f"{project_data_path}/metrics/cities/{city}_{metric_id}.csv")
    df = df.set_index(["ensemble", "gcm", "member", "ssp"]).sort_index()
    combos = df.index.unique()

    # Check if done
    stat_str = "stat" if stationary else "nonstat"
    file_name = f"{city}_{metric_id}_{hist_slice[0]}-{hist_slice[1]}_{proj_slice[0]}-{proj_slice[1]}_{fit_method}_{stat_str}.csv"
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
