import os
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

import gev_utils as gevu
from utils import roar_data_path as project_data_path


def read_all(
    metric_id,
    regrid_method,
    proj_slice,
    hist_slice,
):
    """
    Reads all the GEV data for a given metric.

    NOTE: Assumes loca_grid.
    """
    ## Read all
    # LOCA
    loca_ssp245_files = glob(
        f"{project_data_path}/extreme_value/original_grids/{metric_id}/LOCA2_*_ssp245_{proj_slice}.nc"
    )
    loca_ssp370_files = glob(
        f"{project_data_path}/extreme_value/original_grids/{metric_id}/LOCA2_*_ssp370_{proj_slice}.nc"
    )
    loca_ssp585_files = glob(
        f"{project_data_path}/extreme_value/original_grids/{metric_id}/LOCA2_*_ssp585_{proj_slice}.nc"
    )

    ds_loca = xr.concat(
        [
            xr.combine_by_coords(
                [xr.open_dataset(file) for file in loca_ssp245_files]
            ),
            xr.combine_by_coords(
                [xr.open_dataset(file) for file in loca_ssp370_files]
            ),
            xr.combine_by_coords(
                [xr.open_dataset(file) for file in loca_ssp585_files]
            ),
        ],
        dim="ssp",
    )

    # STAR
    star_proj_files = glob(
        f"{project_data_path}/extreme_value/loca_grid/{metric_id}/STAR-ESDM_*_{proj_slice}_{regrid_method}.nc"
    )
    ds_star = xr.combine_by_coords(
        [xr.open_dataset(file) for file in star_proj_files]
    )

    # GARD
    gard_proj_files = glob(
        f"{project_data_path}/extreme_value/loca_grid/{metric_id}/GARD-LENS_*_{proj_slice}_{regrid_method}.nc"
    )
    ds_gard = xr.combine_by_coords(
        [xr.open_dataset(file) for file in gard_proj_files]
    )

    # Hist if needed
    if hist_slice is not None:
        # LOCA
        loca_hist_files = glob(
            f"{project_data_path}/extreme_value/original_grids/{metric_id}/LOCA2_*_{hist_slice}.nc"
        )
        ds_loca_hist = xr.combine_by_coords(
            [xr.open_dataset(file) for file in loca_hist_files]
        )
        ds_loca = xr.concat([ds_loca, ds_loca_hist], dim="ssp")

        # STAR
        star_hist_files = glob(
            f"{project_data_path}/extreme_value/loca_grid/{metric_id}/STAR-ESDM_*_{hist_slice}_{regrid_method}.nc"
        )
        ds_star_hist = xr.combine_by_coords(
            [xr.open_dataset(file) for file in star_hist_files]
        )
        ds_star = xr.concat([ds_star, ds_star_hist], dim="ssp")

        # GARD
        gard_hist_files = glob(
            f"{project_data_path}/extreme_value/loca_grid/{metric_id}/GARD-LENS_*_{hist_slice}_{regrid_method}.nc"
        )
        ds_gard_hist = xr.combine_by_coords(
            [xr.open_dataset(file) for file in gard_hist_files]
        )
        ds_gard = xr.concat([ds_gard, ds_gard_hist], dim="ssp")

    return ds_loca, ds_star, ds_gard


def ensemble_gcm_range(ds, min_members, var_name):
    """
    GCM uncertainty: range across forced responses
    """
    combos_to_include = ds[var_name].count(dim=["member"]) >= min_members
    ds_forced = ds[var_name].mean(dim="member").where(combos_to_include)
    gcm_range = ds_forced.max(dim="gcm") - ds_forced.min(dim="gcm")
    return gcm_range.where(gcm_range != 0.0)


def compute_gcm_uc(ds_loca, ds_gard, ds_star, var_name, min_members=5):
    """
    Compute GCM uncertainty
    """
    # Compute for individual ensembles
    loca_gcm_range = ensemble_gcm_range(ds_loca, min_members, var_name)
    star_gcm_range = ensemble_gcm_range(ds_star, min_members, var_name)
    gard_gcm_range = ensemble_gcm_range(ds_gard, min_members, var_name)

    # Combine and average over SSPs, ensembles
    # Note: due to the regridding, there are some gridpoints where the
    # range is computed across only 1 ensemble, so we filter any below
    # the maximum count
    gcm_uc = xr.concat(
        [gard_gcm_range, star_gcm_range, loca_gcm_range], dim="ensemble"
    )
    uq_maxs = (
        gcm_uc.count(dim=["ensemble", "ssp"])
        == gcm_uc.count(dim=["ensemble", "ssp"]).max()
    )
    gcm_uc = gcm_uc.where(uq_maxs).mean(dim=["ensemble", "ssp"])

    return gcm_uc


def ensemble_ssp_range(ds, var_name):
    """
    SSP uncertainty: range across ensemble means
    """
    ensemble_mean = ds[var_name].mean(dim=["member", "gcm"])
    ssp_range = ensemble_mean.max(dim="ssp") - ensemble_mean.min(dim="ssp")
    return ssp_range.where(ssp_range != 0.0)


def ensemble_ssp_range_by_gcm(ds, var_name, min_members=5):
    """
    SSP uncertainty: range across forced responses for each GCM
    """
    combos_to_include = ds[var_name].count(dim=["member"]) >= min_members
    ensemble_mean = ds[var_name].mean(dim=["member"]).where(combos_to_include)
    ssp_range = ensemble_mean.max(dim=["ssp"]) - ensemble_mean.min(dim=["ssp"])
    return ssp_range.where(ssp_range != 0.0)


def compute_ssp_uc(ds_loca, ds_gard, ds_star, var_name, by_gcm=False):
    """
    Compute SSP uncertainty
    """
    # Compute for individual ensembles
    if by_gcm:
        loca_ssp_range = ensemble_ssp_range_by_gcm(ds_loca, var_name)
        star_ssp_range = ensemble_ssp_range_by_gcm(ds_star, var_name)
        gard_ssp_range = ensemble_ssp_range_by_gcm(ds_gard, var_name)
    else:
        loca_ssp_range = ensemble_ssp_range(ds_loca, var_name)
        star_ssp_range = ensemble_ssp_range(ds_star, var_name)
        gard_ssp_range = ensemble_ssp_range(ds_gard, var_name)

    # Combine and average over ensembles
    # Again filter due to regridding issues
    ssp_uc = xr.concat(
        [loca_ssp_range, star_ssp_range, gard_ssp_range], dim="ensemble"
    )
    uq_maxs = (
        ssp_uc.count(dim="ensemble") == ssp_uc.count(dim="ensemble").max()
    )
    if by_gcm:
        ssp_uc = ssp_uc.where(uq_maxs).mean(dim=["ensemble", "gcm"])
    else:
        ssp_uc = ssp_uc.where(uq_maxs).mean(dim="ensemble")

    return ssp_uc


def ensemble_iv_range(ds, min_members, var_name):
    """
    Internal variability uncertainty: range across members
    """
    combos_to_include = ds[var_name].count(dim=["member"]) >= min_members
    iv_range = (
        ds[var_name].max(dim="member") - ds[var_name].min(dim="member")
    ).where(combos_to_include)
    return iv_range.where(iv_range != 0.0)


def compute_iv_uc(ds_loca, ds_gard, ds_star, var_name, min_members=5):
    """
    Compute internal variability uncertainty
    """
    # Compute for individual ensembles
    loca_iv_range = ensemble_iv_range(ds_loca, min_members, var_name)
    star_iv_range = ensemble_iv_range(ds_star, min_members, var_name)
    gard_iv_range = ensemble_iv_range(ds_gard, min_members, var_name)

    # Combine and average over ensembles
    # Again filter due to regridding issues -- here there are some gridpoints where
    # I think the GEV fitting is running into invalid L-moments, so I use a less strict
    # filter
    iv_uc = xr.concat(
        [gard_iv_range, star_iv_range, loca_iv_range], dim="ensemble"
    )
    uq_maxs = (
        iv_uc.count(dim=["ensemble", "gcm", "ssp"])
        >= iv_uc.count(dim=["ensemble", "gcm", "ssp"]).max() - 1.0
    )
    iv_uc = iv_uc.where(uq_maxs).mean(dim=["ensemble", "gcm", "ssp"])

    return iv_uc


def compute_dsc_uc(ds_loca, ds_gard, ds_star, var_name):
    # Get GCM/SSP/member combinations for which we can compute downscaling uncertainty
    ilat, ilon = 200, 400  # test point for non-null values
    combos_to_include = (
        xr.concat(
            [
                ds_loca.isel(lat=ilat, lon=ilon),
                ds_star.isel(lat=ilat, lon=ilon),
                ds_gard.isel(lat=ilat, lon=ilon),
            ],
            dim="ensemble",
            join="outer",
        )[var_name].count(dim="ensemble")
        > 1
    ).to_dataframe()

    combos_to_include = combos_to_include[
        combos_to_include[var_name]
    ].reset_index()

    # Get unique GCMs, SSPs, members
    gcms_include = np.sort(combos_to_include["gcm"].unique())
    ssps_include = np.sort(combos_to_include["ssp"].unique())
    members_include = np.sort(combos_to_include["member"].unique())

    # Construct empty dataset to fill in
    ensembles_include = ["GARD-LENS", "LOCA2", "STAR-ESDM"]
    ds = xr.Dataset(
        coords={
            "ensemble": ensembles_include,
            "gcm": gcms_include,
            "member": members_include,
            "ssp": ssps_include,
            "lat": ds_loca.lat,
            "lon": ds_loca.lon,
        },
    )

    # Combine all
    ds_combined = xr.merge(
        [
            xr.combine_by_coords(
                [ds, ds_star], join="left", combine_attrs="drop_conflicts"
            ),
            xr.combine_by_coords(
                [ds, ds_gard], join="left", combine_attrs="drop_conflicts"
            ),
            xr.combine_by_coords(
                [ds, ds_loca], join="left", combine_attrs="drop_conflicts"
            ),
        ],
        combine_attrs="drop_conflicts",
    )

    # Downscaling uncertainty
    dsc_uc = ds_combined.max(dim="ensemble") - ds_combined.min(dim="ensemble")
    # Filter at least 2 ensembles
    dsc_uc = (
        dsc_uc[var_name]
        .where(ds_combined[var_name].count(dim="ensemble") > 1)
        .mean(dim=["gcm", "ssp", "member"])
    )

    return dsc_uc


def compute_tot_uc(ds_loca, ds_gard, ds_star, var_name):
    """
    Computes total uncertainty (full range).
    Need to do this in stages since we can't merge to full ensemble.
    """
    # Max of all ensembles
    loca_max = ds_loca[var_name].max(dim=["member", "gcm", "ssp"])
    gard_max = ds_gard[var_name].max(dim=["member", "gcm", "ssp"])
    star_max = ds_star[var_name].max(dim=["member", "gcm", "ssp"])
    ens_max = xr.concat([loca_max, gard_max, star_max], dim="ensemble").max(
        dim="ensemble"
    )

    # Min of all ensembles
    loca_min = ds_loca[var_name].min(dim=["member", "gcm", "ssp"])
    gard_min = ds_gard[var_name].min(dim=["member", "gcm", "ssp"])
    star_min = ds_star[var_name].min(dim=["member", "gcm", "ssp"])
    ens_min = xr.concat([loca_min, gard_min, star_min], dim="ensemble").min(
        dim="ensemble"
    )

    # Compute total uncertainty
    tot_uc = ens_max - ens_min

    return tot_uc


def uc_all(
    metric_id,
    regrid_method,
    return_period,
    return_level=None,
    proj_slice="2050-2100",
    hist_slice=None,
    return_metric=False,
):
    """
    Perform the UC  for all
    """
    # Read all
    ds_loca, ds_star, ds_gard = read_all(
        metric_id, regrid_method, proj_slice, hist_slice
    )

    # Invert if minima
    if metric_id == "min_tasmin":
        scalar = -1.0
    else:
        scalar = 1.0

    # Compute return level/period
    if return_period is not None:
        ds_loca = scalar * gevu.xr_estimate_return_level(
            return_period, ds_loca
        )
        ds_gard = scalar * gevu.xr_estimate_return_level(
            return_period, ds_gard
        )
        ds_star = scalar * gevu.xr_estimate_return_level(
            return_period, ds_star
        )
        var_name = f"{return_period}yr_return_level"
    elif return_level is not None:
        ds_loca = gevu.xr_estimate_return_period(return_level, ds_loca)
        ds_gard = gevu.xr_estimate_return_period(return_level, ds_gard)
        ds_star = gevu.xr_estimate_return_period(return_level, ds_star)
        var_name = f"{return_level}_return_period"

    # Compute change if desired
    if hist_slice is not None:
        ds_loca = (ds_loca - ds_loca.sel(ssp="historical")).drop_sel(
            ssp="historical"
        )
        ds_gard = (ds_gard - ds_gard.sel(ssp="historical")).drop_sel(
            ssp="historical"
        )
        ds_star = (ds_star - ds_star.sel(ssp="historical")).drop_sel(
            ssp="historical"
        )

    # Compute total uncertainty
    tot_uc = compute_tot_uc(ds_loca, ds_gard, ds_star, var_name)

    # Compute GCM uncertainty
    gcm_uc = compute_gcm_uc(ds_loca, ds_gard, ds_star, var_name)

    # Compute SSP uncertainty
    ssp_uc = compute_ssp_uc(ds_loca, ds_gard, ds_star, var_name)

    # Compute internal variability uncertainty
    iv_uc = compute_iv_uc(ds_loca, ds_gard, ds_star, var_name)

    # Compute downscaling uncertainty
    dsc_uc = compute_dsc_uc(ds_loca, ds_gard, ds_star, var_name)

    # Merge and return
    gcm_uc = gcm_uc.rename("gcm_uc")
    ssp_uc = ssp_uc.rename("ssp_uc")
    iv_uc = iv_uc.rename("iv_uc")
    dsc_uc = dsc_uc.rename("dsc_uc")
    tot_uc = tot_uc.rename("tot_uc")
    uc = xr.merge([gcm_uc, ssp_uc, iv_uc, dsc_uc, tot_uc])

    if return_metric:
        return uc, ds_loca, ds_star, ds_gard
    else:
        return uc


def store_all_cities(
    metric_id,
    regrid_method,
    proj_slice,
    hist_slice,
    return_period,
    city_list,
    return_level=None,
):
    """
    Store all cities as csv files for a given metric.
    """
    # Check if done for all cities
    if return_period is not None:
        metric_save_name = f"{int(return_period)}rl"
    elif return_level is not None:
        metric_save_name = f"{return_level}rp"
    file_names = [
        f"{city}_{metric_id}_{proj_slice}_{hist_slice}_{metric_save_name}_{regrid_method}.csv"
        for city in list(city_list.keys())
    ]

    if not np.all(
        [
            os.path.exists(
                f"{project_data_path}/extreme_value/cities/loca_grid/{file_name}"
            )
            for file_name in file_names
        ]
    ):
        # Read all
        ds_loca, ds_star, ds_gard = read_all(
            metric_id, regrid_method, proj_slice, hist_slice
        )

        # Invert if minima
        if metric_id == "min_tasmin":
            scalar = -1.0
        else:
            scalar = 1.0

        # Compute return level/period
        if return_period is not None:
            ds_loca = scalar * gevu.xr_estimate_return_level(
                return_period, ds_loca
            )
            ds_gard = scalar * gevu.xr_estimate_return_level(
                return_period, ds_gard
            )
            ds_star = scalar * gevu.xr_estimate_return_level(
                return_period, ds_star
            )
        elif return_level is not None:
            ds_loca = gevu.xr_estimate_return_period(return_level, ds_loca)
            ds_gard = gevu.xr_estimate_return_period(return_level, ds_gard)
            ds_star = gevu.xr_estimate_return_period(return_level, ds_star)

        # Loop through cities
        for city in city_list:
            # Read
            lat, lon = city_list[city]
            df_loca = (
                ds_loca.sel(lat=lat, lon=360 + lon, method="nearest")
                .to_dataframe()
                .dropna()
                .drop(columns=["lat", "lon"])
                .reset_index()
            )
            df_star = (
                ds_star.sel(lat=lat, lon=360 + lon, method="nearest")
                .to_dataframe()
                .dropna()
                .drop(columns=["lat", "lon"])
                .reset_index()
            )
            df_gard = (
                ds_gard.sel(lat=lat, lon=360 + lon, method="nearest")
                .to_dataframe()
                .dropna()
                .drop(columns=["lat", "lon"])
                .reset_index()
            )

            # Concat
            df_all = pd.concat([df_loca, df_star, df_gard])

            # Store
            file_name = f"{city}_{metric_id}_{proj_slice}_{hist_slice}_{metric_save_name}_{regrid_method}.csv"
            df_all.to_csv(
                f"{project_data_path}/extreme_value/cities/loca_grid/{file_name}",
                index=False,
            )
