from glob import glob

import numpy as np
import xarray as xr

from utils import roar_data_path as project_data_path


def read_loca(
    metric_id,
    grid,
    regrid_method,
    proj_slice,
    hist_slice,
    stationary,
    fit_method,
    bootstrap,
    cols_to_keep,
    analysis_type,
):
    """
    Reads the LOCA GEV/trend data for a given metric.
    """
    # Get grid info
    if grid == "LOCA2":
        loca_grid_str = "original_grid"
        loca_regrid_str = ""
    elif grid == "GARD-LENS":
        loca_grid_str = "gard_grid"
        loca_regrid_str = f"_{regrid_method}"

    # Get file info
    stat_name = "stat" if stationary else "nonstat"
    boot_name = "bootstrap" if bootstrap else "main"
    if analysis_type == "extreme_value":
        if bootstrap:
            file_info = (
                f"{proj_slice}_{hist_slice}_{stat_name}_{fit_method}_{boot_name}{loca_regrid_str}"
            )
        else:
            file_info = f"{proj_slice}_{stat_name}_{fit_method}_{boot_name}{loca_regrid_str}"
    elif analysis_type == "trends":
        file_info = loca_regrid_str

    # Read all
    loca_ssp245_files = glob(
        f"{project_data_path}/{analysis_type}/{loca_grid_str}/{metric_id}/LOCA2_*_ssp245_{file_info}.nc"
    )
    loca_ssp370_files = glob(
        f"{project_data_path}/{analysis_type}/{loca_grid_str}/{metric_id}/LOCA2_*_ssp370_{file_info}.nc"
    )
    loca_ssp585_files = glob(
        f"{project_data_path}/{analysis_type}/{loca_grid_str}/{metric_id}/LOCA2_*_ssp585_{file_info}.nc"
    )

    ds_loca = xr.concat(
        [
            xr.combine_by_coords(
                [xr.open_dataset(file)[cols_to_keep] for file in loca_ssp245_files]
            ),
            xr.combine_by_coords(
                [xr.open_dataset(file)[cols_to_keep] for file in loca_ssp370_files]
            ),
            xr.combine_by_coords(
                [xr.open_dataset(file)[cols_to_keep] for file in loca_ssp585_files]
            ),
        ],
        dim="ssp",
    )

    if hist_slice is not None:
        if not bootstrap:
            file_info = f"{hist_slice}_{stat_name}_{fit_method}_{boot_name}{loca_regrid_str}"
            loca_hist_files = glob(
                f"{project_data_path}/{analysis_type}/{loca_grid_str}/{metric_id}/LOCA2_*_{file_info}.nc"
            )
            ds_loca_hist = xr.combine_by_coords(
                [xr.open_dataset(file)[cols_to_keep] for file in loca_hist_files]
            )
            ds_loca = xr.concat([ds_loca, ds_loca_hist], dim="ssp")

    return ds_loca


def read_star(
    metric_id,
    grid,
    regrid_method,
    proj_slice,
    hist_slice,
    stationary,
    fit_method,
    bootstrap,
    cols_to_keep,
    analysis_type,
):
    """
    Reads the STAR GEV/trend data for a given metric.
    """
    # Get grid info
    if grid == "LOCA2":
        star_grid_str = "loca_grid"
        star_regrid_str = f"_{regrid_method}"
    elif grid == "GARD-LENS":
        star_grid_str = "gard_grid"
        star_regrid_str = f"_{regrid_method}"
    elif grid == "STAR-ESDM":
        star_grid_str = "original_grid"
        star_regrid_str = ""

    # Get file info
    stat_name = "stat" if stationary else "nonstat"
    boot_name = "bootstrap" if bootstrap else "main"
    if analysis_type == "extreme_value":
        if bootstrap:
            file_info = (
                f"{proj_slice}_{hist_slice}_{stat_name}_{fit_method}_{boot_name}{star_regrid_str}"
            )
        else:
            file_info = f"{proj_slice}_{stat_name}_{fit_method}_{boot_name}{star_regrid_str}"
    elif analysis_type == "trends":
        file_info = star_regrid_str

    # Read all files
    print(f"{project_data_path}/{analysis_type}/{star_grid_str}/{metric_id}/STAR-ESDM_*_{file_info}.nc")
    star_proj_files = glob(
        f"{project_data_path}/{analysis_type}/{star_grid_str}/{metric_id}/STAR-ESDM_*_{file_info}.nc"
    )
    ds_star = xr.combine_by_coords(
        [xr.open_dataset(file)[cols_to_keep] for file in star_proj_files]
    )

    # Read historical if desired
    if hist_slice is not None:
        if not bootstrap:
            file_info = f"{hist_slice}_{stat_name}_{fit_method}_{boot_name}{star_regrid_str}"
            star_hist_files = glob(
                f"{project_data_path}/{analysis_type}/{star_grid_str}/{metric_id}/STAR-ESDM_*_{file_info}.nc"
            )
            ds_star_hist = xr.combine_by_coords(
                [xr.open_dataset(file)[cols_to_keep] for file in star_hist_files]
            )
            ds_star = xr.concat([ds_star, ds_star_hist], dim="ssp")

    # Drop TaiESM1 -- too hot! (outputs were recalled)
    ds_star = ds_star.drop_sel(gcm="TaiESM1")

    return ds_star


def read_gard(
    metric_id,
    grid,
    regrid_method,
    proj_slice,
    hist_slice,
    stationary,
    fit_method,
    bootstrap,
    cols_to_keep,
    analysis_type,
):
    """
    Reads the GARD GEV/trend data for a given metric.
    """
    # Get grid info
    if grid == "LOCA2":
        gard_grid_str = "loca_grid"
        gard_regrid_str = f"_{regrid_method}"
    elif grid == "GARD-LENS":
        gard_grid_str = "original_grid"
        gard_regrid_str = ""

    # Get file info
    stat_name = "stat" if stationary else "nonstat"
    boot_name = "bootstrap" if bootstrap else "main"
    if analysis_type == "extreme_value":
        if bootstrap:
            file_info = (
                f"{proj_slice}_{hist_slice}_{stat_name}_{fit_method}_{boot_name}{gard_regrid_str}"
            )
        else:
            file_info = f"{proj_slice}_{stat_name}_{fit_method}_{boot_name}{gard_regrid_str}"
    elif analysis_type == "trends":
        file_info = gard_regrid_str

    # Read all files
    gard_proj_files = glob(
        f"{project_data_path}/{analysis_type}/{gard_grid_str}/{metric_id}/GARD-LENS_*_{file_info}.nc"
    )
    ds_gard = xr.combine_by_coords(
        [xr.open_dataset(file)[cols_to_keep] for file in gard_proj_files]
    )

    # Get hist if desired (only for main as boot already contains)
    if hist_slice is not None:
        if not bootstrap:
            file_info = f"{hist_slice}_{stat_name}_{fit_method}_{boot_name}{gard_regrid_str}"
            gard_hist_files = glob(
                f"{project_data_path}/{analysis_type}/{gard_grid_str}/{metric_id}/GARD-LENS_*_{file_info}.nc"
            )
            ds_gard_hist = xr.combine_by_coords(
                [xr.open_dataset(file)[cols_to_keep] for file in gard_hist_files]
            )
            ds_gard = xr.concat([ds_gard, ds_gard_hist], dim="ssp")

    return ds_gard


def read_all(
    metric_id,
    grid,
    regrid_method,
    proj_slice,
    hist_slice,
    stationary,
    fit_method,
    bootstrap,
    cols_to_keep,
    analysis_type,
):
    """
    Reads all the GEV data for a given metric.
    """
    ds_loca = read_loca(
        metric_id=metric_id,
        grid=grid,
        regrid_method=regrid_method,
        proj_slice=proj_slice,
        hist_slice=hist_slice,
        stationary=stationary,
        fit_method=fit_method,
        bootstrap=bootstrap,
        cols_to_keep=cols_to_keep,
        analysis_type=analysis_type,
    )
    ds_star = read_star(
        metric_id=metric_id,
        grid=grid,
        regrid_method=regrid_method,
        proj_slice=proj_slice,
        hist_slice=hist_slice,
        stationary=stationary,
        fit_method=fit_method,
        bootstrap=bootstrap,
        cols_to_keep=cols_to_keep,
        analysis_type=analysis_type,
    )
    ds_gard = read_gard(
        metric_id=metric_id,
        grid=grid,
        regrid_method=regrid_method,
        proj_slice=proj_slice,
        hist_slice=hist_slice,
        stationary=stationary,
        fit_method=fit_method,
        bootstrap=bootstrap,
        cols_to_keep=cols_to_keep,
        analysis_type=analysis_type,
    )

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
    gcm_uc = xr.concat([gard_gcm_range, star_gcm_range, loca_gcm_range], dim="ensemble")
    uq_maxs = gcm_uc.count(dim=["ensemble", "ssp"]) == gcm_uc.count(dim=["ensemble", "ssp"]).max()
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
    ssp_uc = xr.concat([loca_ssp_range, star_ssp_range, gard_ssp_range], dim="ensemble")
    uq_maxs = ssp_uc.count(dim="ensemble") == ssp_uc.count(dim="ensemble").max()
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
    iv_range = (ds[var_name].max(dim="member") - ds[var_name].min(dim="member")).where(
        combos_to_include
    )
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
    iv_uc = xr.concat([gard_iv_range, star_iv_range, loca_iv_range], dim="ensemble")
    # uq_maxs = (
    #     iv_uc.count(dim=["ensemble", "gcm", "ssp"])
    #     >= iv_uc.count(dim=["ensemble", "gcm", "ssp"]).max() - 1.0
    # )
    uq_maxs = (
        iv_uc.count(dim=["ensemble", "gcm", "ssp"])
        == iv_uc.count(dim=["ensemble", "gcm", "ssp"]).max()
    )
    iv_uc = iv_uc.where(uq_maxs).mean(dim=["ensemble", "gcm", "ssp"])

    return iv_uc


def compute_dsc_uc(ds_loca, ds_gard, ds_star, var_name):
    # Get GCM/SSP/member combinations for which we can compute downscaling uncertainty
    ilat, ilon = (
        int(len(ds_loca["lat"]) / 2),
        int(len(ds_loca["lon"]) / 2),
    )  # test point for non-null values
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

    combos_to_include = combos_to_include[combos_to_include[var_name]].reset_index()

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
            xr.combine_by_coords([ds, ds_star], join="left", combine_attrs="drop_conflicts"),
            xr.combine_by_coords([ds, ds_gard], join="left", combine_attrs="drop_conflicts"),
            xr.combine_by_coords([ds, ds_loca], join="left", combine_attrs="drop_conflicts"),
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


def ensemble_gev_range(ds, time_sel, var_name):
    """
    GEV uncertainty: 95% range
    """
    gev_range = ds[var_name].sel(time=time_sel, quantile="q975") - ds[var_name].sel(
        time=time_sel, quantile="q025"
    )
    return gev_range.where(gev_range != 0.0)


def compute_gev_uc(ds_loca, ds_gard, ds_star, time_sel, var_name):
    """
    Compute internal variability uncertainty
    """
    # Compute for individual ensembles
    loca_gev_range = ensemble_gev_range(ds_loca, time_sel, var_name)
    star_gev_range = ensemble_gev_range(ds_star, time_sel, var_name)
    gard_gev_range = ensemble_gev_range(ds_gard, time_sel, var_name)

    # Combine and average over ensembles
    # Compute the average 'by hand' to avoid issues with concat memory requirements
    loca_count = loca_gev_range.count(dim=["gcm", "member", "ssp"])
    star_count = star_gev_range.count(dim=["gcm", "member", "ssp"])
    gard_count = gard_gev_range.count(dim=["gcm", "member", "ssp"])
    total_count = (
        star_count.isel(ensemble=0) + loca_count.isel(ensemble=0) + gard_count.isel(ensemble=0)
    )

    loca_sum = loca_gev_range.sum(dim=["gcm", "member", "ssp"])
    star_sum = star_gev_range.sum(dim=["gcm", "member", "ssp"])
    gard_sum = gard_gev_range.sum(dim=["gcm", "member", "ssp"])
    total_sum = star_sum.isel(ensemble=0) + loca_sum.isel(ensemble=0) + gard_sum.isel(ensemble=0)

    # Again filter due to regridding issues
    gev_uc = (total_sum / total_count).where(total_count == total_count.max())

    return gev_uc


def compute_tot_uc(ds_loca, ds_gard, ds_star, time_sel, var_name):
    """
    Computes total uncertainty (full range).
    Need to do via stacking a new dimension since we can't merge all.
    """
    # Average of upper/lower quantiles
    ds_upper = xr.concat(
        [
            ds_loca[var_name]
            .sel(time=time_sel, quantile="q975")
            .stack(z=("ensemble", "gcm", "ssp", "member")),
            ds_star[var_name]
            .sel(time=time_sel, quantile="q975")
            .stack(z=("ensemble", "gcm", "ssp", "member")),
            ds_gard[var_name]
            .sel(time=time_sel, quantile="q975")
            .stack(z=("ensemble", "gcm", "ssp", "member")),
        ],
        dim="z",
    )

    ds_lower = xr.concat(
        [
            ds_loca[var_name]
            .sel(time=time_sel, quantile="q025")
            .stack(z=("ensemble", "gcm", "ssp", "member")),
            ds_star[var_name]
            .sel(time=time_sel, quantile="q025")
            .stack(z=("ensemble", "gcm", "ssp", "member")),
            ds_gard[var_name]
            .sel(time=time_sel, quantile="q025")
            .stack(z=("ensemble", "gcm", "ssp", "member")),
        ],
        dim="z",
    )

    # Measures of uncertainty
    uc_99w = ds_upper.quantile(0.995, dim='z') - ds_lower.quantile(0.005, dim='z')

    return uc_99w


def uc_all(
    metric_id,
    grid,
    regrid_method,
    stationary,
    fit_method,
    col_name,
    proj_slice,
    hist_slice,
    return_metric=False,
    analysis_type="extreme_value",
):
    """
    Perform the UC for all.
    """
    # Read all: main
    ds_loca, ds_star, ds_gard = read_all(
        metric_id=metric_id,
        grid=grid,
        regrid_method=regrid_method,
        proj_slice=proj_slice,
        hist_slice=hist_slice,
        stationary=stationary,
        fit_method=fit_method,
        bootstrap=False,
        cols_to_keep=[col_name],
        analysis_type=analysis_type,
    )

    # Compute change if desired
    if hist_slice is not None:
        ds_loca = (ds_loca - ds_loca.sel(ssp="historical")).drop_sel(ssp="historical")
        ds_gard = (ds_gard - ds_gard.sel(ssp="historical")).drop_sel(ssp="historical")
        ds_star = (ds_star - ds_star.sel(ssp="historical")).drop_sel(ssp="historical")

    # Compute GCM uncertainty
    gcm_uc = compute_gcm_uc(ds_loca, ds_gard, ds_star, col_name)

    # Compute SSP uncertainty
    ssp_uc = compute_ssp_uc(ds_loca, ds_gard, ds_star, col_name)
    ssp_uc_by_gcm = compute_ssp_uc(ds_loca, ds_gard, ds_star, col_name, by_gcm=True)

    # Compute internal variability uncertainty
    iv_uc = compute_iv_uc(ds_loca, ds_gard, ds_star, col_name)

    # Compute downscaling uncertainty
    dsc_uc = compute_dsc_uc(ds_loca, ds_gard, ds_star, col_name)

    del ds_loca, ds_star, ds_gard # memory management

    # Read all: bootstrap
    ds_loca, ds_star, ds_gard = read_all(
        metric_id=metric_id,
        grid=grid,
        regrid_method=regrid_method,
        proj_slice=proj_slice,
        hist_slice="1950-2014",
        stationary=stationary,
        fit_method=fit_method,
        bootstrap=True,
        cols_to_keep=[col_name],
        analysis_type=analysis_type,
    )

    # Compute change if desired
    if bool(hist_slice and proj_slice):
        time_sel = "diff"
    elif hist_slice is None:
        time_sel = "proj"
    elif proj_slice is None:
        time_sel = "hist"

    # Compute GEV uncertainty
    gev_uc = compute_gev_uc(ds_loca, ds_gard, ds_star, time_sel, col_name)

    # Compute total uncertainty
    uc_99w = compute_tot_uc(ds_loca, ds_gard, ds_star, time_sel, col_name)

    # Merge and return
    ssp_uc = ssp_uc.rename("ssp_uc")
    ssp_uc_by_gcm = ssp_uc_by_gcm.rename("ssp_uc_by_gcm")
    gcm_uc = gcm_uc.rename("gcm_uc")
    iv_uc = iv_uc.rename("iv_uc")
    dsc_uc = dsc_uc.rename("dsc_uc")
    gev_uc = gev_uc.rename("gev_uc")
    # uc_range = uc_range.rename("uc_range")
    # uc_95w = uc_95w.rename("uc_95w")
    uc_99w = uc_99w.rename("uc_99w")

    uc = xr.merge(
        [
            ssp_uc_by_gcm,
            ssp_uc,
            gcm_uc,
            iv_uc,
            dsc_uc,
            gev_uc,
            # uc_range,
            # uc_95w,
            uc_99w,
        ]
    )

    if return_metric:
        return uc, ds_loca, ds_star, ds_gard
    else:
        return uc
