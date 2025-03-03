import numpy as np
import xarray as xr
import xarray_regrid

# LOCA grid
loca_lats = np.linspace(23.90625, 53.46875, 474)
loca_lons = np.linspace(234.53125, 293.46875, 944)

loca_grid = xr.Dataset(coords={"lat": ("lat", loca_lats), "lon": ("lon", loca_lons)})

# GARD-LENS grid
gard_lats = np.arange(25.125, 49.0, 0.125)  # GARD contains NaNs above 49N
gard_lons = np.linspace(-124.875, -67.0, 464)

gard_grid = xr.Dataset(coords={"lat": ("lat", gard_lats), "lon": ("lon", gard_lons)})


def regrid(ds_in, target, method, nan_threshold=0.5):
    """
    Regrids a dataset to the LOCA grid using xarray-regrid.
    """
    # Get target
    if target == "LOCA2":
        ds_target = loca_grid
    elif target == "GARD-LENS":
        ds_target = gard_grid

    # Make sure lat/lon is named correctly
    if "latitude" in ds_in.dims and "longitude" in ds_in.dims:
        ds_in = ds_in.rename({"latitude": "lat", "longitude": "lon"})

    # Regrid
    if method == "conservative":
        ds_out = ds_in.regrid.conservative(ds_target, nan_threshold=nan_threshold)
    elif method == "nearest":
        ds_out = ds_in.regrid.nearest(ds_target)

    return ds_out
