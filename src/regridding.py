import numpy as np
import xarray as xr
import xarray_regrid

# LOCA grid
loca_lats = np.linspace(23.90625, 53.46875, 474)
loca_lons = np.linspace(234.53125, 293.46875, 944)

loca_grid = xr.Dataset(
    coords={"lat": ("lat", loca_lats), "lon": ("lon", loca_lons)}
)


def regrid_to_loca(ds_in, method, nan_threshold=0.5):
    """
    Regrids a dataset to the LOCA grid using xarray-regrid.
    """
    # Make sure lat/lon is named correctly
    if "latitude" in ds_in.dims and "longitude" in ds_in.dims:
        ds_in = ds_in.rename({"latitude": "lat", "longitude": "lon"})

    # Regrid
    if method == "conservative":
        ds_out = ds_in.regrid.conservative(
            loca_grid, nan_threshold=nan_threshold
        )
    elif method == "nearest":
        ds_out = ds_in.regrid.nearest(loca_grid)

    return ds_out
