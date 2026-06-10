import json
import urllib.request
import warnings
from importlib.resources import files
from pathlib import Path
from typing import TypeVar

import numpy as np
import pooch
import xarray as xr

from .options import DATA_VERSION, JS2_HTTPS_BASE, OPTIONS, ZENODO_CONCEPT_ID

# Generic over the two xarray container types, used by functions (e.g.
# check_coords) that accept either a DataArray or a Dataset and return the
# same type they were given. Defined here as the single source of truth;
# frontend.py imports it for consistency.
XrObj = TypeVar("XrObj", xr.DataArray, xr.Dataset)


# monkey patch Python warnings format function
def custom_formatwarning(
    msg: object, cat: type[Warning], *args: object, **kwargs: object
) -> str:
    return str(cat.__name__) + ": " + str(msg) + "\n"


warnings.formatwarning = custom_formatwarning  # type: ignore[assignment]

# filter out warnings from xarray when using the rename function
# ideally, this can be removed in the future
warnings.filterwarnings("ignore", ".*does not create an index anymore.*")


def get_dp(
    ds_4D: xr.DataArray,
    ps: xr.DataArray,
    tropo: xr.DataArray,
    layer: str = "troposphere",
) -> xr.DataArray:
    """Calculate layer thickness using model pressure levels, surface
    pressure, and tropopause pressure. Also specify the layer as either
    'troposphere' or 'stratosphere'.
    """
    # construct a 4D DataArray corresponding to layer thickness
    # for vertical integration later
    # this is achieved by finding the midpoints between pressure levels
    # and bounding that array with surface pressure below
    # and TOA (p=0) above
    aligned = xr.align(ds_4D.plev[1:], ds_4D.plev[:-1], join="override")
    mids = xr.broadcast((aligned[0] + aligned[1]) / 2, ps)[0]
    ps_expand = ps.expand_dims(dim={"plev": [ds_4D.plev[0]]}, axis=1)

    TOA = xr.zeros_like(ps_expand)
    TOA["plev"] = ps_expand.plev * 0

    # this if/else statement accounts for potentially
    # reversed pressure axis direction
    if ds_4D.plev[0] > ds_4D.plev[-1]:
        ilevs = xr.concat([ps_expand, mids, TOA], dim="plev")
        sign_change = -1
    else:
        ilevs = xr.concat([TOA, mids, ps_expand], dim="plev")
        sign_change = 1

    if layer == "troposphere":
        # make points above tropopause equal to tropopause height
        # make points below surface pressure equal to surface pressure
        ilevs = ilevs.where(ilevs > tropo, tropo).where(ilevs < ps, ps)
    elif layer == "stratosphere":
        # make points below tropopause equal to tropopause height
        ilevs = ilevs.where(ilevs < tropo, tropo)

    # get the layer thickness by taking finite difference
    # along pressure axis
    dp = sign_change * ilevs.diff(dim="plev", label="lower")

    # override pressure axis so xarray doesn't throw a fit
    dp["plev"] = ds_4D.plev

    # return dp
    return dp


def check_var_units(da: xr.DataArray, var: str) -> xr.DataArray:
    """Check to see if the xarray DataArray has a units attribute."""
    if "units" not in da.attrs:
        if var == "q":
            warnings.warn("No units found for input q. Assuming kg/kg.", stacklevel=2)
            return da.assign_attrs({"units": "kg/kg"})
        elif var == "T":
            warnings.warn("No units found for input T. Assuming K.", stacklevel=2)
            return da.assign_attrs({"units": "K"})
    # Either units are already present, or we have no default to assume for
    # this variable, so pass the DataArray through unchanged.
    return da


def make_tropo(da: xr.DataArray) -> xr.DataArray:
    """Use the a DataArray containing model lat and lon to make a makeshift
    tropopause.
    """
    tropo = (3e4 - 2e4 * np.cos(np.deg2rad(da.lat))).broadcast_like(da)
    return tropo


def check_plev_units(da: xr.DataArray) -> xr.DataArray:
    if "units" not in da.plev.attrs:
        warnings.warn(
            "No units found for input vertical coordinate. Assuming Pa.", stacklevel=2
        )
        plev = da.plev.assign_attrs({"units": "Pa"})
        return da.assign_coords({"plev": plev})
    elif da.plev.units in ["hPa", "mb", "millibars"]:
        da["plev"] = da.plev * 100
        da.plev.attrs["units"] = "Pa"
        return da
    else:
        return da


def check_pres_units(da: xr.DataArray, var_name: str) -> xr.DataArray:
    if "units" not in da.attrs:
        warnings.warn(
            "Could not determine units of " + var_name + ". Assuming Pa.", stacklevel=2
        )
        return da.assign_attrs({"units": "Pa"})
    elif da.units in ["hPa", "mb", "millibars"]:
        da = da * 100
        da.attrs["units"] = "Pa"
        return da
    else:
        return da


def tile_data(to_tile: xr.DataArray, new_shape: xr.DataArray) -> xr.DataArray:
    """Tile dataset along time axis to match another dataset."""
    # new_shape = _check_time(new_shape)
    if len(new_shape.time) % 12 != 0:
        raise ValueError("dataset time dimension must be divisible by 12")
    if "month" not in to_tile.coords:
        tiled = xr.concat(
            [to_tile for i in range(int(len(new_shape.time) / 12))], dim="time"
        )
    if "month" in to_tile.dims and len(to_tile.month) == 12:
        to_tile = to_tile.rename({"month": "time"})
        tiled = xr.concat(
            [to_tile for i in range(int(len(new_shape.time) / 12))], dim="time"
        )
    tiled["time"] = new_shape.time
    return tiled


def _cache_root() -> Path:
    """Directory where cached/downloaded kernel and tutorial data live."""
    if OPTIONS["cache_dir"]:
        return Path(OPTIONS["cache_dir"])
    return Path(pooch.os_cache("climkern"))


def _load_registry() -> dict[str, str]:
    """Parse the shipped pooch registry (relpath -> hash). Empty if absent."""
    path = files("climkern").joinpath("registry.txt")
    if not path.is_file():
        return {}
    registry = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, file_hash = line.partition(" ")
        registry[name] = file_hash.strip()
    return registry


# Parsed once; the hashes are fixed for a given installed package version.
_REGISTRY = _load_registry()


def _pooch() -> pooch.Pooch:
    """Build a pooch fetcher pointed at the current cache dir and Jetstream2."""
    return pooch.create(
        path=_cache_root(),
        base_url=JS2_HTTPS_BASE,
        registry=_REGISTRY or None,
        env="CLIMKERN_DATA_DIR",
    )


def _version_tuple(version: str) -> tuple[int, ...]:
    """Break up the version string by periods and convert
    to a tuple for better comparison.
    """
    parts = []
    for piece in str(version).split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts)


_version_checked = False


def _check_remote_version() -> None:
    """Warn once per session if a newer kernel data release exists on Zenodo.

    Best-effort and silent on any failure (offline, API change, ...) so it can
    never block data access.
    """
    global _version_checked
    if _version_checked or not OPTIONS["version_check"]:
        return
    _version_checked = True  # set first: a failed/slow check must not retry
    try:
        url = f"https://zenodo.org/api/records/{ZENODO_CONCEPT_ID}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            latest = json.load(resp)["metadata"]["version"]
        if _version_tuple(latest) > _version_tuple(DATA_VERSION):
            warnings.warn(
                f"A newer ClimKern kernel dataset (v{latest}) is available; this "
                f"ClimKern pins v{DATA_VERSION}. Upgrade ClimKern for the latest "
                "kernels, or silence this with "
                "climkern.set_options(version_check=False).",
                stacklevel=2,
            )
    except Exception:
        pass  # version check is advisory only; never break on it


def _open_dataset(relpath: str, **kwargs: object) -> xr.Dataset:
    """Open a kernel/tutorial netCDF per the current ``data_source`` option.

    ``relpath`` is the path below the data root, e.g.
    ``"kernels/GFDL/TOA_GFDL_Kerns.nc"`` or ``"tutorial_data/ctrl.nc"``.
    """
    _check_remote_version()
    source = OPTIONS["data_source"]

    if source == "stream":
        import fsspec  # lazy: only needed for streaming

        url = JS2_HTTPS_BASE + relpath
        # netCDF4's C library cannot read a Python file object, so a streamed
        # HDF5 file must be opened with the h5netcdf engine.
        return xr.open_dataset(
            fsspec.filesystem("https").open(url), engine="h5netcdf", **kwargs
        )

    # Both "cache" and "local" read from disk. Honor an existing pre-download in
    # the old package data/ dir first, so users who downloaded before the cache
    # moved to pooch.os_cache() don't suddenly their kernels.
    legacy = files("climkern").joinpath("data/" + relpath)
    if legacy.is_file():
        return xr.open_dataset(legacy, **kwargs)

    if source == "local":
        path = _cache_root() / relpath
        if not path.is_file():
            raise FileNotFoundError(
                f"{relpath} not found in the local cache ({_cache_root()}). Run "
                "climkern.download() or use data_source='cache' or 'stream'."
            )
        return xr.open_dataset(path, **kwargs)

    # default "cache": pooch serves from / downloads to the cache dir, verifying
    # hashes so a corrected kernel in a new release is re-fetched automatically.
    return xr.open_dataset(_pooch().fetch(relpath), **kwargs)


def get_kern(name: str, loc: str = "TOA") -> xr.Dataset:
    """Read in a radiative kernel, from the local cache, a stream, or disk.

    The source is controlled globally by ``climkern.set_options(data_source=...)``
    (default: cache from Jetstream2 on first use). See :func:`climkern.set_options`.
    """
    relpath = f"kernels/{name}/{loc}_{name}_Kerns.nc"
    try:
        data = _open_dataset(relpath)
    except ValueError:
        data = _open_dataset(relpath, decode_times=False)
    return check_coords(data)


def make_clim(da: xr.DataArray) -> xr.DataArray:
    """Produce monthly climatology of model field."""
    try:
        clim = (
            da.groupby(da.time.dt.month)
            .mean(dim="time", skipna=True)
            .rename({"month": "time"})
        )
    except AttributeError:
        # AttributeError if time is not datetime object
        clim = da
    return clim


def get_albedo(SWup: xr.DataArray, SWdown: xr.DataArray) -> xr.DataArray:
    """Calculate the surface albedo as the ratio of upward to
    downward sfc shortwave.
    """
    # avoid dividing by 0 and assign 0 to those grid boxes
    return (SWup / SWdown.where(SWdown > 0)).fillna(0)


def check_plev(kern: xr.Dataset) -> xr.Dataset:
    """Make sure the vertical pressure units of the kernel are in Pa."""
    if kern.plev.units != "Pa":
        kern["plev"] = kern.plev * 100
        kern.plev.attrs["units"] = "Pa"
    else:
        pass
    return kern


def __calc_qs__(temp: xr.DataArray) -> xr.DataArray:
    """Calculate the saturated specific humidity
    given temperature and pressure.
    """
    if temp.plev.units == "Pa":
        pres = temp.plev / 100
    elif temp.plev.units in ["hPa", "millibars"]:
        pres = temp.plev
    else:
        warnings.warn(
            "Cannot determine units of pressure \
        coordinate. Assuming units are Pa.",
            stacklevel=2,
        )
        pres = temp.plev / 100

    if temp.units == "K":
        temp_c = temp - 273.15
        temp_c.attrs = temp.attrs
        temp_c["units"] = "C"
    elif temp.units == "C":
        temp_c = temp
    else:
        warnings.warn(
            "Warning: Cannot determine units of temperature. \
        Assuming Kelvin.",
            stacklevel=2,
        )
        temp_c = temp - 273.15
        temp_c.attrs = temp.attrs
        temp_c["units"] = "C"

    # Buck 1981 equation for saturated vapor pressure
    esl = (
        (1.0007 + 3.46e-6 * pres)
        * 6.1121
        * np.exp((17.502 * temp_c) / (240.97 + temp_c))
    )
    esi = (
        (1.0003 + 4.18e-6 * pres)
        * 6.1115
        * np.exp((22.452 * temp_c) / (272.55 + temp_c))
    )

    # conversion from vapor pressure to mixing ratio
    wsl = 0.622 * esl / (pres - esl)
    wsi = 0.622 * esi / (pres - esi)

    # use liquid water w when temp is above freezing
    ws = xr.where(temp_c > 0, wsl, wsi)

    # convert to specific humidity
    qs = ws / (1 + ws)
    qs["units"] = "kg/kg"
    return qs


def calc_q_norm(
    ctrl_ta: xr.DataArray, ctrl_q: xr.DataArray, method: int
) -> xr.DataArray:
    """Calculate the change in specific humidity for 1K warming
    assuming fixed relative humidity.
    """
    if ctrl_q.units == "g/kg":
        ctrl_q = ctrl_q / 1000

    # get saturated specific humidity from control air temps
    qs0 = __calc_qs__(ctrl_ta)

    # RH = specific humidity / sat. specific humidity
    RH = ctrl_q / qs0

    # make a DataArray for the temperature plus 1K
    ta1K = ctrl_ta + 1
    ta1K.attrs = ctrl_ta.attrs
    qs1K = __calc_qs__(ta1K)

    if method == 4:
        # get the new specific humidity using the same RH
        q1K = qs1K * RH
        q1K["units"] = "kg/kg"

        # take the difference
        dq1K = 1000 * (q1K - ctrl_q)
        return dq1K

    elif method in [3, 2]:
        dqsdT = qs1K - qs0
        dqdT = RH * dqsdT

        dlogqdT = 1000 * (dqdT / ctrl_q)
        return dlogqdT

    elif method == 1:
        dlogqdT = 1000 * (np.log(qs1K.where(qs1K > 0)) - np.log(qs0.where(qs0 > 0)))
        return dlogqdT

    else:
        raise ValueError("Please select a valid choice for the method argument.")


def check_sky(sky: str) -> str:
    """Make sure the sky argument is either all-sky or clear-sky."""
    if sky not in ["all-sky", "clear-sky"]:
        raise ValueError("The sky argument must either be all-sky or clear-sky.")
    else:
        return sky


def check_coords(ds: XrObj, ndim: int = 3) -> XrObj:
    """Universal function to check that dataset coordinates are in line with
    what the package requires.
    """
    # time
    if "time" in ds.dims:
        pass
    elif "month" in ds.dims:
        ds = ds.rename({"month": "time"})
    else:
        raise AttributeError(
            "There is no 'time' or 'month' dimension in"
            + "one of the input DataArrays. Please rename your time dimension(s)."
        )

    # lat and lon
    if "lat" not in ds.dims and "latitude" not in ds.dims:
        raise AttributeError(
            "There is no 'lat' or 'latitude' dimension in\
        one of the input DataArrays. Please rename your lat dimension(s)."
        )

    if "lon" not in ds.dims and "longitude" not in ds.dims:
        raise AttributeError(
            "There is no 'lon' or 'longitude' dimension in\
        one of the input DataArrays. Please rename your lat dimension(s)."
        )

    if ndim == 4:
        if "plev" in ds.dims:
            pass
        else:
            found = False
            for n in ["lev", "player", "level"]:
                if n in ds.dims:
                    ds = ds.rename({n: "plev"})
                    found = True
            if found is False:
                raise AttributeError(
                    "Cannot find the name of the pressure\
                coordinate. Please rename it to 'plev'."
                )
    return ds
