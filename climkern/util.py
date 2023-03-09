import xarray as xr
import cf_xarray as cfxr
import xesmf as xe
import pkgutil
import climkern as ck

def _tile_data(to_tile,new_shape):
    """tile dataset along time axis to match another dataset"""
    if(len(new_shape.time) % 12 != 0):
        raise ValueError('dataset time dimension must be divisible by 12')
    tiled = xr.concat([to_tile for i in range(
        int(len(new_shape.time)/12))],dim='time')
    tiled['time'] = new_shape.time
    return(tiled)

def _get_kern(name,loc='TOA'):
    """read in kernel from local directory"""
    path = 'data/'+loc + '_' + str(name) + "_Kerns.nc"
    data = xr.open_dataset(pkgutil.get_data('climkern',path))
    if(('latitude' in data.coords) or ('longitude' in data.coords)):
        data = data.rename({'latitude':'lat','longitude':'lon'})
    return data

def _make_clim(da):
    "Produce monthly climatology of model field."
    time = _get_time(da) # also checks to see if time exists
    clim = da.groupby(time.dt.month).mean(dim='time').rename({'month':'time'})
    return clim

def _get_lat_lon(da):
    """Return lat and lon from da."""
    if ('lat' in da and 'lon' in da):
        return da.lat, da.lon
    try:
        lat = da.cf['latitude']
        lon = da.cf['longitude']
    except (KeyError, AttributeError, ValueError):
        # KeyError if cfxr doesn't detect the coords
        # AttributeError if ds is a dict
        raise ValueError('dataset must include "lat"/"lon" dimension or be CF-compliant')
    return lat, lon

def _get_time(da):
    """Return the time dimension from ds."""
    try:
        return da.time
    except(AttributeError):
        # AttributeError if variable does not exist in ds
        raise ValueError('dataset does not have a dimension called "time"')
    return time

def _get_albedo(SWup,SWdown):
    """Calculate the surface albedo as the ratio of upward to downward sfc shortwave."""
    # avoid dividing by 0 and assign 0 to those grid boxes
    return (SWup/SWdown.where(SWdown>0)).fillna(0)
