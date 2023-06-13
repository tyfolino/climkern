import xarray as xr
import cf_xarray as cfxr
import xesmf as xe
import pkgutil
import io

def tile_data(to_tile,new_shape):
    """tile dataset along time axis to match another dataset"""
    if(len(new_shape.time) % 12 != 0):
        raise ValueError('dataset time dimension must be divisible by 12')
    tiled = xr.concat([to_tile for i in range(
        int(len(new_shape.time)/12))],dim='time')
    tiled['time'] = new_shape.time
    return(tiled)

def get_kern(name,loc='TOA'):
    """read in kernel from local directory"""
    path = 'data/'+name + '/' +loc + '_' + str(name) + "_Kerns.nc"
    try:
        data = xr.open_dataset(io.BytesIO(pkgutil.get_data('climkern',path)))
    except(ValueError):
        data = xr.open_dataset(io.BytesIO(pkgutil.get_data('climkern',path)),decode_times=False)
    return _check_coords(data)

def make_clim(da):
    "Produce monthly climatology of model field."
    time = _get_time(da) # also checks to see if time exists
    clim = da.groupby(time.dt.month).mean(dim='time',skipna=True).rename({'month':'time'})
    return clim

def _check_coords(da):
    """Try to grab time, lat, lon, and plev coordinates
    from DataArray.
    """
    # check to see if lat/lon are CF-compliant for regridding
    # if they are, rename to lat/lon for xesmf
    try:
        da = da.rename({da.cf['latitude'].name:'lat',
                       da.cf['longitude'].name:'lon'})
    except (KeyError, AttributeError, ValueError):
        # KeyError if cfxr doesn't detect the coords
        # AttributeError if ds is a dict
        raise ValueError('horizontal coordinates must be CF-compliant')

    # grab vertical level
    vert_names = ['plev','player','lev']
    for v in vert_names:
        if(v in da.dims):
            vert_coord = v
            break
        elif(v == vert_names[-1]):
            raise AttributeError('Could not find vertical coordinate.')
    da = da.rename({vert_coord:'plev'})

    # time dimension
    if('time' not in da.dims):
        try:
            da = da.rename({'month':'time'})
        except(ValueError):
            raise AttributeError('There is no dimension named \'time\' or \'month\'.')

    return(da)
            
def _get_time(da):
    """Return the time dimension from ds."""
    try:
        return da.time
    except(AttributeError):
        # AttributeError if variable does not exist in ds
        raise ValueError('dataset does not have a dimension called "time"')
    return time

def get_albedo(SWup,SWdown):
    """Calculate the surface albedo as the ratio of upward to downward sfc shortwave."""
    # avoid dividing by 0 and assign 0 to those grid boxes
    return (SWup/SWdown.where(SWdown>0)).fillna(0)

def check_plev(kern,output):
    """Make sure the vertical pressure units of the kernel match those of the
    model output. If not, return the kern with updated pressure levels."""
    is_Pa = False
    if((output.plev.units == 'Pa') and (kern.plev.units != 'Pa')):
        kern['plev'] = kern.plev * 100
        kern.plev.attrs['units'] = 'Pa'
        is_Pa = True
    elif((kern.plev.units == 'Pa') and (output.plev.units != 'Pa')):
        output['plev'] = output.plev * 100
        output.plev.attrs['units'] = 'Pa'
        is_Pa = True
    return(kern,is_Pa)