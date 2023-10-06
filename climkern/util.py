import xarray as xr
import cf_xarray as cfxr
import xesmf as xe
from importlib_resources import files
import warnings
import numpy as np

# monkey patch Python warnings format function
def custom_formatwarning(msg,cat,*args,**kwargs):
    return(str(cat.__name__) + ': ' + str(msg) + '\n')
warnings.formatwarning = custom_formatwarning

def check_var_units(da,var):
    """Check to see if the xarray DataArray has a units attribute."""
    if('units' not in da.attrs):
        if(var=='q'):
            warnings.warn('No units found for input q. Assuming kg/kg.')
            return(da.assign_attrs({'units':'kg/kg'}))
        elif(var=='T'):
            warnings.warn('No units found for input T. Assuming K.')
            return(da.assign_attrs({'units':'K'}))
    else:
        return(da)

def check_plev_units(da):
    if('units' not in da.plev.attrs):
        warnings.warn('No units found for input vertical coordinate. Assuming Pa.')
        plev = da.plev.assign_attrs({'units':'Pa'})
        return(da.assign_coords({'plev':plev}))
    else:
        return(da)

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
        data = xr.open_dataset(files('climkern').joinpath(path))
    except(ValueError):
        data = xr.open_dataset(files('climkern').joinpath(path),decode_times=False)
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
        da = da.rename_dims({da.cf['latitude'].name:'lat',
                             da.cf['longitude'].name:'lon'})
    except(ValueError):
        # ValueError if the dims are already named lat/lon
        pass
    except (KeyError, AttributeError):
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
    try:
        da = da.rename_dims({vert_coord:'plev'})
    except(ValueError):
        pass

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
    try:
        if((output.plev.units == 'Pa') and (kern.plev.units != 'Pa')):
            kern['plev'] = kern.plev * 100
            kern.plev.attrs['units'] = 'Pa'
            is_Pa = True
        elif((kern.plev.units == 'Pa') and (output.plev.units != 'Pa')):
            output['plev'] = output.plev * 100
            output.plev.attrs['units'] = 'Pa'
            is_Pa = True
        elif((kern.plev.units == 'Pa') and (output.plev.units == 'Pa')):
            is_Pa = True
    # raise warning if plev doesn't have units attribute
    except(AttributeError):
        warnings.warn('No "units" attribute on pressure axis.'+
                      ' Assuming Pa. Consider adding units.')
        if(kern.plev.units != 'Pa'):
            kern['plev'] = kern.plev * 100
            kern.plev.attrs['units'] = 'Pa'
        is_Pa=True
    return(kern,is_Pa)

def __calc_qs__(temp):
    """Calculate either the saturated specific humidity or mixing ratio
    given temperature and pressure."""
    if(temp.plev.units=='Pa'):
        pres = temp.plev/100
    elif(temp.plev.units in ['hPa','millibars']):
        pres = temp.plev
    else:
        warnings.warn('Cannot determine units of pressure \
        coordinate. Assuming units are Pa.')
        pres = temp.plev/100

    if(temp.units=='K'):
        temp_c = temp - 273.15
        temp_c.attrs = temp.attrs
        temp_c['units'] = 'C'
    elif(temp.units=='C'):
        temp_c = temp
    else:
        warnings.warn('Warning: Cannot determine units of temperature. \
        Assuming Kelvin.')
        temp_c = temp - 273.15
        temp_c.attrs = temp.attrs
        temp_c['units'] = 'C'

    # Buck 1981 equation for saturated vapor pressure
    esl = (1.0007 + 3.46e-6 * pres) * 6.1121 * np.exp(
        (17.502 * temp_c)/(240.97+temp_c))
    esi = (1.0003 + 4.18e-6 * pres) * 6.1115 * np.exp(
        (22.452*temp_c)/(272.55 + temp_c))

    # conversion from vapor pressure to mixing ratio
    wsl = 0.622 * esl / (pres - esl)
    wsi = 0.622 * esi / (pres - esi)

    # use liquid water w when temp is above freezing
    ws = xr.where(temp_c>0,wsl,wsi)

    # convert to specific humidity
    qs = ws / (1+ws)
    qs['units'] = 'kg/kg'
    return(qs)

def calc_q_norm(ctrl_ta,ctrl_q,logq=False):
    """Calculate the change in specific humidity for 1K warming
    assuming fixed relative humidity."""
    if(ctrl_q.units=='g/kg'):
        ctrl_q = ctrl_q/1000

    # get saturated specific humidity from control air temps
    qs0 = __calc_qs__(ctrl_ta)

    # RH = specific humidity / sat. specific humidity
    RH = ctrl_q / qs0

    # make a DataArray for the temperature plus 1K
    ta1K = ctrl_ta + 1
    ta1K.attrs = ctrl_ta.attrs
    qs1K = __calc_qs__(ta1K)

    if(logq==False):
        # get the new specific humidity using the same RH
        q1K = qs1K * RH
        q1K['units'] = 'kg/kg'
    
        # take the difference
        dq1K = 1000 * (q1K - ctrl_q)
        dq1K['units'] = 'g/kg/K'

        return(dq1K)
    else:
        dqsdT = qs1K - qs0
        dqsdT = ((RH / ctrl_q) * dqsdT * 1000)
        dqsdT['units'] = 'g/kg/K'

        return(dqsdT)
def check_sky(sky):
    """Make sure the sky argument is either all-sky or clear-sky"""
    if(sky not in ['all-sky','clear-sky']):
        raise ValueError('The sky argument must either be all-sky or clear-sky.')
    else:
        return(sky)
