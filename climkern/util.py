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

warnings.filterwarnings('ignore','.*does not create an index anymore.*')

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

def make_tropo(PS):
    """Use the surface pressure DataArray to make a makeshift tropopause."""
    tropo = (3e4 - 2e4*np.cos(np.deg2rad(PS.lat))).broadcast_like(PS)
    return(tropo)

def check_plev_units(da):
    if('units' not in da.plev.attrs):
        warnings.warn('No units found for input vertical coordinate. Assuming Pa.')
        plev = da.plev.assign_attrs({'units':'Pa'})
        return(da.assign_coords({'plev':plev}))
    elif(da.plev.units in ['hPa','mb','millibars']):
        da['plev'] = da.plev * 100
        da.plev.attrs['units'] = 'Pa'
        return(da)
    else:
        return(da)

def check_pres_units(da,var_name):
    if('units' not in da.attrs):
        warnings.warn("Could not determine units of " + var_name +
                      ". Assuming Pa.")
        da.assign_attrs({'units':'Pa'})
    elif(da.units in ['hPa','mb','millibars']):
        da = da * 100
        da.attrs['units'] = 'Pa'
        return(da)
    else:
        return(da)

def tile_data(to_tile,new_shape):
    """tile dataset along time axis to match another dataset"""
    # new_shape = _check_time(new_shape)
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
    return check_coords(data)

def make_clim(da):
    "Produce monthly climatology of model field."
    # da = _check_time(da)
    try:
        clim = da.groupby(da.time.dt.month).mean(dim='time',skipna=True).rename(
            {'month':'time'})
    except(TypeError):
        # TypeError if time is not datetime object
        clim = da
    return clim

def get_albedo(SWup,SWdown):
    """Calculate the surface albedo as the ratio of upward to downward sfc shortwave."""
    # avoid dividing by 0 and assign 0 to those grid boxes
    # SWup = _check_time(SWup)
    # SWdown = _check_time(SWdown)
    return (SWup/SWdown.where(SWdown>0)).fillna(0)

def check_plev(kern):
    """Make sure the vertical pressure units of the kernel are in Pa."""
    if(kern.plev.units != 'Pa'):
        kern['plev'] = kern.plev*100
        kern.plev.attrs['units'] = 'Pa'
    else:
        pass
    return(kern)

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

def check_coords(ds,ndim=3):
    """Universal function to check that dataset coordinates are in line with
    what the package requires."""
    # time
    if('time' in ds.dims):
        pass
    elif('month' in ds.dims):
        ds = ds.rename({'month':'time'})
    else:
        raise AttributeError('There is no \'time\' or \'month\' dimension in\
        one of the input DataArrays. Please rename your time dimension(s).')

    # lat and lon
    if('lat' not in ds.dims and 'latitude' not in ds.dims):
        raise AttributeError('There is no \'lat\' or \'latitude\' dimension in\
        one of the input DataArrays. Please rename your lat dimension(s).')

    if('lon' not in ds.dims and 'longitude' not in ds.dims):
        raise AttributeError('There is no \'lon\' or \'longitude\' dimension in\
        one of the input DataArrays. Please rename your lat dimension(s).')

    if(ndim==4):
        if('plev' in ds.dims):
            pass
        else:
            bool = False
            for n in ['lev','player','level']:
                if(n in ds.dims):
                    ds = ds.rename({n:'plev'})
                    bool = True
            if(bool == False):
                raise AttributeError('Cannot find the name of the pressure\
                coordinate. Please rename it to \'plev\'.')
    return(ds)