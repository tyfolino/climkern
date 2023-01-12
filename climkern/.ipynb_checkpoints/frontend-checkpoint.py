import xarray as xr
import cf_xarray as cfxr
import xesmf as xe
import pkgutil

from climkern.util import _make_clim,_get_albedo,_tile_data,_get_kern

def calc_alb_feedback(ctrl_rsus,ctrl_rsds,pert_rsus,pert_rsds,kern='GFDL',loc='TOA'):
    """
    Calculate the SW radiative perturbation (W/m^2) resulting from changes in surface albedo
    at the TOA or surface with the specific radiative kernel. Horizontal resolution
    is kept at input data's resolution.
    
    Parameters
    ----------
    ctrl_rsus : xarray DataArray
        Contains upwelling SW radiation at the surface in the control simulation.
        Must be 3D with coords of time, lat, and lon and in units of W/m^2.
        
    ctrl_rsds : xarray DataArray
        Contains downwelling SW radiation at the surface in the control simulation.
        Must be 3D with coords of time, lat, and lon and in units of W/m^2.
        
    pert_rsus : xarray DataArray
        Contains upwelling SW radiation at the surface in the perturbed simulation.
        Must be 3D with coords of time, lat, and lon and in units of W/m^2.
        
    pert_rsds : xarray DataArray
        Contains downwelling SW radiation at the surface in the perturbed simulation.
        Must be 3D with coords of time, lat, and lon and in units of W/m^2.

    kern : string
        String specifying the institution name of the desired kernels. Defaults to GFDL.
        
    loc : string
        String specifying level at which radiative perturbations are desired, either TOA or sfc.
        Defaults to TOA

    Returns
    -------
    rad_pert : xarray DataArray
        3D DataArray containing radiative perturbations at the desired level
        caused by changes in surface albedo. Has coordinates of time, lat, and lon.
    """
    ctrl_alb_clim = _make_clim(_get_albedo(ctrl_rsus,ctrl_rsds))
    pert_alb = _get_albedo(pert_rsus, pert_rsds)
        
    # tile climatology and take difference
    ctrl_alb_clim_tiled = _tile_data(ctrl_alb_clim,pert_alb)
    diff_alb = pert_alb - ctrl_alb_clim_tiled
    
    # read in and regrid surface albedo kernel
    kern = _get_kern(kern,loc).sw_a
    regridder = xe.Regridder(kern,diff_alb,method='bilinear')
    kern = regridder(kern)
    
    # calculate feedbacks
    kern_tiled = _tile_data(kern,diff_alb)
    rad_pert = diff_alb * kern_tiled * 100
    return rad_pert