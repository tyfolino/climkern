import xarray as xr
import cf_xarray as cfxr
import xesmf as xe

from climkern.util import make_clim,get_albedo,tile_data,get_kern

class Kernel:
    def __init__(
        self,
        ds,
        name,
        logq = False,
        loc='TOA'
    ):
        """
        Make radiative kernel object
        Parameters
        ----------
        data : xarray Dataset
            Contains variables corresponding to all-sky and clear-sky
            radiative fluxes. Note that the variable name requirements are
            strict: sw/lw[clr]_[a|T|q|Ts]. Will return a ValueError
            if variables are missing.
        name : str
            Name of model from which the kernels were produced.
        logq : bool, optional
            Whether the water vapor feedback calculations require
            the difference in the natural log of specific humidity.
        loc : str, optional
            Kernel levels. Options are
            - 'TOA'
            - 'sfc'
            
        Returns
        -------
        kernel : Kernel object
        """
        self.loc = loc
        self.name = name
        self.logq = logq
        try:
            self.sw_a = ds.sw_a
            self.sw_q = ds.sw_q
            self.swclr_a = ds.swclr_a
            self.swclr_q = ds.swclr_q
            self.PS = ds.PS
            self.lw_q = ds.lw_q
            self.lwclr_q = ds.lwclr_q
            self.lw_t = ds.lw_t
            self.lw_ts = ds.lw_ts
            self.lwclr_t = ds.lwclr_t
            self.lwclr_ts = ds.lwclr_ts
        except(AttributeError):
            raise ValueError('Kernel input data is missing required variables.')

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
    ctrl_alb_clim = make_clim(get_albedo(ctrl_rsus,ctrl_rsds))
    pert_alb = get_albedo(pert_rsus, pert_rsds)
        
    # tile climatology and take difference
    ctrl_alb_clim_tiled = tile_data(ctrl_alb_clim,pert_alb)
    diff_alb = pert_alb - ctrl_alb_clim_tiled
    
    # read in and regrid surface albedo kernel
    kernel = Kernel(_get_kern(kern,loc),kern)
    regridder = xe.Regridder(kernel.sw_a,diff_alb,method='bilinear',reuse_weights=True)
    kernel = regridder(kernel.sw_a)
    
    # calculate feedbacks
    kernel_tiled = _tile_data(kernel,diff_alb)
    rad_pert = diff_alb * kernel_tiled * 100
    return rad_pert

def calc_LR_feedback(ctrl_ta,ctrl_ts,ctrl_PS,pert_ta,pert_ts,pert_PS,kern='GFDL',loc='TOA'):
    """
    Calculate the LW radiative perturbations (W/m^2) from changes in surface skin
    and air temperature at the TOA or surface with the specified radiative kernel.
    at the TOA or surface with the specific radiative kernel. Horizontal resolution
    is kept at input data's resolution.
    
    Parameters
    ----------
    ctrl_ta : xarray DataArray
        Contains air temperature on standard pressure levels in the control simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of K.
        
    ctrl_ts : xarray DataArray
        Contains surface skin temperature in the control simulation.
        Must be 3D with coords of time, lat, and lon and with units of K.
        
    ctrl_PS : xarray DataArray
        Contains the surface pressure in the control simulation. Must
        be 3D with coords of time, lat, and lon and units of Pa.
        
    pert_ta : xarray DataArray
        Contains air temperature on standard pressure levels in the perturbed simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of K.
        
    pert_ts : xarray DataArray
        Contains surface skin temperature in the perturbed simulation.
        Must be 3D with coords of time, lat, and lon and with units of K.
        
    pert_PS : xarray DataArray
        Contains the surface pressure in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon and units of Pa.

    kern : string
        String specifying the institution name of the desired kernels. Defaults to GFDL.
        
    loc : string
        String specifying level at which radiative perturbations are desired, either TOA or sfc.
        Defaults to TOA

    Returns
    -------
    rad_pert : xarray DataArray
        3D DataArray containing the vertically integrated radiative perturbations
        at the desired level caused by changes in temperature. Has coordinates
        of time, lat, and lon.
    """
    # mask values below the surface, make climatology
    ctrl_ta_clim = make_clim(ctrl_ta.where(ctrl_ps > ctrl_ta.plev))
    ctrl_ts_clim = make_clim(ctrl_ts)
    
    # calculate change in Ta/Ts
    diff_ta = (pert_ta - tile_data(ctrl_ta_clim,pert_ta)).where(
        pert_PS > pert_ta.plev)
    diff_ts = pert_ts - tile_data(ctrl_ts_clim,pert_ts)
    
    # read in and regrid temperature kernel
    kernel = Kernel(get_kern(kern,loc),kern)
    regridder = xe.Regridder(kernel.lw_t,diff_ts,method='bilinear',reuse_weights=True)
    ta_kernel = regridder(kernel.lw_t)
    ts_kernel = regridder(kernel.lw_ts)
    
    # calculate feedbacks
    lr_feedback = ta_kernel * (diff_ta - diff_ts)
    planck_feedback = (ts_kernel * diff_ts) + (ta_kernel * xr.broadcast(diff_ts,diff_ta)[0])
    
    # to-do:
    # put check for vertical coordinates in kernels (units)
    # vertically integrate feedbacks