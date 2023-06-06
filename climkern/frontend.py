import xarray as xr
import cf_xarray as cfxr
import xesmf as xe

from climkern.util import make_clim,get_albedo,tile_data,get_kern, check_plev

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
            self.plev = ds.plev
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
    kernel = Kernel(get_kern(kern,loc),kern)
    regridder = xe.Regridder(kernel.sw_a,diff_alb,method='bilinear',reuse_weights=True)
    kernel = regridder(kernel.sw_a)
    
    # calculate feedbacks
    kernel_tiled = tile_data(kernel,diff_alb)
    rad_pert = diff_alb * kernel_tiled * 100
    return rad_pert

def calc_T_feedbacks(ctrl_ta,ctrl_ts,ctrl_ps,pert_ta,pert_ts,pert_ps,pert_trop,kern,loc='TOA'):
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
        
    ctrl_ps : xarray DataArray
        Contains the surface pressure in the control simulation. Must
        be 3D with coords of time, lat, and lon and units of Pa.
        
    pert_ta : xarray DataArray
        Contains air temperature on standard pressure levels in the perturbed simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of K.
        
    pert_ts : xarray DataArray
        Contains surface skin temperature in the perturbed simulation.
        Must be 3D with coords of time, lat, and lon and with units of K.
        
    pert_ps : xarray DataArray
        Contains the surface pressure in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon and units of Pa.
        
    pert_trop : xarray DataArray
        Contains the tropopause height in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon and units of Pa.

    kern : string
        String specifying the institution name of the desired kernels. Defaults to GFDL.
        
    loc : string
        String specifying level at which radiative perturbations are desired, either TOA or sfc.
        Defaults to TOA

    Returns
    -------
    lr_feedback : xarray DataArray
        3D DataArray containing the vertically integrated radiative perturbations
        caused by changes in lapse rate. Has coordinates
        of time, lat, and lon.
        
    planck_feedback : xarray DataArray
        3D DataArray containing the vertically integrated radiative perturbations
        caused by changes in a vertically-uniform warming. Has coordinates of time, 
        lat, and lon.
    """
    # mask values below the surface, make climatology
    ctrl_ta_clim = make_clim(ctrl_ta.where(ctrl_ps > ctrl_ta.plev))
    ctrl_ts_clim = make_clim(ctrl_ts)
    
    # calculate change in Ta/Ts
    diff_ta = (pert_ta - tile_data(ctrl_ta_clim,pert_ta)).where(
        pert_ps > pert_ta.plev).where(pert_trop < pert_ta.plev)
    diff_ts = pert_ts - tile_data(ctrl_ts_clim,pert_ts)
    
    # read in and regrid temperature kernel
    kernel = Kernel(check_plev(get_kern(kern,loc),diff_ta),kern)
    regridder = xe.Regridder(kernel.lw_t,diff_ts,method='bilinear',reuse_weights=True)
    ta_kernel = tile_data(regridder(kernel.lw_t),diff_ta)
    ts_kernel = tile_data(regridder(kernel.lw_ts),diff_ta)

    # regrid diff_ta to kernel pressure levels
    diff_ta = diff_ta.interp(plev=kernel.plev)
    
    # # ignore vertical levels that do not exist in the kernel
    # diff_ta = diff_ta.sel(plev=ta_kernel.plev)    
    
    # construct a 4D DataArray corresponding to layer thickness
    # for vertical integration later
    aligned = xr.align(diff_ta.plev[1:],diff_ta.plev[:-1],join='override')
    mids = xr.broadcast((aligned[0] + aligned[1])/2,pert_ps)[0]
    ps_expand = pert_ps.expand_dims(dim={"plev":[diff_ta.plev[0]]},axis=1)
    
    TOA = xr.zeros_like(ps_expand)
    TOA['plev']=ps_expand.plev*0
    
    ilevs = xr.concat([ps_expand,mids,TOA],dim='plev')
    ilevs = ilevs.where(ilevs>pert_trop,pert_trop).where(ilevs<pert_ps,pert_ps)
    
    dp = -1 * ilevs.diff(dim='plev')
    dp['plev'] = diff_ta.plev
                    
    # calculate feedbacks
    # for lapse rate, use the deviation of the air temperature response
    # from vertically uniform warming
    lr_feedback = ((ta_kernel * (diff_ta - diff_ts).fillna(0)) * dp/10000
                  ).sum(dim='plev')
    
    # for planck, assume vertically uniform warming and 
    # account for surface temperature change
    planck_feedback = ((ts_kernel * diff_ts) + (ta_kernel * xr.broadcast(
        diff_ts,diff_ta)[0].fillna(0) * dp/10000).sum(dim='plev'))
       
    return(lr_feedback,planck_feedback)