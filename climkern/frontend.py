import xarray as xr
import cf_xarray as cfxr
import xesmf as xe
import warnings

from climkern.util import make_clim,get_albedo,tile_data,get_kern, check_plev

# class Kernel:
#     def __init__(
#         self,
#         ds,
#         name,
#         logq = False,
#         loc='TOA'
#     ):
#         """
#         Make radiative kernel object
#         Parameters
#         ----------
#         data : xarray Dataset
#             Contains variables corresponding to all-sky and clear-sky
#             radiative fluxes. Note that the variable name requirements are
#             strict: sw/lw[clr]_[a|T|q|Ts]. Will return a ValueError
#             if variables are missing.
#         name : str
#             Name of model from which the kernels were produced.
#         logq : bool, optional
#             Whether the water vapor feedback calculations require
#             the difference in the natural log of specific humidity.
#         loc : str, optional
#             Kernel levels. Options are
#             - 'TOA'
#             - 'sfc'
            
#         Returns
#         -------
#         kernel : Kernel object
#         """
#         self.loc = loc
#         self.name = name
#         self.logq = logq
#         try:
#             self.sw_a = ds.sw_a
#             self.sw_q = ds.sw_q
#             self.swclr_a = ds.swclr_a
#             self.swclr_q = ds.swclr_q
#             self.PS = ds.PS
#             self.lw_q = ds.lw_q
#             self.lwclr_q = ds.lwclr_q
#             self.lw_t = ds.lw_t
#             self.lw_ts = ds.lw_ts
#             self.lwclr_t = ds.lwclr_t
#             self.lwclr_ts = ds.lwclr_ts
#             self.plev = ds.plev
#         except(AttributeError):
#             raise ValueError('Kernel input data is missing required variables.')

def calc_alb_feedback(ctrl_rsus,ctrl_rsds,pert_rsus,pert_rsds,kern,loc='TOA'):
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
    a_feedback : xarray DataArray
        3D DataArray containing radiative perturbations at the desired level
        caused by changes in surface albedo. Has coordinates of time, lat, and lon.
    """
    ctrl_alb_clim = make_clim(get_albedo(ctrl_rsus,ctrl_rsds))
    pert_alb = get_albedo(pert_rsus, pert_rsds)
        
    # tile climatology and take difference
    ctrl_alb_clim_tiled = tile_data(ctrl_alb_clim,pert_alb)
    diff_alb = pert_alb - ctrl_alb_clim_tiled
    
    # read in and regrid surface albedo kernel
    # kernel = Kernel(get_kern(kern,loc),kern)
    kernel = get_kern(kern,loc)
    regridder = xe.Regridder(kernel.sw_a,diff_alb,method='bilinear',
                             reuse_weights=False,periodic=True)
    kernel = regridder(kernel.sw_a)
    
    # calculate feedbacks
    kernel_tiled = tile_data(kernel,diff_alb)
    a_feedback = diff_alb * kernel_tiled * 100
    return a_feedback

def calc_T_feedbacks(ctrl_ta,ctrl_ts,ctrl_ps,pert_ta,pert_ts,pert_ps,
                     pert_trop,kern,loc='TOA'):
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
    # ctrl_ta_clim = make_clim(ctrl_ta.where(ctrl_ps > ctrl_ta.plev))
    ctrl_ta_clim = make_clim(ctrl_ta)
    ctrl_ts_clim = make_clim(ctrl_ts)
    
    # calculate change in Ta & Ts
    # diff_ta = (pert_ta - tile_data(ctrl_ta_clim,pert_ta)).where(
    #     pert_ps > pert_ta.plev).where(pert_trop < pert_ta.plev)
    diff_ta = (pert_ta - tile_data(ctrl_ta_clim,pert_ta))
    diff_ts = pert_ts - tile_data(ctrl_ts_clim,pert_ts)
    
    # read in and regrid temperature kernel
    # kernel = Kernel(check_plev(get_kern(kern,loc),diff_ta),kern)
    kernel,is_Pa = check_plev(get_kern(kern,loc),diff_ta)
    regridder = xe.Regridder(kernel.lw_t,diff_ts,method='bilinear',
                             reuse_weights=False,periodic=True)
    ta_kernel = tile_data(regridder(kernel.lw_t),diff_ta)
    ts_kernel = tile_data(regridder(kernel.lw_ts),diff_ta)

    # regrid diff_ta to kernel pressure levels
    diff_ta = diff_ta.interp_like(ta_kernel)
    
    # construct a 4D DataArray corresponding to layer thickness
    # for vertical integration later
    # this is achieved by finding the midpoints between pressure levels
    # and bounding that array with surface pressure below
    # and TOA (p=0) above
    aligned = xr.align(diff_ta.plev[1:],diff_ta.plev[:-1],join='override')
    mids = xr.broadcast((aligned[0] + aligned[1])/2,pert_ps)[0]
    ps_expand = pert_ps.expand_dims(dim={"plev":[diff_ta.plev[0]]},axis=1)
    
    TOA = xr.zeros_like(ps_expand)
    TOA['plev']=ps_expand.plev*0

    # this if/else statement accounts for potentially reversed pressure axis direction
    if(diff_ta.plev[0] > diff_ta.plev[-1]):
        ilevs = xr.concat([ps_expand,mids,TOA],dim='plev')
        sign_change = -1
    else:
        ilevs = xr.concat([TOA,mids,ps_expand],dim='plev')
        sign_change = 1

    # make points above tropopause equal to tropopause height
    # make points below surface pressure equal to surface pressure
    ilevs = ilevs.where(ilevs>pert_trop,pert_trop).where(ilevs<pert_ps,pert_ps)

    # get the layer thickness by taking finite difference along pressure axis
    dp = sign_change * ilevs.diff(dim='plev',label='lower')
    # if dp is in Pascals, just divide by 100 to make it hPa
    if(is_Pa == True):
        dp = dp/100

    # override pressure axis so xarray doesn't throw a fit
    dp['plev'] = diff_ta.plev
                    
    # calculate feedbacks
    # for lapse rate, use the deviation of the air temperature response
    # from vertically uniform warming
    lr_feedback = ((ta_kernel * (diff_ta - diff_ts).fillna(0)) * dp/100
                  ).sum(dim='plev')
    
    # for planck, assume vertically uniform warming and 
    # account for surface temperature change
    planck_feedback = ((ts_kernel * diff_ts) + (ta_kernel * xr.broadcast(
        diff_ts,diff_ta)[0].fillna(0) * dp/100).sum(dim='plev'))
       
    return(lr_feedback,planck_feedback)


def calc_q_feedbacks(ctrl_q,ctrl_ps,pert_q,pert_ps,pert_trop,kern,loc='TOA'):
    """
    Calculate the LW and SW radiative perturbations (W/m^2) using model output
    specific humidity and the chosen radiative kernel. Horizontal resolution
    is kept at input data's resolution.
    
    Parameters
    ----------
    ctrl_q : xarray DataArray
        Contains specific on pressure levels in the control simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "1", "kg/kg", or "g/kg".
        
    ctrl_ps : xarray DataArray
        Contains the surface pressure in the control simulation. Must
        be 3D with coords of time, lat, and lon and units of Pa.
        
    pert_q : xarray DataArray
        Contains specific on pressure levels in the perturbed simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "1", "kg/kg", or "g/kg".
        
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
    lw_q_feedback : xarray DataArray
        3D DataArray containing the vertically integrated radiative perturbations
        from changes in specific humidity (longwave). Has coordinates
        of time, lat, and lon.
        
    sw_q_feedback : xarray DataArray
        3D DataArray containing the vertically integrated radiative perturbations
        from changes in specific humidity (shortwave). Has coordinates of time, 
        lat, and lon.
    """
    # mask values below the surface, make climatology
    ctrl_q_clim = make_clim(ctrl_q)

    # if q has units of unity or kg/kg, we will have to 
    # multiply by 1000 later on to make it g/kg
    if(ctrl_q.units in ['1','kg/kg']):
        conv_factor = 1000
    elif(ctrl_q.units in ['g/kg']):
        conv_factor = 1
    else:
        warnings.warn("Warning: cannot determine units of q. Assuming kg/kg.")
        conv_factor = 1000

    # calculate change in q
    diff_q = (pert_q - tile_data(ctrl_q_clim,pert_q))
    
    # read in and regrid water vapor kernel
    # kernel = Kernel(check_plev(get_kern(kern,loc),diff_ta),kern)
    kernel,is_Pa = ck.util.check_plev(ck.util.get_kern('BMRC','TOA'),diff_q)
    regridder = xe.Regridder(kernel.lw_q,diff_q,method='bilinear',
                             extrap_method="nearest_s2d",
                             reuse_weights=False,periodic=True)
    qlw_kernel = ck.util.tile_data(regridder(kernel.lw_q),diff_q)
    qsw_kernel = ck.util.tile_data(regridder(kernel.sw_q),diff_q)

    # regrid diff_q to kernel pressure levels
    diff_q = diff_q.interp_like(qlw_kernel)
    
    # construct a 4D DataArray corresponding to layer thickness
    # for vertical integration later
    # this is achieved by finding the midpoints between pressure levels
    # and bounding that array with surface pressure below
    # and TOA (p=0) above
    aligned = xr.align(diff_q.plev[1:],diff_q.plev[:-1],join='override')
    mids = xr.broadcast((aligned[0] + aligned[1])/2,pert_ps)[0]
    ps_expand = pert_ps.expand_dims(dim={"plev":[diff_q.plev[0]]},axis=1)
    
    TOA = xr.zeros_like(ps_expand)
    TOA['plev']=ps_expand.plev*0

    # this if/else statement accounts for potentially reversed pressure axis direction
    if(diff_q.plev[0] > diff_q.plev[-1]):
        ilevs = xr.concat([ps_expand,mids,TOA],dim='plev')
        sign_change = -1
    else:
        ilevs = xr.concat([TOA,mids,ps_expand],dim='plev')
        sign_change = 1

    # make points above tropopause equal to tropopause height
    # make points below surface pressure equal to surface pressure
    ilevs = ilevs.where(ilevs>pert_trop,pert_trop).where(ilevs<pert_ps,pert_ps)

    # get the layer thickness by taking finite difference along pressure axis
    dp = ilevs.diff(dim='plev',label='lower') * sign_change
    # if dp is in Pascals, just divide by 100 to make it hPa
    if(is_Pa == True):
        dp = dp/100

    # override pressure axis so xarray doesn't throw a fit
    dp['plev'] = diff_q.plev
                    
    # calculate feedbacks
    # first, we need to see if the units of diff_q a
    qlw_feedback = (qlw_kernel * diff_q * conv_factor * dp/100).sum(dim='plev')
    qsw_feedback = (qsw_kernel * diff_q * conv_factor * dp/100).sum(dim='plev')
    
    return(qlw_feedback,qsw_feedback)
