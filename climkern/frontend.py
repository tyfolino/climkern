import xarray as xr
import cf_xarray as cfxr
import xesmf as xe
import warnings
import numpy as np

from climkern.util import make_clim,get_albedo,tile_data,get_kern
from climkern.util import check_plev,calc_q_norm,check_sky,check_coords
from climkern.util import check_var_units,custom_formatwarning
from climkern.util import check_pres_units,check_plev_units,make_tropo
from climkern.util import get_dp

warnings.formatwarning = custom_formatwarning

warnings.filterwarnings('ignore','.*does not create an index anymore.*')

def calc_alb_feedback(ctrl_rsus,ctrl_rsds,pert_rsus,pert_rsds,kern='GFDL',
                      sky="all-sky"):
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

    sky : string
        String specifying whether the all-sky or clear-sky feedbacks should be used.

    Returns
    -------
    a_feedback : xarray DataArray
        3D DataArray containing radiative perturbations at the desired level
        caused by changes in surface albedo. Has coordinates of time, lat, and lon.
    """
    # determine whether we want all-sky or clear-sky
    alb_key = 'sw_a' if check_sky(sky)=='all-sky' else 'swclr_a'

    # check dataset dimensions
    for d in [ctrl_rsus,ctrl_rsds,pert_rsus,pert_rsds]:
        d = check_coords(d)
    
    ctrl_alb_clim = make_clim(get_albedo(ctrl_rsus,ctrl_rsds))
    pert_alb = get_albedo(pert_rsus, pert_rsds)
        
    # tile climatology and take difference
    ctrl_alb_clim_tiled = tile_data(ctrl_alb_clim,pert_alb)
    diff_alb = pert_alb - ctrl_alb_clim_tiled
    
    # read in and regrid surface albedo kernel
    kernel = get_kern(kern)
    regridder = xe.Regridder(kernel[alb_key],diff_alb,method='bilinear',
                             reuse_weights=False,periodic=True,
                            extrap_method='nearest_s2d')
    kernel = regridder(kernel[alb_key])
    
    # calculate feedbacks
    kernel_tiled = tile_data(kernel,diff_alb)
    a_feedback = diff_alb * kernel_tiled * 100
    return a_feedback

def calc_T_feedbacks(ctrl_ta,ctrl_ts,ctrl_ps,pert_ta,pert_ts,pert_ps,
                     pert_trop=None,kern='GFDL',sky='all-sky',fixRH=False):
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
        be 3D with coords of time, lat, and lon.
        
    pert_ta : xarray DataArray
        Contains air temperature on standard pressure levels in the perturbed simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of K.
        
    pert_ts : xarray DataArray
        Contains surface skin temperature in the perturbed simulation.
        Must be 3D with coords of time, lat, and lon and with units of K.
        
    pert_ps : xarray DataArray
        Contains the surface pressure in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon.
        
    pert_trop : xarray DataArray
        Contains the tropopause height in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon.

    kern : string
        String specifying the institution name of the desired kernels.
        Defaults to GFDL.
        
    sky : string
        String specifying whether the all-sky or clear-sky feedbacks 
        should be used.

    fixRH : boolean
        Specifies whether to calculate alternative Planck and lapse rate
        feedbacks using relative humidity as a state variable, as outlined
        in Held & Shell (2012).

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
    t_key = 'lw_t' if check_sky(sky)=='all-sky' else 'lwclr_t'
    ts_key = 'lw_ts' if sky=='all-sky' else 'lwclr_ts'
    if(fixRH==True):
        qlw_key = 'lw_q' if sky=='all-sky' else 'lwclr_q'
        qsw_key = 'sw_q' if sky=='all-sky' else 'swclr_q'

    # check model output coordinates
    for d in [ctrl_ta,pert_ta]:
        d = check_coords(d,ndim=4)
    for d in [ctrl_ts,ctrl_ps,pert_ts,pert_ps]:
        d = check_coords(d)

    # unit check
    ctrl_ta = check_var_units(check_plev_units(ctrl_ta),'T')
    pert_ta = check_var_units(check_plev_units(pert_ta),'T')
    ctrl_ts = check_var_units(ctrl_ts,'T')
    pert_ts = check_var_units(pert_ts,'T')

    # check units of ps and trop
    ctrl_ps = check_pres_units(ctrl_ps,"ctrl PS")
    pert_ps = check_pres_units(pert_ps,"pert PS")

    # make a fake tropopause if it's not provided
    if(type(pert_trop) == type(None)):
        pert_trop = make_tropo(ctrl_ps)
    else:
        pert_trop = check_coords(pert_trop)
        pert_trop = check_pres_units(pert_trop,"pert tropopause")

    # mask values below the surface, make climatology
    ctrl_ps_clim = make_clim(ctrl_ps)
    ctrl_ta_clim = make_clim(ctrl_ta)
    ctrl_ta_clim = ctrl_ta_clim.where(ctrl_ta_clim.plev < ctrl_ps_clim)
    ctrl_ts_clim = make_clim(ctrl_ts)
    
    # calculate change in Ta & Ts
    diff_ta = (pert_ta - tile_data(ctrl_ta_clim,pert_ta))
    diff_ts = pert_ts - tile_data(ctrl_ts_clim,pert_ts)
    
    # read in and regrid temperature kernel
    kernel = check_plev(get_kern(kern))
    regridder = xe.Regridder(kernel[t_key],diff_ts,method='bilinear',
                             reuse_weights=False,periodic=True,
                            extrap_method='nearest_s2d')
    ta_kernel = tile_data(regridder(kernel[t_key]),diff_ta)
    ts_kernel = tile_data(regridder(kernel[ts_key],skipna=True),diff_ta)
    if(fixRH==True):
        qlw_kernel = tile_data(regridder(kernel[qlw_key]),diff_ta)
        qsw_kernel = tile_data(regridder(kernel[qsw_key]),diff_ta)
        # overwrite ta_kernel to include q kernel
        ta_kernel = ta_kernel + qlw_kernel + qsw_kernel

    # regrid diff_ta to kernel pressure levels
    # we have to extrapolate in case the lowest model plev is above the
    # kernel's. We will also mask values below the surface.
    diff_ta = diff_ta.interp_like(ta_kernel,kwargs={
        "fill_value": "extrapolate"})

    # use get_dp in climkern.util to calculate layer thickness
    dp = get_dp(diff_ta,pert_ps,pert_trop,layer='troposphere')
    
    # calculate feedbacks
    # for lapse rate, use the deviation of the air temperature response
    # from vertically uniform warming
    lr_feedback = ((ta_kernel * (diff_ta - diff_ts).fillna(0)) * dp/10000
                  ).sum(dim='plev',min_count=1)
    
    # for planck, assume vertically uniform warming and 
    # account for surface temperature change
    planck_feedback = ((ts_kernel * diff_ts) + (ta_kernel * xr.broadcast(
        diff_ts,diff_ta)[0].fillna(0) * dp/10000).sum(dim='plev',min_count=1))
       
    return(lr_feedback,planck_feedback)


def calc_q_feedbacks(ctrl_q,ctrl_ta,ctrl_ps,pert_q,pert_ps,pert_trop=None,
                     kern='GFDL',sky='all-sky',method='pendergrass'):
    """
    Calculate the LW and SW radiative perturbations (W/m^2) using model output
    specific humidity and the chosen radiative kernel. Horizontal resolution
    is kept at input data's resolution.
    
    Parameters
    ----------
    ctrl_q : xarray DataArray
        Contains specific humidity on pressure levels in the control simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "1", "kg/kg", or "g/kg".

    ctrl_ta : xarray DataArray
        Contains air temperature on pressure levels in the control simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "K" or "C".
        
    ctrl_ps : xarray DataArray
        Contains the surface pressure in the control simulation. Must
        be 3D with coords of time, lat, and lon.
        
    pert_q : xarray DataArray
        Contains specific on pressure levels in the perturbed simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "1", "kg/kg", or "g/kg".
        
    pert_ps : xarray DataArray
        Contains the surface pressure in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon.
        
    pert_trop : xarray DataArray
        Contains the tropopause height in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon.

    kern : string
        String specifying the institution name of the desired kernels. Defaults to GFDL.

    sky : string
        String specifying whether the all-sky or clear-sky kernels should be used.

    method : string
        Specifies the method to use to calculate the specific humidity
        feedback. Options are "pendergrass" (default), "kramer", "zelinka",
        and "linear". 

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
    qlw_key = 'lw_q' if check_sky(sky)=='all-sky' else 'lwclr_q'
    qsw_key = 'sw_q' if sky=='all-sky' else 'swclr_q'

    # check model output coordinates
    for d in [ctrl_q,ctrl_ta,pert_q]:
        d = check_coords(d,ndim=4)
    for d in [ctrl_ps,pert_ps]:
        d = check_coords(d)

    # unit check
    ctrl_ta = check_var_units(check_plev_units(ctrl_ta),'T')
    ctrl_q = check_var_units(check_plev_units(ctrl_q),'q')
    pert_q = check_var_units(check_plev_units(pert_q),'q')

    # check units of ps and trop
    ctrl_ps = check_pres_units(ctrl_ps,"ctrl PS")
    pert_ps = check_pres_units(pert_ps,"pert PS")

    # make a fake tropopause if it's not provided
    if(type(pert_trop) == type(None)):
        pert_trop = make_tropo(ctrl_ps)
    else:
        pert_trop = check_coords(pert_trop)
        pert_trop = check_pres_units(pert_trop,"pert tropopause")
    
    # make climatology
    ctrl_ps_clim = make_clim(ctrl_ps)
    ctrl_q_clim = make_clim(ctrl_q)
    ctrl_q_clim = ctrl_q_clim.where(ctrl_q_clim.plev<ctrl_ps_clim)
    ctrl_ta_clim = make_clim(ctrl_ta)
    ctrl_ta_clim = ctrl_ta_clim.where(ctrl_ta_clim.plev<ctrl_ps_clim)

    # if q has units of unity or kg/kg, we will have to 
    # multiply by 1000 later on to make it g/kg
    if(ctrl_q.units in ['1','kg/kg']):
        conv_factor = 1000
    elif(ctrl_q.units in ['g/kg']):
        conv_factor = 1
    else:
        warnings.warn("Cannot determine units of q. Assuming kg/kg.")
        conv_factor = 1000

    if(method=='pendergrass'):
        diff_q = (pert_q - tile_data(ctrl_q_clim,pert_q))/tile_data(ctrl_q_clim,pert_q)
    elif(method=='linear'):
        diff_q = pert_q - tile_data(ctrl_q_clim,pert_q)
    elif(method in ['kramer','zelinka']):
        diff_q = np.log(pert_q) - np.log(tile_data(ctrl_q_clim,pert_q))
    else:
        raise ValueError(
            "Please select a valid choice for the method argument.")
    
    # read in and regrid water vapor kernel
    kernel = check_plev(get_kern(kern))
    regridder = xe.Regridder(kernel[qlw_key],diff_q,method='bilinear',
                             extrap_method="nearest_s2d",
                             reuse_weights=False,periodic=True)

    qlw_kernel = tile_data(regridder(kernel[qlw_key],skipna=True),diff_q)
    qsw_kernel = tile_data(regridder(kernel[qsw_key],skipna=True),diff_q)  

    # regrid diff_q, ctrl_q_clim, and ctrl_ta_clim to kernel pressure levels
    kwargs = {'fill_value':'extrapolate'}
    diff_q = diff_q.interp_like(qlw_kernel,kwargs=kwargs)
    ctrl_q_clim = ctrl_q_clim.interp(plev=qlw_kernel.plev,kwargs=kwargs)
    ctrl_q_clim.plev.attrs['units'] = ctrl_q.plev.units
    ctrl_ta_clim = ctrl_ta_clim.interp(plev=qlw_kernel.plev,kwargs=kwargs)
    ctrl_ta_clim.plev.attrs['units'] = ctrl_ta.plev.units

    norm = tile_data(calc_q_norm(ctrl_ta_clim,ctrl_q_clim,method=method),diff_q)
    
    # use get_dp in climkern.util to calculate layer thickness
    dp = get_dp(diff_q,pert_ps,pert_trop,layer='troposphere')
                    
    # calculate feedbacks
    qlw_feedback = (qlw_kernel/norm * diff_q * conv_factor * dp/10000).sum(
        dim='plev',min_count=1)
    qsw_feedback = (qsw_kernel/norm * diff_q * conv_factor * dp/10000).sum(
        dim='plev',min_count=1).fillna(0)

    # one complication: CloudSat needs to be masked so we don't fill the NaNs
    # with zeros
    if(kern=='CloudSat'):
        qsw_feedback = qsw_feedback.where(qsw_kernel.sum(
            dim='plev',min_count=1).notnull())
    
    return(qlw_feedback,qsw_feedback)

def calc_dCRE_SW(ctrl_FSNT,pert_FSNT,ctrl_FSNTC,pert_FSNTC):
    """
    Calculate the change in the shortwave cloud radiative effect at the 
    top-of-atmosphere..

    Parameters
    ----------
    ctrl_FSNT : xarray DataArray
        Three-dimensional DataArray containing the all-sky net shortwave flux
        at the top-of-atmosphere in the control simulation
        with coords of time, lat, and lon and units of Wm^-2. It should be
        oriented such that positive = downwards.
        
    pert_FSNT : xarray DataArray
        Three-dimensional DataArray containing the all-sky net shortwave flux
        at the top-of-atmosphere in the perturbed simulation
        with coords of time, lat, and lon and units of Wm^-2. It should be
        oriented such that positive = downwards.

    ctrl_FSNTC : xarray DataArray
        Three-dimensional DataArray containing the clear-sky net shortwave flux
        at the top-of-atmosphere in the control simulation
        with coords of time, lat, and lon and units of Wm^-2. It should be
        oriented such that positive = downwards.
        
    pert_FSNTC : xarray DataArray
        Three-dimensional DataArray containing the clear-sky net shortwave flux
        at the top-of-atmosphere in the perturbed simulation
        with coords of time, lat, and lon and units of Wm^-2. It should be
        oriented such that positive = downwards.
    

    Returns
    -------
    dCRE_SW : xarray DataArray
        Three-dimensional DataArray containing the change in shortwave cloud
        radiative effect at the top-of-atmosphere with coords of time, lat, 
        and lon and units of Wm^-2. positive = downwards.
    """
    # double check the signs of LW/SW fluxes
    sw_coeff = -1 if ctrl_FSNT.mean() < 0 else 1

    ctrl_CRE_SW = sw_coeff * (ctrl_FSNT - ctrl_FSNTC)
    pert_CRE_SW = sw_coeff * (pert_FSNT - pert_FSNTC)

    return(pert_CRE_SW - ctrl_CRE_SW)

def calc_dCRE_LW(ctrl_FLNT,pert_FLNT,ctrl_FLNTC,pert_FLNTC):
    """
    Calculate the change in the shortwave cloud radiative effect at the 
    top-of-atmosphere.

    Parameters
    ----------
    ctrl_FLNT : xarray DataArray
        Three-dimensional DataArray containing the all-sky net longwave flux
        at the top-of-atmosphere in the control simulation
        with coords of time, lat, and lon and units of Wm^-2. It should be
        oriented such that positive = downwards.
        
    pert_FLNT : xarray DataArray
        Three-dimensional DataArray containing the all-sky net longwave flux
        at the top-of-atmosphere in the perturbed simulation
        with coords of time, lat, and lon and units of Wm^-2. It should be
        oriented such that positive = downwards.

    ctrl_FLNTC : xarray DataArray
        Three-dimensional DataArray containing the clear-sky net longwave flux
        at the top-of-atmosphere in the control simulation
        with coords of time, lat, and lon and units of Wm^-2. It should be
        oriented such that positive = downwards.
        
    pert_FLNTC : xarray DataArray
        Three-dimensional DataArray containing the clear-sky net longwave flux
        at the top-of-atmosphere in the perturbed simulation
        with coords of time, lat, and lon and units of Wm^-2. It should be
        oriented such that positive = downwards.
    

    Returns
    -------
    dCRE_LW : xarray DataArray
        Three-dimensional DataArray containing the change in longwave cloud
        radiative effect at the top-of-atmosphere with coords of time, lat, 
        and lon and units of Wm^-2. positive = downwards.
    """
    # double check the signs of LW/SW fluxes
    lw_coeff = -1 if ctrl_FLNT.mean() > 0 else 1

    ctrl_CRE_LW = lw_coeff * (ctrl_FLNT - ctrl_FLNTC)
    pert_CRE_LW = lw_coeff * (pert_FLNT - pert_FLNTC)

    return(pert_CRE_LW - ctrl_CRE_LW)

def calc_cloud_LW(t_as,t_cs,q_lwas,q_lwcs,dCRE_lw,IRF_lwas,IRF_lwcs):
    """
    Calculate the radiative perturbation from the longwave cloud feedback
    using the adjustment method outlined in Soden et al. (2008).

    Parameters
    ----------
    t_as : xarray DataArray
        The vertically integrated all-sky radiative perturbation at the TOA
        from the total temperature feedback. The total temperature feedback
        is the sum of the Planck and lapse rate feedbacks. Should have dims
        of lat, lon, and time.

    t_cs : xarray DataArray
        The vertically integrated clear-sky radiative perturbation at the TOA
        from the total temperature feedback. The total temperature feedback
        is the sum of the Planck and lapse rate feedbacks. Should have dims
        of lat, lon, and time.

    q_lwas : xarray DataArray
        The vertically integrated all-sky radiative perturbation at the TOA
        from the longwave water vapor feedback. Should have dims of lat, lon,
        and time.

    q_lwcs : xarray DataArray
        The vertically integrated clear-sky readiative perturbation at the TOA
        from the longwave water vapor feedback. SHould have dims of lat, lon,
        and time.

    dCRE_lw : xarray DataArray
        Three-dimensional DataArray containing the change in longwave cloud
        radiative effect at the top-of-atmosphere with coords of time, lat, 
        and lon and units of Wm^-2. positive = downwards.

    IRF_lwas : xarray DataArray
        The longwave all-sky instantaneous radiative forcing in units of Wm^-2
        with coords of lat, lon, and time.

    IRF_lwcs : xarray DataArray
        The longwave clear-sky instantaneous radiative forcing in units of Wm^-2
        with coords of lat, lon, and time.

    Returns
    -------
    lw_cld_feedback : xarray DataArray
        Three-dimensional DataArray containing the TOA radiative perturbation
        from the longwave cloud feedback.
    """
    # For now, we will assume all are on the same grid.
    # water vapor cloud masking term
    dq_lw = q_lwcs - q_lwas

    # temperature cloud masking term
    dt = t_cs - t_as

    # IRF cloud masking term
    # first double check that the LW IRF is positive
    irf_coeff = -1 if IRF_lwcs.mean() < 0 else 1
    dIRF_lw = irf_coeff * (IRF_lwcs - IRF_lwas)

    # calculate longwave cloud feedback
    lw_cld_feedback = dCRE_lw + dt + dq_lw + dIRF_lw

    return(lw_cld_feedback)

def calc_cloud_SW(alb_as,alb_cs,q_swas,q_swcs,dCRE_sw,IRF_swas,IRF_swcs):
    """
    Calculate the radiative perturbation from the shortwave cloud feedback
    using the adjustment method outlined in Soden et al. (2008).

    Parameters
    ----------
    alb_as : xarray DataArray
        The all-sky radiative perturbation at the TOA
        from the surface albedo feedback. Should have dims
        of lat, lon, and time.

    alb_cs : xarray DataArray
        The clear-sky radiative perturbation at the TOA
        from the surface albedo feedback. Should have dims
        of lat, lon, and time.

    q_swas : xarray DataArray
        The vertically integrated all-sky radiative perturbation at the TOA
        from the shortwave water vapor feedback. Should have dims of lat, lon,
        and time.

    q_swcs : xarray DataArray
        The vertically integrated clear-sky readiative perturbation at the TOA
        from the shortwave water vapor feedback. SHould have dims of lat, lon,
        and time.

    dCRE_sw : xarray DataArray
        Three-dimensional DataArray containing the change in shortwave cloud
        radiative effect at the top-of-atmosphere with coords of time, lat, 
        and lon and units of Wm^-2. positive = downwards.

    IRF_swas : xarray DataArray
        The shortwave all-sky instantaneous radiative forcing in units of Wm^-2
        with coords of lat, lon, and time.

    IRF_swcs : xarray DataArray
        The shortwave clear-sky instantaneous radiative forcing in units of Wm^-2
        with coords of lat, lon, and time.

    Returns
    -------
    sw_cld_feedback : xarray DataArray
        Three-dimensional DataArray containing the TOA radiative perturbation
        from the shortwave cloud feedback.
    """
    # For now, we will assume all are on the same grid.
    # water vapor cloud masking term
    dq_sw = q_swcs - q_swas

    # surface albedo cloud masking term
    dalb = alb_cs - alb_as

    # IRF cloud masking term
    dIRF_sw = (IRF_swcs - IRF_swas)

    # calculate longwave cloud feedback
    sw_cld_feedback = dCRE_sw + dalb + dq_sw + dIRF_sw

    return(sw_cld_feedback)

def calc_cloud_LW_res(ctrl_FLNT,pert_FLNT,IRF_lw,t_lw,q_lw):
    """
    Calculate the radiative perturbation from the shortwave cloud feedback
    using the residual method outlined in Soden & Held (2006).

    Parameters
    ----------
    ctrl_FLNT : xarray DataArray
        The all-sky net longwave radiative flux at the TOA in the control
        simulation. It should have coordinates of lat, lon, and time and
        be in units of Wm^-2.

    pert_FLNT : xarray DataArray
        The all-sky net longwave radiative flux at the TOA in the perturbed
        simulation. It should have coordinates of lat, lon, and time and
        be in units of Wm^-2.

    ERF_lw : xarray DataArray
        The longwave all-sky effective radiative forcing in units of Wm^-2
        with coords of lat, lon, and time.

    t_lw : xarray DataArray
        The vertically integrated all-sky readiative perturbation at the TOA
        from the longwave total temperature feedback. Should have dims of lat,
        lon, and time.

    q_lw : xarray DataArray
        The vertically integrated all-sky readiative perturbation at the TOA
        from the longwave water vapor feedback. Should have dims of lat,
        lon, and time.
        
    Returns
    -------
    lw_cld_feedback : xarray DataArray
        Three-dimensional DataArray containing the TOA radiative perturbation
        from the longwave cloud feedback.
    """
    # Calculate ΔR as the difference in net longwave flux
    # double check that sign is correct first, though
    lw_coeff = 1 if ctrl_FLNT.mean() < 0 else -1
    dR_lw = lw_coeff * (pert_FLNT - ctrl_FLNT)

    irf_coeff = -1 if IRF_lw.mean() < 0 else 1
    lw_cld_feedback = dR_lw - (irf_coeff * IRF_lw) - t_lw - q_lw
    return(lw_cld_feedback)

def calc_cloud_SW_res(ctrl_FSNT,pert_FSNT,ERF_sw,q_sw,alb_sw):
    """
    Calculate the radiative perturbation from the shortwave cloud feedback
    using the residual method outlined in Soden & Held (2006).

    Parameters
    ----------
    ctrl_FSNT : xarray DataArray
        The all-sky net shortwave radiative flux at the TOA in the control
        simulation. It should have coordinates of lat, lon, and time and
        be in units of Wm^-2.

    pert_FSNT : xarray DataArray
        The all-sky net shortwave radiative flux at the TOA in the perturbed
        simulation. It should have coordinates of lat, lon, and time and
        be in units of Wm^-2.

    ERF_sw : xarray DataArray
        The shortwave all-sky instantaneous radiative forcing in units of Wm^-2
        with coords of lat, lon, and time.

    q_sw : xarray DataArray
        The vertically integrated all-sky readiative perturbation at the TOA
        from the longwave water vapor feedback. Should have dims of lat,
        lon, and time.

    alb_sw : xarray DataArray
        The all-sky radiative perturbation at the TOA from the surface albedo
        feedback with coords of lat, lon, and time.
        
    Returns
    -------
    sw_cld_feedback : xarray DataArray
        Three-dimensional DataArray containing the TOA radiative perturbation
        from the shortwave cloud feedback.
    """
    # Calculate ΔR as the difference in net shortwave flux
    dR_sw = pert_FSNT - ctrl_FSNT

    sw_cld_feedback = dR_sw - ERF_sw - q_sw - alb_sw
    return(sw_cld_feedback)

def calc_strato_T(ctrl_ta,pert_ta,pert_ps,pert_trop=None,kern='GFDL',
                  sky='all-sky'):
    """
    Calculate the LW radiative perturbations (W/m^2) from changes
    in stratospheric temperature at the TOA or surface with the specified 
    radiative kernel. Horizontal resolution is kept at input data's 
    resolution.
    
    Parameters
    ----------
    ctrl_ta : xarray DataArray
        Contains air temperature on standard pressure levels in the control simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of K.
        
    pert_ta : xarray DataArray
        Contains air temperature on standard pressure levels in the perturbed simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of K.
        
    pert_ps : xarray DataArray
        Contains the surface pressure in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon.
        
    pert_trop : xarray DataArray
        Contains the tropopause height in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon.

    kern : string
        String specifying the institution name of the desired kernels.
        Defaults to GFDL.
        
    sky : string
        String specifying whether the all-sky or clear-sky feedbacks 
        should be used.

    Returns
    -------
    T_feedback : xarray DataArray
        3D DataArray containing the vertically integrated radiative perturbations
        caused by changes in stratospheric temperature. Has coordinates
        of time, lat, and lon.
    """

    t_key = 'lw_t' if check_sky(sky)=='all-sky' else 'lwclr_t'

    # check model output coordinates
    for d in [ctrl_ta,pert_ta]:
        d = check_coords(d,ndim=4)
    pert_ps = check_coords(pert_ps,ndim=3)

    # unit check
    ctrl_ta = check_var_units(check_plev_units(ctrl_ta),'T')
    pert_ta = check_var_units(check_plev_units(pert_ta),'T')
    pert_ps = check_pres_units(pert_ps,"pert PS")

    # make a fake tropopause if it's not provided
    if(type(pert_trop) == type(None)):
        pert_trop = make_tropo(pert_ps)
    else:
        pert_trop = check_coords(pert_trop)
        pert_trop = check_pres_units(pert_trop,"pert tropopause")

    # make climatology
    ctrl_ta_clim = make_clim(ctrl_ta)
    
    # calculate change in Ta
    diff_ta = (pert_ta - tile_data(ctrl_ta_clim,pert_ta))
    
    # read in and regrid temperature kernel
    kernel = check_plev(get_kern(kern))
    regridder = xe.Regridder(kernel[t_key],diff_ta,method='bilinear',
                             reuse_weights=False,periodic=True,
                            extrap_method='nearest_s2d')
    ta_kernel = tile_data(regridder(kernel[t_key]),diff_ta)

    # regrid diff_ta to kernel pressure levels
    # we have to extrapolate in case the lowest model plev is above the
    # kernel's. We will also mask values below the surface.
    diff_ta = diff_ta.interp_like(ta_kernel,kwargs={
        "fill_value": "extrapolate"})
    
    # use get_function in climkern.util to calculate layer thickness
    dp = get_dp(diff_ta,pert_ps,pert_trop,layer='stratosphere')

    # calculate feedbacks
    # for lapse rate, use the deviation of the air temperature response
    # from vertically uniform warming
    T_feedback = ((ta_kernel * diff_ta.fillna(0)) * dp/10000
                  ).sum(dim='plev',min_count=1)
       
    return(T_feedback)

def calc_strato_q(ctrl_q,ctrl_ta,pert_q,pert_ps,pert_trop=None,
                     kern='GFDL',sky='all-sky',method='pendergrass'):
    """
    Calculate the LW and SW radiative perturbations (W/m^2) using model output
    specific humidity and the chosen radiative kernel. Horizontal resolution
    is kept at input data's resolution.
    
    Parameters
    ----------
    ctrl_q : xarray DataArray
        Contains specific humidity on pressure levels in the control simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "1", "kg/kg", or "g/kg".

    ctrl_ta : xarray DataArray
        Contains air temperature on pressure levels in the control simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "K" or "C".
        
    pert_q : xarray DataArray
        Contains specific on pressure levels in the perturbed simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "1", "kg/kg", or "g/kg".
        
    pert_ps : xarray DataArray
        Contains the surface pressure in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon.
        
    pert_trop : xarray DataArray
        Contains the tropopause height in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon.

    kern : string
        String specifying the institution name of the desired kernels. Defaults to GFDL.

    sky : string
        String specifying whether the all-sky or clear-sky kernels should be used.

    logq : boolean
        Specifies whether to use the natural log of the specific humidity to calculate the
        water vapor feedbacks. Defaults to False.

    method : string
        Specifies the method to use to calculate the specific humidity
        feedback. Options are "pendergrass" (default), "kramer", "zelinka",
        and "linear". 

    Returns
    -------
    lw_q_feedback : xarray DataArray
        3D DataArray containing the vertically integrated radiative perturbations
        from changes in specific humidity in the stratosphere (longwave). 
        Has coordinates of time, lat, and lon.
        
    sw_q_feedback : xarray DataArray
        3D DataArray containing the vertically integrated radiative perturbations
        from changes in specific humidity in the stratosphere (shortwave).
        Has coordinates of time, lat, and lon.
    """
    qlw_key = 'lw_q' if check_sky(sky)=='all-sky' else 'lwclr_q'
    qsw_key = 'sw_q' if sky=='all-sky' else 'swclr_q'

    # check model output coordinates
    for d in [ctrl_q,ctrl_ta,pert_q]:
        d = check_coords(d,ndim=4)
    for d in [pert_ps]:
        d = check_coords(d)

    # unit check
    ctrl_ta = check_var_units(check_plev_units(ctrl_ta),'T')
    ctrl_q = check_var_units(check_plev_units(ctrl_q),'q')
    pert_q = check_var_units(check_plev_units(pert_q),'q')

    # check units of ps and trop
    pert_ps = check_pres_units(pert_ps,"pert PS")

    # make a fake tropopause if it's not provided
    if(type(pert_trop) == type(None)):
        pert_trop = make_tropo(ctrl_ps)
    else:
        pert_trop = check_coords(pert_trop)
        pert_trop = check_pres_units(pert_trop,"pert tropopause")
    
    # make climatology
    ctrl_q_clim = make_clim(ctrl_q)
    ctrl_ta_clim = make_clim(ctrl_ta)

    # if q has units of unity or kg/kg, we will have to 
    # multiply by 1000 later on to make it g/kg
    if(ctrl_q.units in ['1','kg/kg']):
        conv_factor = 1000
    elif(ctrl_q.units in ['g/kg']):
        conv_factor = 1
    else:
        warnings.warn("Cannot determine units of q. Assuming kg/kg.")
        conv_factor = 1000

    if(method=='pendergrass'):
        diff_q = (pert_q - tile_data(ctrl_q_clim,pert_q))/tile_data(ctrl_q_clim,pert_q)
    elif(method=='linear'):
        diff_q = pert_q - tile_data(ctrl_q_clim,pert_q)
    elif(method in ['kramer','zelinka']):
        diff_q = np.log(pert_q) - np.log(tile_data(ctrl_q_clim,pert_q))
    else:
        raise ValueError(
            "Please select a valid choice for the method argument.")
    
    # read in and regrid water vapor kernel
    kernel = check_plev(get_kern(kern))
    regridder = xe.Regridder(kernel[qlw_key],diff_q,method='bilinear',
                             extrap_method="nearest_s2d",
                             reuse_weights=False,periodic=True)

    qlw_kernel = tile_data(regridder(kernel[qlw_key],skipna=True),diff_q)
    qsw_kernel = tile_data(regridder(kernel[qsw_key],skipna=True),diff_q)  

    # regrid diff_q, ctrl_q_clim, and ctrl_ta_clim to kernel pressure levels
    kwargs = {'fill_value':'extrapolate'}
    diff_q = diff_q.interp_like(qlw_kernel,kwargs=kwargs)
    ctrl_q_clim = ctrl_q_clim.interp(plev=qlw_kernel.plev,kwargs=kwargs)
    ctrl_q_clim.plev.attrs['units'] = ctrl_q.plev.units
    ctrl_ta_clim = ctrl_ta_clim.interp(plev=qlw_kernel.plev,kwargs=kwargs)
    ctrl_ta_clim.plev.attrs['units'] = ctrl_ta.plev.units

    norm = tile_data(calc_q_norm(ctrl_ta_clim,ctrl_q_clim,method=method),diff_q)
    
    # use get_function in climkern.util to calculate layer thickness
    dp = get_dp(diff_q,pert_ps,pert_trop,layer='stratosphere')
                    
    # calculate feedbacks
    qlw_feedback = (qlw_kernel/norm * diff_q * conv_factor * dp/10000).sum(
        dim='plev',min_count=1)
    qsw_feedback = (qsw_kernel/norm * diff_q * conv_factor * dp/10000).sum(
        dim='plev',min_count=1).fillna(0)

    # one complication: CloudSat needs to be masked so we don't fill the NaNs
    # with zeros
    if(kern=='CloudSat'):
        qsw_feedback = qsw_feedback.where(qsw_kernel.sum(
            dim='plev',min_count=1).notnull())
    
    return(qlw_feedback,qsw_feedback)

def calc_RH_feedback(ctrl_q,ctrl_ta,ctrl_ps,pert_q,pert_ta,pert_ps,
                     pert_trop=None,kern='GFDL',sky='all-sky',
                     method='pendergrass'):
    """
    Calculate the TOA radiative perturbations from changes in relative
    humidity following Held & Shell (2012). Horizontal resolution is 
    kept at input data's resolution.
    
    Parameters
    ----------
    ctrl_q : xarray DataArray
        Contains specific humidity on pressure levels in the control simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "1", "kg/kg", or "g/kg".

    ctrl_ta : xarray DataArray
        Contains air temperature on pressure levels in the control simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "K" or "C".
        
    ctrl_ps : xarray DataArray
        Contains the surface pressure in the control simulation. Must
        be 3D with coords of time, lat, and lon.
        
    pert_q : xarray DataArray
        Contains specific on pressure levels in the perturbed simulation.
        Must be 4D with coords of time, lat, lon, and plev with units of either
        "1", "kg/kg", or "g/kg".

    pert_ta : xarray DataArray
        Contains air temperature on pressure levels in the perturbed 
        simulation. Must be 4D with coords of time, lat, lon, and plev
        with units of either "K" or "C".
        
    pert_ps : xarray DataArray
        Contains the surface pressure in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon.
        
    pert_trop : xarray DataArray
        Contains the tropopause height in the perturbed simulation. Must
        be 3D with coords of time, lat, and lon. If none is provided, function
        will assume a makeshift tropopause.

    kern : string
        String specifying the institution name of the desired kernels. Defaults to GFDL.

    sky : string
        String specifying whether the all-sky or clear-sky kernels should be used.

    method : string
        Specifies the method to use to calculate the specific humidity
        feedback. Options are "pendergrass" (default), "kramer", "zelinka",
        and "linear". 

    Returns
    -------
    RH_feedback : xarray DataArray
        3D DataArray containing the vertically integrated radiative 
        perturbations from changes in relative humidity (LW+SW).
        Has coordinates of time, lat, and lon.
    """
    qlw_key = 'lw_q' if check_sky(sky)=='all-sky' else 'lwclr_q'
    qsw_key = 'sw_q' if sky=='all-sky' else 'swclr_q'

    # check model output coordinates
    for d in [ctrl_q,ctrl_ta,pert_q,pert_ta]:
        d = check_coords(d,ndim=4)
    for d in [ctrl_ps,pert_ps]:
        d = check_coords(d)

    # unit check
    ctrl_ta = check_var_units(check_plev_units(ctrl_ta),'T')
    pert_ta = check_var_units(check_plev_units(pert_ta),'T')
    ctrl_q = check_var_units(check_plev_units(ctrl_q),'q')
    pert_q = check_var_units(check_plev_units(pert_q),'q')

    # check units of ps and trop
    ctrl_ps = check_pres_units(ctrl_ps,"ctrl PS")
    pert_ps = check_pres_units(pert_ps,"pert PS")

    # make a fake tropopause if it's not provided
    if(type(pert_trop) == type(None)):
        pert_trop = make_tropo(ctrl_ps)
    else:
        pert_trop = check_coords(pert_trop)
        pert_trop = check_pres_units(pert_trop,"pert tropopause")
    
    # make climatology
    ctrl_ps_clim = make_clim(ctrl_ps)
    ctrl_q_clim = make_clim(ctrl_q)
    ctrl_q_clim = ctrl_q_clim.where(ctrl_q_clim.plev<ctrl_ps_clim)
    ctrl_ta_clim = make_clim(ctrl_ta)
    ctrl_ta_clim = ctrl_ta_clim.where(ctrl_ta_clim.plev<ctrl_ps_clim)

    # if q has units of unity or kg/kg, we will have to 
    # multiply by 1000 later on to make it g/kg
    if(ctrl_q.units in ['1','kg/kg']):
        conv_factor = 1000
    elif(ctrl_q.units in ['g/kg']):
        conv_factor = 1
    else:
        warnings.warn("Cannot determine units of q. Assuming kg/kg.")
        conv_factor = 1000

    # the formulation of the q response depends on the method used
    if(method=='pendergrass'):
        diff_q = (pert_q - tile_data(ctrl_q_clim,pert_q))/tile_data(
            ctrl_q_clim,pert_q)
    elif(method=='linear'):
        diff_q = pert_q - tile_data(ctrl_q_clim,pert_q)
    elif(method in ['kramer','zelinka']):
        diff_q = np.log(pert_q) - np.log(tile_data(ctrl_q_clim,pert_q))
    else:
        raise ValueError(
            "Please select a valid choice for the method argument.")

    # for the RH feedback, we also need the T response
    diff_ta = pert_ta - tile_data(ctrl_ta_clim,pert_ta)
    
    # read in and regrid water vapor kernel
    kernel = check_plev(get_kern(kern))
    regridder = xe.Regridder(kernel[qlw_key],diff_q,method='bilinear',
                             extrap_method="nearest_s2d",
                             reuse_weights=False,periodic=True)
    q_kernel = kernel[qlw_key] + kernel[qsw_key]
    q_kernel = tile_data(regridder(q_kernel,skipna=True),diff_q)

    # regrid diff_q, ctrl_q_clim, and ctrl_ta_clim to kernel pressure levels
    kwargs = {'fill_value':'extrapolate'}
    diff_q = diff_q.interp_like(q_kernel,kwargs=kwargs)
    diff_ta = diff_ta.interp_like(q_kernel,kwargs=kwargs)
    ctrl_q_clim = ctrl_q_clim.interp(plev=q_kernel.plev,kwargs=kwargs)
    ctrl_q_clim.plev.attrs['units'] = ctrl_q.plev.units
    ctrl_ta_clim = ctrl_ta_clim.interp(plev=q_kernel.plev,kwargs=kwargs)
    ctrl_ta_clim.plev.attrs['units'] = ctrl_ta.plev.units

    norm = tile_data(calc_q_norm(ctrl_ta_clim,ctrl_q_clim,method=method),diff_q)
    
    # use get_dp in climkern.util to calculate layer thickness
    dp = get_dp(diff_q,pert_ps,pert_trop,layer='troposphere')

    # calculate RH_feedback
    RH_feedback = ((q_kernel/norm * diff_q * conv_factor * dp/10000)
                   - (q_kernel * diff_ta * dp/10000)).sum(
        dim='plev',min_count=1).fillna(0)

    # one complication: CloudSat needs to be masked so we don't fill the NaNs
    # with zeros
    if(kern=='CloudSat'):
        RH_feedback = RH_feedback.where(q_kernel.sum(
            dim='plev',min_count=1).notnull())
    
    return(RH_feedback)
    