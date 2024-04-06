# ClimKern: a Python package for calculating radiative feedbacks

[![DOI](https://zenodo.org/badge/588323813.svg)](https://zenodo.org/doi/10.5281/zenodo.10291284)

## Citation
If you use this package or any part of this code, please cite it! Until we have a paper prepared, please cite this package at software.

Janoski, T. P., & Mitevski, I. (2024, April 6). ClimKern (Version 1.1.0). Retrieved from https://pypi.org/project/climkern/1.1.0/. <https://doi.org/10.5281/zenodo.10291284>.

## Overview

The radiative kernel technique outlined in [Soden & Held (2006)](https://journals.ametsoc.org/view/journals/clim/19/14/jcli3799.1.xml) and [Soden et al. (2008)](https://journals.ametsoc.org/view/journals/clim/21/14/2007jcli2110.1.xml) is commonly used to calculate climate feedbacks. The "kernels" refer to datasets containing the radiative sensitivities of TOA (or surface) radiation to changes in fields such as temperature, specific humidity, and surface albedo; they are typically computed using offline radiative transfer calculations.

ClimKern
* standardizes the assumptions used in producing radiative feedbacks using kernels
* simplifies the calculations by giving users access to functions tailored for climate model output
* provides access to a repository of **12 different radiative kernels** to quantify interkernel spread

The below information is meant to be a quickstart guide, but all functions and capabilities can be found at ClimKern's [documentation site](https://tyfolino.github.io/climkern/).

## Installation

ClimKern is built on the Xarray architecture and requires several other packages for
regridding and climate model output compatibility. The easiest method is to create a
new conda environment using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mamba](https://mamba-framework.readthedocs.io/en/latest/installation_guide.html):

`conda create -n ck_env python=3.10 esmpy -c conda-forge`

A conda environment is necessary because [ESMPy](https://earthsystemmodeling.org/esmpy/), which is required for regridding kernels, is unavailable via `pip`.

Next, activate the new environment:

`conda activate ck_env`

Finally, install ClimKern with [pip](https://pip.pypa.io/en/stable/#):

`pip install climkern`

Once installed, ClimKern requires kernels found on [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10223376). These kernels (and tutorial data) are stored separately because of PyPI size limitations. You can download the kernels easily using the download script included in the package:  

`python -m climkern download`

Note: The kernels & tutorial data are approximately 5.5 GB.

<i>Optional</i>:

You can test your installation via pytest.

```
pip install pytest
pytest -v --pyargs climkern
```

All three tests should pass.

## Basic tutorial
### Temperature, water vapor, and surface albedo feedbacks

This brief tutorial will cover the basics of using ClimKern. Please check the [documentation](https://tyfolino.github.io/climkern/) for a more complete list of available functions. We start by importing ClimKern and accessing our tutorial data:
```python
import climkern as ck

ctrl,pert = ck.tutorial_data('ctrl'),ck.tutorial_data('pert')
```

These datasets have all the necessary variables for computing feedbacks. Let's start with temperature feedbacks.
```python
LR,Planck = ck.calc_T_feedbacks(ctrl.T,ctrl.TS,ctrl.PS,
                                pert.T,pert.TS,pert.PS,pert.TROP_P,
                                kern="GFDL")
```
To produce succinct output, let's use ClimKern's spatial average function. Additionally, we will normalize the feedbacks by global average surface temperature change to convert from Wm<sup>-2</sup>, the output of ClimKern functions, to the more commonly used units of Wm<sup>-2</sup>K<sup>-1</sup>.
```python
# compute global average surface temperature change
dTS_glob_avg = ck.spat_avg(pert.TS - ctrl.TS)

# normalize temperature feedbacks by temperature change and take
# the annual average
print("The global average lapse rate feedback is {val:.2f} W/m^2/K.".format(
    val=(ck.spat_avg(LR)/dTS_glob_avg).mean()))
print("The global average Planck feedback is {val:.2f} W/m^2/K.".format(
    val=(ck.spat_avg(Planck)/dTS_glob_avg).mean()))
```
Expected result with the GFDL kernel:
> `The global average lapse rate feedback is -0.41 W/m^2/K.`
> 
> `The global average Planck feedback is -3.12 W/m^2/K.`

The water vapor and surface albedo feedbacks are calculated similarly:
```python
q_lw,q_sw = ck.calc_q_feedbacks(ctrl.Q,ctrl.T,ctrl.PS,
                                pert.Q,pert.PS,pert.TROP_P,
                                kern="GFDL",method="zelinka")
alb = ck.calc_alb_feedback(ctrl.FSUS,ctrl.FSDS,
                           pert.FSUS,pert.FSDS,
                           kern="GFDL")

print("The global average water vapor feedback is {val:.2f} W/m^2/K.".format(
    val=(ck.spat_avg(q_lw+q_sw)/dTS_glob_avg).mean()))
print("The global average surface albedo feedback is {val:.2f} W/m^2/K."
      .format(
    val=(ck.spat_avg(alb)/dTS_glob_avg).mean()))
```
Expected result:
>`The global average water vapor feedback is 1.44 W/m^2/K.`
>
>`The global average surface albedo feedback is 0.38 W/m^2/K.`

### Cloud feedbacks
The cloud feedbacks, calculated using [Soden et al. (2008)](https://journals.ametsoc.org/view/journals/clim/21/14/2007jcli2110.1.xml) adjustment method, require all-sky and clear-sky versions of other feedbacks and the instantaneous radiative forcing.

First, we need the longwave and shortwave cloud radiative effects, which ClimKern can calculate.
```python
dCRE_LW = ck.calc_dCRE_LW(ctrl.FLNT,pert.FLNT,ctrl.FLNTC,pert.FLNTC)
dCRE_SW = ck.calc_dCRE_SW(ctrl.FSNT,pert.FSNT,ctrl.FSNTC,pert.FSNTC)
```
Let's also read in the tutorial IRF.
```python
IRF = ck.tutorial_data('IRF')
# overwrite IRF latitude because of rounding error
IRF['lat'] = ctrl.lat
```
Next, we need the clear-sky versions of the temperature, water vapor, and surface albedo feedbacks.
```python
#_cs means clear-sky
LR_cs,Planck_cs = ck.calc_T_feedbacks(ctrl.T,ctrl.TS,ctrl.PS,
                                pert.T,pert.TS,pert.PS,pert.TROP_P,
                                kern="GFDL",sky="clear-sky")
q_lw_cs,q_sw_cs = ck.calc_q_feedbacks(ctrl.Q,ctrl.T,ctrl.PS,
                                pert.Q,pert.PS,pert.TROP_P,
                                kern="GFDL",method="zelinka",sky="clear-sky")
alb_cs = ck.calc_alb_feedback(ctrl.FSUS,ctrl.FSDS,
                           pert.FSUS,pert.FSDS,
                           kern="GFDL",sky="clear-sky")
```
At last, we can calculate the longwave and shortwave cloud feedbacks.
```python
cld_lw = ck.calc_cloud_LW(LR + Planck,LR_cs+Planck_cs,q_lw,q_lw_cs,dCRE_LW,
                          IRF.IRF_lwas,IRF.IRF_lwcs)
cld_sw = ck.calc_cloud_SW(alb,alb_cs,q_sw,q_sw_cs,dCRE_SW,IRF.IRF_swas,
                          IRF.IRF_swcs)

print("The global average SW cloud feedback is {val:.2f} W/m^2/K.".format(
    val=(ck.spat_avg(cld_sw)/dTS_glob_avg).mean()))
print("The global average LW cloud feedback is {val:.2f} W/m^2/K.".format(
    val=(ck.spat_avg(cld_lw)/dTS_glob_avg).mean()))
```
Expected result:
>`The global average SW cloud feedback is 0.48 W/m^2/K.`
>
>`The global average LW cloud feedback is 0.02 W/m^2/K.`

## Troubleshooting

If you are having issues downloading dependencies with `pip`, you can also try adding them to your conda environment with `conda`, i.e.:

`conda install xesmf -c conda-forge`

If you are having trouble downloading the  kernels and tutorial data using the package's download function, you can also download the data directly from the [Zenodo repository](https://zenodo.org/doi/10.5281/zenodo.10223376) and put it in the climkern/data directory located wherever your conda/mamba environments are stored.

## Other features & coming soon
ClimKern has several other useful features:
- Four different methods for calculating water vapor feedbacks.
- The ability to calculate the "relative humidity" version of all feedbacks following [Held & Shell (2012)](https://journals.ametsoc.org/view/journals/clim/25/8/jcli-d-11-00721.1.xml) and [Zelinka et al. (2020)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL085782).
- Functions to calculate stratospheric temperature and water vapor feedbacks.

We are continuously updating the package. Please check out the [GitHub issues page](https://github.com/tyfolino/climkern/issues) for this repository for plans for new features.

## Want to help? Get involved!

We deeply appreciate contributions from other scientists and programmers and are happy to attribute credit accordingly. If you wish to contribute, please create a fork or branch from the `dev` channel (<b>not</b> `main`) and submit a pull request when you are done with your changes.