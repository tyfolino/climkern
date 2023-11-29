# ClimKern: a Python package for calculating radiative feedbacks

## Overview

The radiative kernel technique outlined in [Soden & Held (2006)](https://journals.ametsoc.org/view/journals/clim/19/14/jcli3799.1.xml) and [Soden et al. (2008)](https://journals.ametsoc.org/view/journals/clim/21/14/2007jcli2110.1.xml) is commonly used to calculate climate feedbacks. The "kernels" refer to datasets containing the radiative sensitivities of TOA (or surface) radiation to changes in fields such as temperature, specific humidity, and surface albedo; they are typically computed using offline radiative transfer calculations.

ClimKern
* standardizes the assumptions used in producing radiative feedbacks using kernels
* simplifies the calculations by giving users access to functions tailored for climate model output
* provides access to a repository of **12 different radiative kernels** to quantify interkernel spread

## Installation

ClimKern is built on the architecture of Xarray and requires several other packages for compatibility with climate model output. The easiest method to install is to create a new conda environment with prerequisite packages using [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) or [mamba](https://mamba-framework.readthedocs.io/en/latest/installation_guide.html):
`conda create -n ckenv python=3.9 esmpy xarray xesmf cftime`
or
`mamba create -n ckenv python=3.9 esmpy xarray xesmf cftime`
Then, activate the environment:
`conda activate ckenv`
or
`mamba activate ckenv`
Finally, install ClimKern with pip:
`pip install -i https://test.pypi.org/simple/ climkern`