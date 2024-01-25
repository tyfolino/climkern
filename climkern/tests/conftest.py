import pytest
import climkern as ck
import xarray as xr


@pytest.fixture
def ctrl() -> xr.Dataset:
    """
    Read in the control tutorial data.
    """
    return ck.tutorial_data('ctrl')
@pytest.fixture
def pert() -> xr.Dataset:
    """
    Read in the 2xCO2 tutorial data.
    """
    return ck.tutorial_data('pert')
@pytest.fixture
def dTS_glob_avg(ctrl : xr.Dataset, pert : xr.Dataset) -> xr.Dataset:
    return ck.spat_avg(pert.TS - ctrl.TS)