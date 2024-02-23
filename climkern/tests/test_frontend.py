import xarray as xr

from climkern.frontend import *


def test_calc_T_feedbacks(
    ctrl: xr.Dataset, pert: xr.Dataset, dTS_glob_avg: xr.DataArray
) -> None:
    LR, Planck = calc_T_feedbacks(
        ctrl.T, ctrl.TS, ctrl.PS, pert.T, pert.TS, pert.PS, pert.TROP_P, kern="GFDL"
    )
    LR_val = (spat_avg(LR) / dTS_glob_avg).mean()
    Planck_val = (spat_avg(Planck) / dTS_glob_avg).mean()
    xr.testing.assert_allclose(LR_val, xr.DataArray(-0.41), atol=0.01)
    xr.testing.assert_allclose(Planck_val, xr.DataArray(-3.12), atol=0.01)


def test_albedo_feedbacks(
    ctrl: xr.Dataset, pert: xr.Dataset, dTS_glob_avg: xr.DataArray
) -> None:
    alb = calc_alb_feedback(ctrl.FSUS, ctrl.FSDS, pert.FSUS, pert.FSDS, kern="GFDL")
    val_to_test = (spat_avg(alb) / dTS_glob_avg).mean()
    xr.testing.assert_allclose(val_to_test, xr.DataArray(0.38), atol=0.01)

def test_calc_q_feedbacks(
        ctrl: xr.Dataset, pert: xr.Dataset, dTS_glob_avg: xr.DataArray
) -> None:
    lw,sw = calc_q_feedbacks(
        ctrl.Q,ctrl.T,ctrl.PS,pert.Q,pert.PS,pert.TROP_P,kern="GFDL",method="Zelinka"
    )
    val_to_test = (spat_avg(lw + sw) / dTS_glob_avg).mean()
    xr.testing.assert_allclose(val_to_test,xr.DataArray(1.44), atol=0.01)
