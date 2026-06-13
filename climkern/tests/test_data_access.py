"""Tests for the streaming/caching data-access layer (issue #47)."""

import urllib.request
from pathlib import Path

import pytest
import xarray as xr

import climkern as ck
from climkern.options import _VALID_SOURCES, JS2_HTTPS_BASE, OPTIONS
from climkern.util import _load_registry


def _online() -> bool:
    """Return True if Jetstream2 is reachable (used to gate network tests)."""
    try:
        urllib.request.urlopen(JS2_HTTPS_BASE + "tutorial_data/ctrl.nc", timeout=10)
        return True
    except Exception:
        return False


needs_net = pytest.mark.skipif(
    not _online(), reason="requires network access to Jetstream2"
)


# --- offline: options machinery -------------------------------------------------


def test_default_data_source() -> None:
    assert OPTIONS["data_source"] in _VALID_SOURCES


def test_set_options_invalid_key() -> None:
    with pytest.raises(ValueError):
        ck.set_options(not_a_real_option=1)


def test_set_options_invalid_source() -> None:
    with pytest.raises(ValueError):
        ck.set_options(data_source="bogus")


def test_set_options_context_manager_restores() -> None:
    before = OPTIONS["data_source"]
    with ck.set_options(data_source="stream"):
        assert OPTIONS["data_source"] == "stream"
    assert OPTIONS["data_source"] == before


def test_registry_well_formed() -> None:
    registry = _load_registry()
    assert registry, "registry.txt should ship with the package"
    for relpath, file_hash in registry.items():
        assert relpath.endswith(".nc")
        assert relpath.startswith(("kernels/", "tutorial_data/"))
        assert len(file_hash) == 64
        assert all(c in "0123456789abcdef" for c in file_hash)
    # ECHAM5 was removed from the dataset in v1.2.1; it must not be pinned.
    assert not any(r.startswith("kernels/ECHAM5/") for r in registry)


# --- network-gated: real data access -------------------------------------------


def test_local_missing_raises(tmp_path: Path) -> None:
    # purely offline: empty cache + version_check off -> FileNotFoundError
    with ck.set_options(
        data_source="local", cache_dir=str(tmp_path), version_check=False
    ):
        with pytest.raises(FileNotFoundError):
            ck.tutorial_data("ctrl")


@needs_net
def test_stream_open() -> None:
    pytest.importorskip("h5netcdf")
    pytest.importorskip("fsspec")
    with ck.set_options(data_source="stream", version_check=False):
        ds = ck.tutorial_data("ctrl")
    assert isinstance(ds, xr.Dataset)
    assert len(ds.data_vars) > 0


@needs_net
def test_cache_fetch_and_reuse(tmp_path: Path) -> None:
    with ck.set_options(
        data_source="cache", cache_dir=str(tmp_path), version_check=False
    ):
        ds = ck.tutorial_data("ctrl")
        assert isinstance(ds, xr.Dataset)
    # pooch verified the download against the shipped hash and cached it here
    assert (tmp_path / "tutorial_data" / "ctrl.nc").is_file()
