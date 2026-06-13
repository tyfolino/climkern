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
        urllib.request.urlopen(JS2_HTTPS_BASE + "tutorial_data/IRF.nc", timeout=10)
        return True
    except Exception:
        return False


needs_net = pytest.mark.skipif(
    not _online(), reason="requires network access to Jetstream2"
)


def _make_nc(path: Path) -> None:
    """Write a tiny valid netCDF at ``path`` (for offline disk-read tests)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset({"sentinel": ("t", [1.0, 2.0, 3.0])}).to_netcdf(path)


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


# --- offline: local mode + legacy fallback (no network) -------------------------


def test_local_missing_raises(tmp_path: Path) -> None:
    # empty cache + version_check off -> FileNotFoundError, no network.
    with ck.set_options(
        data_source="local", cache_dir=str(tmp_path), version_check=False
    ):
        with pytest.raises(FileNotFoundError):
            ck.tutorial_data("ctrl")


def test_local_reads_cached(tmp_path: Path) -> None:
    # local mode's happy path: read a file already present in the cache dir.
    _make_nc(tmp_path / "tutorial_data" / "ctrl.nc")
    with ck.set_options(
        data_source="local", cache_dir=str(tmp_path), version_check=False
    ):
        ds = ck.tutorial_data("ctrl")
    assert isinstance(ds, xr.Dataset)
    assert "sentinel" in ds  # confirms it read *our* local file


def test_legacy_fallback_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # An existing pre-download in the old package data/ dir is honored even in
    # cache mode, so upgrading users don't "lose" kernels when the cache moved.
    pkg = tmp_path / "pkg"
    _make_nc(pkg / "data" / "tutorial_data" / "ctrl.nc")
    monkeypatch.setattr("climkern.util.files", lambda _pkg: pkg)
    with ck.set_options(
        data_source="cache", cache_dir=str(tmp_path / "empty"), version_check=False
    ):
        ds = ck.tutorial_data("ctrl")  # found in legacy dir, no network
    assert isinstance(ds, xr.Dataset)
    assert "sentinel" in ds


# --- network-gated: real Jetstream2 access (uses the ~10 MB IRF.nc) -------------


@needs_net
def test_stream_open_writes_nothing(tmp_path: Path) -> None:
    pytest.importorskip("h5netcdf")
    pytest.importorskip("fsspec")
    with ck.set_options(
        data_source="stream", cache_dir=str(tmp_path), version_check=False
    ):
        ds = ck.tutorial_data("IRF")
    assert isinstance(ds, xr.Dataset)
    assert len(ds.data_vars) > 0
    # the whole point of stream: nothing lands on disk
    assert not list(tmp_path.rglob("*.nc"))


@needs_net
def test_cache_fetch_and_reuse(tmp_path: Path) -> None:
    cached = tmp_path / "tutorial_data" / "IRF.nc"
    with ck.set_options(
        data_source="cache", cache_dir=str(tmp_path), version_check=False
    ):
        ds = ck.tutorial_data("IRF")
        assert isinstance(ds, xr.Dataset)
        assert cached.is_file()  # downloaded + hash-verified into the cache
        mtime = cached.stat().st_mtime_ns
        ck.tutorial_data("IRF")  # second call must reuse, not re-download
    assert cached.stat().st_mtime_ns == mtime
