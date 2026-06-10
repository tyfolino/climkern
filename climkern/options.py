"""Contains the options where users configure where ClimKern
gets its data. ClimKern needs to access radiative kernels that, as of 2026,
are too big to fit on PyPI. To get around this, ClimKern can read the kernels
from an on-disk cache that is built on demand, direct streaming from Jetstream2,
or as a pre-downloaded local copy.
"""

import os
from typing import Any  # necessary because MyPy gets confused

# Version of the ClimKern kernels dataset. Whenever you release new kernels
# on Jetstream2/Zenodo, update this version string and the registry.txt hashes,
# so that users get a warning if they have an old version of the kernels cached.
DATA_VERSION = "1.2.1"

# Simply the URL prefix for streaming data from Jetstream2. The file paths
# come from registry.txt.
JS2_HTTPS_BASE = "https://js2.jetstream-cloud.org:8001/pythia/ClimKern/"

# This is the Zenodo concept ID for the ClimKern kernels dataset.
# We use this to check to see if there is a newer version on Zenodo.
# This will never change.
ZENODO_CONCEPT_ID = "10223376"

# Cache means that it will search for files in a local cache directory, and if not
# found, download them from Jetstream2 and save them to the cache.
# Stream means it will read directly from Jetstream without checking the cache
# Local means it will only read from the local cache and not download anything.
_VALID_SOURCES = {"cache", "stream", "local"}

# Users can set environment variables, userful for testing on cloud/CI environments,
# so that they don't have to change code to configure climkern.
# The default data source is "cache", and the default cache directory is None or
# whatever pooch.os_cache("climkern") returns.
OPTIONS: dict[str, Any] = {
    "data_source": os.environ.get("CLIMKERN_DATA_SOURCE", "cache"),
    "cache_dir": os.environ.get("CLIMKERN_DATA_DIR") or None,
    "version_check": True,
}


# Simple function for making sure the options are valid. Called by set_options.
def _validate(key: str, value: object) -> None:
    """Raise ValueError if ``key``/``value`` is not a recognized option."""
    if key not in OPTIONS:
        raise ValueError(
            f"{key!r} is not a valid option. Valid options: {sorted(OPTIONS)}."
        )
    if key == "data_source" and value not in _VALID_SOURCES:
        raise ValueError(
            f"data_source must be one of {sorted(_VALID_SOURCES)}, got {value!r}."
        )
    if key == "version_check" and not isinstance(value, bool):
        raise ValueError(f"version_check must be a bool, got {value!r}.")


class set_options:  # noqa: N801 (lowercase mirrors xarray.set_options, a public API)
    """Set ClimKern data-access options, globally or within a ``with`` block.

    Parameters
    ----------
    data_source : {"cache", "stream", "local"}, optional
        Where to read kernels and tutorial data.
        - ``"cache"`` (default): download each file from Jetstream2 on first
          use into a local cache, then reuse the cached copy. Files are
          verified against shipped hashes, so a corrected kernel in a new
          ClimKern release is re-downloaded automatically.
        - ``"stream"``: read directly from Jetstream2 without writing anything
          to disk.
        - ``"local"``: only use files already present on disk (from a prior
          cache fetch or ``download()``); never access the network.
    cache_dir : str or os.PathLike, optional
        Override the cache location. Defaults to ``pooch.os_cache("ClimKern")``
        (e.g. ``~/.cache/ClimKern`` on Linux), or the ``CLIMKERN_DATA_DIR``
        environment variable if set.
    version_check : bool, optional
        Whether to warn, once per session, when a newer kernel data release is
        available on Zenodo. Default ``True``.

    Examples
    --------
    >>> import climkern as ck
    >>> ck.set_options(data_source="stream")  # change globally
    >>> with ck.set_options(data_source="local"):  # change temporarily
    ...     ctrl = ck.tutorial_data("ctrl")
    """

    def __init__(self, **kwargs: object) -> None:
        self._prev: dict[str, Any] = {}
        for key, value in kwargs.items():
            _validate(key, value)
            self._prev[key] = OPTIONS[key]
            OPTIONS[key] = value

    def __enter__(self) -> "set_options":
        """Enter the context manager, keeping the chosen options applied."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Restore the options that were in effect before the ``with`` block."""
        OPTIONS.update(self._prev)
