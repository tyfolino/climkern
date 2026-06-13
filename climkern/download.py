# import statements

import os
import shutil
import zipfile

import pooch

from .options import DATA_VERSION, ZENODO_CONCEPT_ID
from .util import _cache_root

# md5 of the latest Zenodo data.zip; bump alongside DATA_VERSION / registry.txt
# whenever a new data release is published.
ZENODO_ZIP_HASH = "md5:e11bf7530ea17c1404b1fa97a50bfdaa"


def download() -> None:
    """Download the full kernel + tutorial dataset into the ClimKern cache.

    This is the optional "grab everything" path: it pulls the ~5 GB archive
    from Zenodo and unpacks it into the same cache directory used for on-demand
    streaming/caching (``pooch.os_cache("climkern")`` by default, overridable
    via the ``CLIMKERN_DATA_DIR`` environment variable). Afterwards the data is
    available offline via ``set_options(data_source="local")``.
    """
    cache = _cache_root()
    cache.mkdir(parents=True, exist_ok=True)

    zip_path = pooch.retrieve(
        url=f"doi:10.5281/zenodo.{ZENODO_CONCEPT_ID}/data.zip",
        known_hash=ZENODO_ZIP_HASH,
        fname="data.zip",
        path=cache,
        progressbar=True,
    )

    # The archive wraps everything in a top-level "data/" folder. Strip it so
    # the bulk download lands at <cache>/kernels/... and <cache>/tutorial_data/...,
    # matching the per-file cache layout (and Jetstream2) exactly.
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            rel = member.filename
            if rel.startswith("data/"):
                rel = rel[len("data/") :]
            if not rel:
                continue
            dest = cache / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)

    # remove the zip now that its contents are extracted
    os.remove(zip_path)

    print(f"ClimKern data (v{DATA_VERSION}) downloaded to {cache}")
