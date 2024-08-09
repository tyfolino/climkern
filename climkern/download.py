# import statements

import os

from pooch import Unzip, retrieve

import climkern as ck  # if this doesn't work, it's probably the wrong env


def download():
    # get path of climkern package
    path = ck.__file__.replace("/__init__.py", "")
    fname = "data.zip"  # name of file to save before unzipping

    retrieve(
        url="doi:10.5281/zenodo.10223376/data.zip",
        known_hash=None,
        fname=fname,
        path=path,
        processor=Unzip(extract_dir="data"),
        progressbar=True,
    )

    # delete the zip file after unzipping
    os.remove(path + "/data.zip")
