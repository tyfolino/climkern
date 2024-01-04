# import statements

import climkern as ck  # if this doesn't work, it's probably the wrong env
from pooch import Unzip, retrieve
import os


def download():
    # get path of climkern package
    path = ck.__file__.replace("/__init__.py", "")
    fname = "data.zip"  # name of file to save before unzipping

    f_in = retrieve(
        url="doi:10.5281/zenodo.10223376/data.zip",
        known_hash="md5:8718deb9ed358dde36f3a9c1fd8a46c4",
        fname=fname,
        path=path,
        processor=Unzip(extract_dir="data"),
        progressbar=True,
    )

    # delete the zip file after unzipping
    os.remove(path + "/data.zip")
