from setuptools import setup, find_packages

DISTNAME = 'climkern'
DESCRIPTION = 'Python package for calculating radiative feedbacks using \
radiative kernels.'
AUTHOR = 'Ty Janoski'
AUTHOR_EMAIL = 'tjanoski@ccny.cuny.edu'
PYTHON_REQUIRES = '>=3.9'

INSTALL_REQUIRES = [
    'xarray>=0.16.2',
    'cf-xarray>=0.5.1',
    'cftime',
    'xesmf>=0.7.1',
    'esmpy',
    'importlib_resources',
    'pooch',
    'tqdm',
    'plac',
    'netCDF4'
]

setup(
    name=DISTNAME,
    author=AUTHOR,
    version='1.0.4',
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    }
)
