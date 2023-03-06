from setuptools import setup, find_packages

DISTNAME = 'climkern'
DESCRIPTION = 'Radiative Kernel Tool for Calculating Climate Feedbacks'
AUTHOR = 'Ty Janoski'
AUTHOR_EMAIL = 'janoski@ldeo.columbia.edu'
PYTHON_REQUIRES = '>=3.6'

INSTALL_REQUIRES = [
    'xarray>=2022.12.0',
    'cf-xarray>=0.7.2',
    'cftime',
    'xesmf>=0.3.0'
]

setup(
    name=DISTNAME,
    author=AUTHOR,
    version='0.0.12',
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    package_data={
        'climkern':['data/*.nc'],
    }
)
