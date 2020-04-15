import io, os
from pathlib import Path

from setuptools import find_packages, setup

# Package meta_data
NAME = 'regression_model'
DESCRIPTION = 'Regression model for train house price prediction dataset'
URL = 'https://github.com/ChandanVerma/MachineLearning_CI_CD.git'
EMAIL = 'verma.chandan01@gmail.com'
AUTHOR = 'ChandanVerma'
REQUIRES_PYTHON = '>=3.6.0'

# Packages required for this module to be executed
def list_reqs(fname = 'requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()

here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding = 'utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

## Load the packages __version__.py module as dictionary
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'regression_model'
about = {}
with open(PACKAGE_DIR /'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version

setup(
    name = NAME,
    version = about['__version__'],
    description = DESCRIPTION,
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author = AUTHOR,
    author_email = EMAIL,
    python_requires = REQUIRES_PYTHON,
    url = URL,
    packages = find_packages(exclude = ('tests', )),
    package_data = {'regression_model': ['VERSION']},
    install_requires = list_reqs(),
    extras_require = {},
    include_package_data = True,
    license = 'BSD 3',
    classifiers = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
