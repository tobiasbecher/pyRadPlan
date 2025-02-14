import pytest
import pymatreader
import sys

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+


@pytest.fixture
def tg119_raw():
    phantom_data_str = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")
    phantom_data = pymatreader.read_mat(phantom_data_str)
    return phantom_data["ct"], phantom_data["cst"]
