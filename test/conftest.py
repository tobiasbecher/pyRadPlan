import pytest
import pymatreader

try:
    from importlib import resources  # Standard from Python 3.9+
except ImportError:
    import importlib_resources as resources  # Backport for older versions


@pytest.fixture
def tg119_raw():
    phantom_data_str = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")
    phantom_data = pymatreader.read_mat(phantom_data_str)
    return phantom_data["ct"], phantom_data["cst"]
