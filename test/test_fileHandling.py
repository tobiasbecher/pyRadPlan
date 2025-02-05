import os

try:
    from importlib import resources  # Standard from Python 3.9+
except ImportError:
    import importlib_resources as resources  # Backport for older versions

from pyRadPlan.io.matLabFileHandler import MatLabFileHandler
import pyRadPlan.io.matRad as matRadIO

# Paths


def test_file_handler(tmp_path):
    """Test loading method."""

    phantoms = resources.files("pyRadPlan.data.phantoms")
    tg119_path = phantoms.joinpath("TG119.mat")
    tg119_load = matRadIO.load(tg119_path)

    tg119 = {
        "ct": tg119_load["ct"],
        "cst": tg119_load["cst"],
    }

    file_handler = MatLabFileHandler(tmp_path)
    file_handler.save(**tg119)

    assert os.path.exists(tmp_path.joinpath("ct.mat"))
    assert os.path.exists(tmp_path.joinpath("cst.mat"))

    ct = file_handler.load("ct")
    assert isinstance(ct, dict)
    cst = file_handler.load("cst")
    assert isinstance(cst, dict)

    file_handler.delete("ct", "cst")
    assert not os.path.exists(tmp_path.joinpath("ct.mat"))
    assert not os.path.exists(tmp_path.joinpath("cst.mat"))
