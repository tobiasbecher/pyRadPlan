from scipy import io
import pymatreader


def load(path2file):
    """Loads a .mat file and returns the data as a dictionary."""

    try:
        matRad_patient = pymatreader.read_mat(filename=path2file)
    except NotImplementedError:
        matRad_patient = io.loadmat(
            file_name=path2file, mat_dtype=True, squeeze_me=True, struct_as_record=True
        )
    except Exception as exep:
        raise ValueError(f"Could not load the .mat file: {path2file}") from exep

    return matRad_patient


def save(path2file, dict):
    """Saves a dictionary as a .mat file."""
    io.savemat(file_name=path2file, mdict=dict)
