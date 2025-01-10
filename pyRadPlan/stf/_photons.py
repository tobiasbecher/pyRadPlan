import numpy as np
from pyRadPlan.ct import CT
from pyRadPlan.stf._externalbeam import StfGeneratorExternalBeamRayBixel
from pyRadPlan.machines import PhotonLINAC


class StfGeneratorPhotonIMRT(StfGeneratorExternalBeamRayBixel):
    """Class representing a Photon IMRT Geometry Stf Generator.

    Attributes
    ----------
    name : str
        The name of the generator ("Photon IMRT Geometry").
    short_name : str
        The short name of the generator ("photonIMRT").
    possible_radiation_modes : list[str]
        A list of possible radiation modes (["photons"]).
    """

    name = "Photon IMRT Geometry"
    short_name = "photonIMRT"
    possible_radiation_modes = ["photons"]

    def __init__(self, pln=None):
        self.radiation_mode = "photons"  # Radiation mode must be photons
        super().__init__(pln)

    def _initialize(self):
        super()._initialize()

        if not isinstance(self.machine, PhotonLINAC):
            raise ValueError("Machine must be an instance of PhotonLINAC.")

    def _generate_source_geometry(self):
        """Generates the source geometry for the photon IMRT geometry."""
        stf = super()._generate_source_geometry()
        return stf


class StfGeneratorPhotonCollimatedSquareFields(StfGeneratorExternalBeamRayBixel):
    """Class representing a Photon Collimated Square Fields Stf Generator.

    Attributes
    ----------
    name : str
        The name of the generator ("Photon Collimated Square Fields").
    short_name : str
        The short name of the generator ("photonCollimatedSquareFields").
    possible_radiation_modes : list[str]
        A list of possible radiation modes (["photons"]).
    """

    name = "Photon Open Fields Geometry"
    short_name = "photonOpenFields"
    possible_radiation_modes = ["photons"]

    # Alias for field_width
    @property
    def field_width(self) -> float:
        """Alias for the bixel_width property."""
        return self.bixel_width

    @field_width.setter
    def field_width(self, value: float):
        self.bixel_width = value

    def __init__(self, pln=None):
        self.radiation_mode = "photons"
        super().__init__(pln)

    def _generate_ray_positions_in_isocenter_plane(self, beam):
        """Generates the ray positions in the isocenter plane.
        As we have a square field, this is a single ray to the isocenter.

        Parameters
        ----------
        beam : dict
            The beam dictionary. Ignored in this case

        Returns
        -------
        np.ndarray
            A numpy array with the single ray position of (0, 0, 0) in the isocenter.
        """

        return np.zeros((3, 1), dtype=float)

    def _generate_source_geometry(self):
        """Generates the source geometry for the photon collimated square
        fields.
        """
        stf = super()._generate_source_geometry()

        for i, field in enumerate(stf):
            field["field_width"] = self.field_width

        return stf


if __name__ == "__main__":
    import SimpleITK as sitk

    sample_array = np.random.rand(50, 100, 100) * 1000  # Random HU values
    image = sitk.GetImageFromArray(sample_array)
    image.SetOrigin((0, 0, 0))
    image.SetSpacing((1, 1, 2))
    image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))

    ct = CT(cube_hu=image)
    cst = {"dummy": {"indices": [int(1e5)], "type": "TARGET"}}

    stf_gen = StfGeneratorPhotonIMRT()

    stf_gen.machine = "Generic"

    stf_gen.mult_scen = "nomScen"

    stf_gen.gantry_angles = [90.0, 270.0]
    stf_gen.couch_angles = [0.0, 0.0]

    stf_gen.bixel_width = 5.0

    stf = stf_gen.generate(ct, cst)

    print(stf)
