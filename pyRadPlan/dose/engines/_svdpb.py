"""
Implementation of a pencil beam dose calculation engine for photon beams
based on the Singular-value
decomposition (SVD) method by Bortfeld.
"""
from typing import TypedDict, Literal, Any
import logging
import random

import numpy as np

from pyRadPlan.plan import PhotonPlan
from ._base_pencilbeam import PencilBeamEngineAbstract

logger = logging.getLogger(__name__)


class DijSamplingConfig(TypedDict):
    """Properties for Dij sampling configuration."""

    rel_dose_threshold: float
    lat_cut_off: float
    type: Literal["radius", "depth"]
    delta_rad_depth: float


class PhotonPencilBeamSVDEngine(PencilBeamEngineAbstract):
    """
    Implementation of a pencil beam dose calculation engine for photon beams
    based on the Singular-value decomposition (SVD) method by Bortfeld.

    Parameters
    ----------
    pln : PhotonPlan
        A photon plan object.

    Attributes
    ----------
    use_custom_primary_photon_fluence : bool
        Use custom primary photon fluence.
    kernel_cutoff : float
        Kernel cutoff.
    random_seed : int
        Random seed.
    int_conv_resolution : float
        Intensity convolution resolution.
    enable_dij_sampling : bool
        Enable Dij sampling.
    dij_sampling : DijSamplingConfig
        Dij sampling configuration.
    """

    short_name = "SVDPB"
    name = "SVD Pencil Beam"
    possible_radiation_modes = ["photons"]

    use_custom_primary_photon_fluence: bool
    kernel_cutoff: float
    random_seed: int
    int_conv_resolution: float = 0.5
    enable_dij_sampling: bool = True
    dij_sampling: DijSamplingConfig

    def __init__(self, pln: PhotonPlan):  # more parameters necessary for both classes. !

        self.use_custom_primary_photon_fluence = False
        self.kernel_cutoff = 10.0
        self.random_seed = 0
        self.int_conv_resolution = 0.5
        self.enable_dij_sampling = True
        self.dij_sampling = DijSamplingConfig(
            rel_dose_threshold=0.01, lat_cut_off=20, type="radius", delta_rad_depth=5
        )

        super().__init__(PhotonPlan)

        # Protected/Private attributes (equivalent to SetAccess = protected)
        self._is_field_based_dose_calc = None  # will be set
        self._field_width = None  # will be obtained during calculation

        # Kernel Grid for convolution
        self._kernel_conv_size = None  # size of the convolution kernel
        self._kernel_x = None  # meshgrid in X
        self._kernel_z = None  # meshgrid in Z
        self._kernel_mxs = None  # list of kernel matrices

        self._gauss_filter = None  # 2D gaussian filter to model penumbra
        self._gauss_conv_size = None  # size of the gaussian convolution kernel

        self._conv_mx_x = None  # convolution meshgrid in X
        self._conv_mx_z = None  # convolution meshgrid in Z

        self._f_x = None  # fluence meshgrid in X
        self._f_z = None  # fluence meshgrid in Z

        self._f_pre = None  # precomputed fluence if uniform fluence is used for calculation
        self._interp_kernel_cache = (
            None  # cached kernel interpolators (if precomputation per beam is possible)
        )

        self._collimation = None  # collimation structure from DICOM import

    def _init_dose_calc(self, ct, cst, stf) -> dict[str, Any]:

        # TODO: What is "s" here?
        # self._is_field_based_dose_calc = any(str(s['bixelWidth']) == 'field' for s in stf)

        dij = super()._init_dose_calc(ct, cst, stf)

        # dij = []

        # checks of values
        # matrad here tests the kernel cutoff against the tabulated kernels
        # pyRadPlan, however, is more flexible and allows energy-specific kernels,
        # so we can't check this here but only when we load the energy-kernel?
        if self.kernel_cutoff < self.geometric_lateral_cutoff:
            logger.warning(
                "Kernel cutoff smaller than the geometric lateral cutoff. Using geometric cutoff."
            )

        if self._is_field_based_dose_calc:
            logger.debug("Enabling field-based dose calculation!")
            self.int_conv_resolution = self._collimation["conv_resolution"]
            self._field_width = self._collimation["field_width"]
        else:
            logger.debug("Enabling bixel-based dose calculation!")

            bws = np.unique(np.asarray([b.bixel_width for b in stf.beams], dtype=float))

            if len(bws) != 1:
                raise ValueError(
                    "Bixel widths inconsistent across beams, which is currently not supported!"
                )

            self._field_width = bws[0]

        field_limit = np.ceil(self._field_width / (2 * self.int_conv_resolution))
        # TODO: should this be +1 as end or is it correct? In matRad it is equivalent to no +1
        field_grid = self.int_conv_resolution * np.arange(-field_limit, field_limit)
        self._f_x, self._f_z = np.meshgrid(field_grid, field_grid, indexing="xy")

        # matRad does the gaussian filter here, but can we do that here?
        # We should be as flexible as to allow beams with different energies / penumbras / kernels
        # moved the krnel filtering to beam initialization

        # Initialize random number generator
        random.seed(self.random_seed)

        return dij

    def _init_beam(self, beam_info, ct, cst, stf, i):
        """
        Method for initializing the beams for analytical pencil beam
        dose calculation.

        call
          self.initBeam(ct,stf,dij,i)

        input
          beam_info:                  beam information struct
          ct:                         matRad ct struct
          stf:                        matRad steering information struct
          dij:                        matRad dij struct
          i:                          index of beam

        output
          dij:                        updated dij struct
        """

        beam_info = super()._init_beam(beam_info, ct, cst, stf, i)

        # get index of central ray or closest to the central ray
        # center = np.argmin(
        # np.sum(np.array([ray["ray_pos_bev"] for ray in beam_info["beam"]["rays"]]) ** 2, axis=1)
        # )

        # get correct kernel for given SSD at central ray (nearest neighbor approximation)
        # curr_ssd_ix = np.argmin(
        #     np.abs([self.machine.data.kernel.SSD - beam_info["beam"]["rays"][center].SSD])
        # )

        # Display console message
        # logger.info(f"\tSSD = {self.machine.data.kernel[curr_ssd_ix].SSD} mm ...")

        raise NotImplementedError("This method is not implemented yet.")

        currBeam = []
        return currBeam

    def _compute_bixel(self, curr_ray, k):
        """
        PyRadPlan photon dose calculation for a single bixel.

        call
            bixel = self.computeBixel(currRay,k)
        """
        bixel = {}

        raise NotImplementedError("This method is not implemented yet.")

        # if "physicalDose" in self._tmp_matrix_containers:
        #     bixel["physicalDose"] = self.calc_single_bixel(
        #         curr_ray.SAD,
        #         self._machine.data.m,
        #         self._machine.data.betas,
        #         curr_ray.interp_kernels,
        #         curr_ray.rad_depths,
        #         curr_ray.geo_depths,
        #         curr_ray.iso_lat_dists[:, 0],
        #         curr_ray.iso_lat_dists[:, 1],
        #     )

        #     # Sample dose only for bixel-based dose calculation
        #     if self.enable_dij_sampling and not self._is_field_based_dose_calc:
        #         bixel["ix"], bixel["physicalDose"] = self._sample_dij(
        #             curr_ray.ix,
        #             bixel["physicalDose"],
        #             curr_ray.rad_depths,
        #             curr_ray.radial_dist_sq,
        #             curr_ray.bixel_width,
        #         )
        #     else:
        #         bixel["ix"] = curr_ray.ix
        # else:
        #     bixel["ix"] = []
        # return bixel

    def _get_kernel_interpolators(self, _Fx):
        """Get kernel interpolator for photon dose calculation."""

        raise NotImplementedError("This method is not implemented yet.")

        # n_kernels = len(self._kernel_mxs)
        # interp_kernels = [None] * n_kernels
        # TODO: MATH and STUFF

        # # for ik in range(n_kernels):
        # # 2D convolution of Fluence and Kernels in Fourier domain
        # conv_mx = np.real(
        #     np.fft.ifft2(
        #         np.fft.fft2(
        #             Fx, (self.kernel_conv_size, self.kernel_conv_size)
        #         ) * np.fft.fft2(
        #             self._kernel_mxs[ik], (self.kernel_conv_size, self.kernel_conv_size)
        #         )
        #     )
        # )

        # # Creates an interpolant for kernels from vectors position X and Z
        # if pyRadPlan_cfg.is_matlab:
        #     interp_kernels[ik] = RegularGridInterpolator(
        #         (self.conv_mx_X, self.conv_mx_Z),
        #         conv_mx, method='linear', bounds_error=False, fill_value=None
        #     )

        interpKernels = []
        return interpKernels

    def _sample_dij(self, ix, bixel_dose, rad_depth_v, rad_distances_sq, bixel_width):
        """Performs lateral sampling of the beam."""

        raise NotImplementedError("This method is not implemented yet.")

    def _init_ray(self, curr_beam, j):
        """Initializes the current ray."""

        ray = super()._init_ray(curr_beam, j)
        return ray

    @staticmethod
    def calc_single_bixel(
        sad, m, betas, interp_kernels, rad_depths, geo_dists, iso_lat_dists_x, iso_lat_dists_z
    ):
        """Performs a beamlet calculation."""
        raise NotImplementedError("This method is not implemented yet.")
