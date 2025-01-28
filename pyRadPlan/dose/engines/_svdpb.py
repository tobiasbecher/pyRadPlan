"""
Implementation of a pencil beam dose calculation engine for photon beams
based on the Singular-value
decomposition (SVD) method by Bortfeld.
"""
from typing import TypedDict, Literal, Any, cast
import logging
import random

import numpy as np
import scipy.fft as fft

from pyRadPlan.plan import PhotonPlan

# from pyRadPlan.stf import Beam
from pyRadPlan.machines import PhotonLINAC
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

    def __init__(self, pln: PhotonPlan):

        self.use_custom_primary_photon_fluence = False
        self.kernel_cutoff = 10.0
        self.random_seed = 0
        self.int_conv_resolution = 0.5
        self.enable_dij_sampling = True
        self.dij_sampling = DijSamplingConfig(
            rel_dose_threshold=0.01, lat_cut_off=20, type="radius", delta_rad_depth=5
        )

        super().__init__(pln)

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

        # matRad does the gaussian filter here, but can we do that here?
        # We should be as flexible as to allow beams with different energies / penumbras / kernels
        # moved the kernel filtering to beam initialization

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

        field_based_dose_calc = False
        if self._is_field_based_dose_calc:
            logger.debug("Enabling field-based dose calculation for beam %d!", i)
            self.int_conv_resolution = self._collimation["conv_resolution"]
            field_width = self._collimation["field_width"]
            field_based_dose_calc = True
        else:
            logger.debug("Enabling bixel-based dose calculation for beam %d!", i)
            field_width = beam_info["beam"]["bixel_width"]

        # TODO: obtain maximum field limits

        field_limit = np.ceil(field_width / (2 * self.int_conv_resolution))
        # TODO: should this be +1 as end or is it correct? In matRad it is equivalent to no +1
        field_grid = self.int_conv_resolution * np.arange(-field_limit, field_limit)
        beam_info["f_x"], beam_info["f_z"] = np.meshgrid(field_grid, field_grid, indexing="xy")

        # Get the kernel
        beamlets = [beamlet for ray in beam_info["beam"]["rays"] for beamlet in ray["beamlets"]]

        energies = np.unique([beamlet["energy"] for beamlet in beamlets])

        if len(energies) > 1:
            raise ValueError("Different energies in one photon beam not supported yet.")

        energy = energies[0]

        kernel = cast(PhotonLINAC, self._machine).get_kernel_by_energy(energy)

        sigma_gauss = kernel.penumbra / np.sqrt(8 * np.log(2))  # [mm]

        # use 5 times sigma as the limits for the gaussian convolution
        gauss_limit = np.ceil(5 * sigma_gauss / self.int_conv_resolution)
        gauss_grid = self.int_conv_resolution * np.arange(-gauss_limit, gauss_limit)

        gauss_filter_x, gauss_filter_z = np.meshgrid(gauss_grid, gauss_grid, indexing="xy")
        gauss_filter = (
            1.0
            / (2 * np.pi * sigma_gauss**2 / self.int_conv_resolution**2)
            * np.exp(-(gauss_filter_x**2 + gauss_filter_z**2) / (2 * sigma_gauss**2))
        )

        gauss_conv_size = 2 * (field_limit + gauss_limit).astype(int)

        # get kernel size and distances
        kernel_limit = np.ceil(self.kernel_cutoff / self.int_conv_resolution)
        kernel_grid = self.int_conv_resolution * np.arange(-kernel_limit, kernel_limit)

        kernel_x, kernel_z = np.meshgrid(kernel_grid, kernel_grid, indexing="xy")

        # calculate also the total size and distance as we need this during convolution extensively
        kernel_conv_limit = field_limit + gauss_limit + kernel_limit
        kernel_conv_grid = self.int_conv_resolution * np.arange(
            -kernel_conv_limit, kernel_conv_limit
        )
        conv_mx_x, conv_mx_z = np.meshgrid(kernel_conv_grid, kernel_conv_grid, indexing="xy")

        kernel_conv_size = 2 * kernel_conv_limit.astype(int)

        effective_lateral_cut_off = self.geometric_lateral_cutoff + field_width / np.sqrt(2)

        if not field_based_dose_calc:
            n = np.floor(field_width / self.int_conv_resolution).astype(int)
            f_pre = np.ones((n, n), dtype=np.float32)

            if not self.use_custom_primary_photon_fluence:
                f_pre = fft.ifft2(
                    fft.fft2(f_pre, (gauss_conv_size, gauss_conv_size))
                    * fft.fft2(gauss_filter, (gauss_conv_size, gauss_conv_size))
                )
                f_pre = np.real(f_pre)

        beam_info["f_pre"] = f_pre
        beam_info["kernel_xz"] = (kernel_x, kernel_z)
        beam_info["conv_mx_xz"] = (conv_mx_x, conv_mx_z)

        raise NotImplementedError("You shall not pass!")
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

        return beam_info

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
