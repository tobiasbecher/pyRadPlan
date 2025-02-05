"""Siddon Ray Tracing Algorithm for Voxelized Geometry."""

from typing import Union
import numpy as np
import SimpleITK as sitk
from pyRadPlan.raytracer._base import RayTracerBase

# from ._perf import _fast_compute_all_alphas, _fast_compute_plane_alphas


class RayTracerSiddon(RayTracerBase):
    """
    Implementation for Siddon Ray Tracing Algorithm through voxelized
    geometry.
    """

    # @jit(nopython=True)
    def trace_ray(
        self,
        isocenter: Union[list, np.ndarray],
        source_points: Union[list, np.ndarray],
        target_points: Union[list, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Traces an individual ray
        Detailed explanation goes here.
        """

        # Check type and dimension of target_point (should be nx3)
        # Check type and dimension of source_point (should be nx3)
        # Check type and dimension of cubes (should be either a list of or a single 3D numpy array)
        # Check type and dimension of resolution (should be a dictionary with keys 'x', 'y', 'z'
        # and values as floats)
        isocenter = np.asarray(isocenter)
        source_points = np.asarray(source_points)
        target_points = np.asarray(target_points)

        if target_points.size != 3 or source_points.size != 3 or isocenter.size != 3:
            raise ValueError(
                "Number of target Points and source points needs to be equal to one! If you want "
                "to trace multiple rays at once, use trace_rays instead!"
            )
        alphas, lengths, rho, d12, ix = self.trace_rays(
            isocenter, source_points.reshape((1, 3)), target_points.reshape((1, 3))
        )

        # Squeeze Dimensions
        alphas = alphas.squeeze()
        lengths = lengths.squeeze()
        rho = [r.squeeze() for r in rho]
        ix = ix.squeeze()

        return alphas, lengths, rho, d12, ix

    def trace_rays(
        self,
        isocenter: Union[list, np.ndarray],
        source_points: Union[list, np.ndarray],
        target_points: Union[list, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Vectorized Implementation of RayTracing.
        Uses padding to create matrices of ray information.

        Notes
        -----
        Currently, the vectorized implementation uses padding with NaN values. This is not the most
        efficient way to handle the different lengths of the rays. A more efficient way would be to
        use more performant padding values (e.g. an unrealistically large value like the respective
        maximum floating point value)
        """

        num_rays = target_points.shape[0]
        num_sources = source_points.shape[0]

        if num_sources != num_rays and num_sources != 1:
            raise ValueError(
                f"Number of source points ({num_sources}) needs to be one or equal to number of "
                f"target points ({num_rays})!"
            )
        if num_sources == 1:
            source_points = np.tile(source_points, (num_rays, 1))
            num_sources = num_rays

        self._source_points = (source_points + isocenter).astype(self.precision)
        self._target_points = (target_points + isocenter).astype(self.precision)
        self._ray_vec = self._target_points - self._source_points

        alphas = self._compute_all_alphas()

        d12 = np.linalg.norm(self._ray_vec, axis=1, keepdims=True)

        tmp_diff = np.diff(alphas, axis=1)

        lengths = d12 * tmp_diff
        alphas_mid = alphas[:, :-1] + 0.5 * tmp_diff

        cube_origin = np.asarray(self._cubes[0].GetOrigin())

        # Compute coordinates
        i_float = (
            self._source_points[:, 0, np.newaxis]
            + alphas_mid * self._ray_vec[:, 0, np.newaxis]
            - cube_origin[0]
        ) / self._resolution[0]
        j_float = (
            self._source_points[:, 1, np.newaxis]
            + alphas_mid * self._ray_vec[:, 1, np.newaxis]
            - cube_origin[1]
        ) / self._resolution[1]
        k_float = (
            self._source_points[:, 2, np.newaxis]
            + alphas_mid * self._ray_vec[:, 2, np.newaxis]
            - cube_origin[2]
        ) / self._resolution[2]

        # Round and convert to int
        i = np.full_like(i_float, -1, dtype=np.int32)
        j = np.full_like(j_float, -1, dtype=np.int32)
        k = np.full_like(k_float, -1, dtype=np.int32)

        i[np.isfinite(i_float)] = np.round(i_float[np.isfinite(i_float)]).astype(np.int32)
        j[np.isfinite(j_float)] = np.round(j_float[np.isfinite(j_float)]).astype(np.int32)
        k[np.isfinite(k_float)] = np.round(k_float[np.isfinite(k_float)]).astype(np.int32)

        # sanitize
        val_ix = (
            ~np.isnan(alphas_mid)
            & np.logical_and(i >= 0, i < self._cube_dim[0])
            & np.logical_and(j >= 0, j < self._cube_dim[1])
            & np.logical_and(k >= 0, k < self._cube_dim[2])
        )

        valid_indices = np.nonzero(val_ix)
        if all(arr.size == 0 for arr in valid_indices):
            alphas = np.empty((num_rays, 0), dtype=self.precision)
            lengths = np.empty((num_rays, 0), dtype=self.precision)
            rho = [np.empty((num_rays, 0), dtype=self.precision) for _ in self._cubes]
            ix = np.empty((num_rays, 0), dtype=np.int64)

        else:
            ix = np.full(val_ix.shape, -1, dtype=np.int64)

            ix[valid_indices] = np.ravel_multi_index(
                (i[valid_indices], j[valid_indices], k[valid_indices]),
                self._cube_dim,
            )

            rho = [np.full(val_ix.shape, np.nan) for _ in self._cubes]
            for s, cube in enumerate(self._cubes):
                # Views SimpleITKs image buffer as a numpy array, preserving dimension ordering of
                # sitk
                cube_linear = sitk.GetArrayViewFromImage(cube).ravel(order="F")
                rho[s][valid_indices] = cube_linear[ix[valid_indices].astype(int)]

        return alphas, lengths, rho, d12, ix

    def _compute_all_alphas(self) -> np.ndarray:
        """
        Here we setup grids to enable logical indexing when computing
        the alphas along each dimension. All alphas between the
        minimum and maximum index will be computed, with additional
        exclusion of singular plane occurrences (max == min)
        All values out of scope will be set to NaN.
        """

        num_rays = self._ray_vec.shape[0]

        a_x_min = np.zeros(num_rays, dtype=self.precision)
        a_x_max = np.ones(num_rays, dtype=self.precision)
        a_x_ix = np.nonzero(self._ray_vec[:, 0])
        if len(a_x_ix[0]) > 0:
            tmp = (
                np.column_stack(
                    (
                        self._x_planes[0] - self._source_points[a_x_ix[0], 0],
                        self._x_planes[-1] - self._source_points[a_x_ix[0], 0],
                    )
                )
                / self._ray_vec[a_x_ix[0], 0, None]
            )
            a_x_min[a_x_ix[0]] = np.nanmin(tmp, axis=1)
            a_x_max[a_x_ix[0]] = np.nanmax(tmp, axis=1)

        a_y_min = np.zeros(num_rays, dtype=self.precision)
        a_y_max = np.ones(num_rays, dtype=self.precision)
        a_y_ix = np.nonzero(self._ray_vec[:, 1])
        if len(a_y_ix[0]) > 0:
            tmp = (
                np.column_stack(
                    (
                        self._y_planes[0] - self._source_points[a_y_ix[0], 1],
                        self._y_planes[-1] - self._source_points[a_y_ix[0], 1],
                    )
                )
                / self._ray_vec[a_y_ix[0], 1, None]
            )
            a_y_min[a_y_ix[0]] = np.nanmin(tmp, axis=1)
            a_y_max[a_y_ix[0]] = np.nanmax(tmp, axis=1)

        a_z_min = np.zeros(num_rays, dtype=self.precision)
        a_z_max = np.ones(num_rays, dtype=self.precision)
        a_z_ix = np.nonzero(self._ray_vec[:, 2])
        if len(a_z_ix[0]) > 0:
            tmp = (
                np.column_stack(
                    (
                        self._z_planes[0] - self._source_points[a_z_ix[0], 2],
                        self._z_planes[-1] - self._source_points[a_z_ix[0], 2],
                    )
                )
                / self._ray_vec[a_z_ix[0], 2, None]
            )
            a_z_min[a_z_ix[0]] = np.nanmin(tmp, axis=1)
            a_z_max[a_z_ix[0]] = np.nanmax(tmp, axis=1)

        # Compute alpha_limits
        alpha_min_values = np.nanmax(
            np.column_stack(
                (
                    np.zeros(num_rays, dtype=self.precision),
                    a_x_min,
                    a_y_min,
                    a_z_min,
                )
            ),
            axis=1,
        )
        alpha_max_values = np.nanmin(
            np.column_stack(
                (
                    np.ones(num_rays, dtype=self.precision),
                    a_x_max,
                    a_y_max,
                    a_z_max,
                )
            ),
            axis=1,
        )

        alpha_limits = np.column_stack((alpha_min_values, alpha_max_values))

        # Determine the direction of ray movement
        ix_tp_bigger_sp = self._target_points > self._source_points
        ix_tp_smaller_sp = self._target_points < self._source_points

        for d in range(3):  # Loop through x, y, z dimensions
            alpha_tmp = np.full((num_rays, 2), np.nan, dtype=self.precision)
            alpha_tmp[ix_tp_bigger_sp[:, d], 0] = alpha_limits[ix_tp_bigger_sp[:, d], 0]
            alpha_tmp[ix_tp_smaller_sp[:, d], 0] = alpha_limits[ix_tp_smaller_sp[:, d], 1]
            alpha_tmp[ix_tp_bigger_sp[:, d], 1] = alpha_limits[ix_tp_bigger_sp[:, d], 1]
            alpha_tmp[ix_tp_smaller_sp[:, d], 1] = alpha_limits[ix_tp_smaller_sp[:, d], 0]

            # Compute min and max indices (i_min, i_max, j_min, j_max, k_min, k_max)
            if d == 0:  # x dimension
                planes = self._x_planes
                resolution = self._resolution[0]
            elif d == 1:  # y dimension
                planes = self._y_planes
                resolution = self._resolution[1]
            else:  # z dimension
                planes = self._z_planes
                resolution = self._resolution[2]

            dim_min = (
                self._num_planes[d]
                - (planes[-1] - alpha_tmp[:, 0] * self._ray_vec[:, d] - self._source_points[:, d])
                / resolution
                - 1
            )
            dim_max = (
                self._source_points[:, d] + alpha_tmp[:, 1] * self._ray_vec[:, d] - planes[0]
            ) / resolution
            # Rounding
            dim_min = np.ceil(1 / 1000 * (np.round(1000 * dim_min)))
            dim_max = np.floor(1 / 1000 * (np.round(1000 * dim_max)))

            if d == 0:  # Assigning computed min and max indices to i_min, i_max
                i_min, i_max = dim_min, dim_max
            elif d == 1:  # Assigning computed min and max indices to j_min, j_max
                j_min, j_max = dim_min, dim_max
            else:  # Assigning computed min and max indices to k_min, k_max
                k_min, k_max = dim_min, dim_max

        # Compute alphas for each plane and merge parametric sets

        alpha_x = self._compute_plane_alphas(
            i_min,
            i_max,
            self._x_planes,
            self._source_points[:, 0],
            self._ray_vec[:, 0],
        )
        alpha_y = self._compute_plane_alphas(
            j_min,
            j_max,
            self._y_planes,
            self._source_points[:, 1],
            self._ray_vec[:, 1],
        )
        alpha_z = self._compute_plane_alphas(
            k_min,
            k_max,
            self._z_planes,
            self._source_points[:, 2],
            self._ray_vec[:, 2],
        )

        alphas = np.concatenate((alpha_limits, alpha_x, alpha_y, alpha_z), axis=1)

        # Row wise Unique (could be externalized and potentially jitted if performance critical)
        np_unique = np.unique  # Lookup function once to improve performance
        for i in range(alphas.shape[0]):
            v = np_unique(alphas[i, :])
            alphas[i, : v.size] = v
            alphas[i, v.size + 1 :] = np.nan

        # Size Reduction
        max_num_columns = np.max(np.sum(~np.isnan(alphas), axis=1))
        alphas = alphas[:, :max_num_columns]

        return alphas

    def _compute_plane_alphas(self, dim_min, dim_max, planes, source, ray) -> np.ndarray:
        num_rays = source.shape[0]
        dim_min = np.atleast_2d(dim_min).T
        dim_max = np.atleast_2d(dim_max).T

        plane_grid = np.tile(planes, (num_rays, 1))
        plane_ix = np.atleast_2d(np.arange(0, len(planes)))

        # set invalid indices to NaN
        ix = (
            (plane_ix < dim_min)
            | (plane_ix > dim_max)
            | ((plane_ix == dim_min) & (plane_ix == dim_max))
            | np.isnan(dim_min)
            | np.isnan(dim_max)
        )
        # np.isnan()

        plane_grid[ix] = np.nan

        alphas = (plane_grid - np.tile(source, (plane_grid.shape[1], 1)).T) / np.tile(
            ray, (plane_grid.shape[1], 1)
        ).T

        # for i in range(plane_grid.shape[0]):
        #     plane_grid[i, :] = plane_ix + dim_min[i]
        # planeCoordinates = plane_grid * resolution + planes[0]
        # alphas = (planeCoordinates - np.tile(source, (plane_grid.shape[1], 1)).T) / np.tile(ray,
        # (plane_grid.shape[1], 1)).T

        return alphas

    def _initialize_geometry(self):
        """
        Initializes the geometry for the ray tracing.

        Parameters
        ----------
        resolution : dict
            Resolution of the cubes in millimeters per voxel. Keys should be 'x', 'y', and 'z'.
        cube_dim : list or tuple
            Dimensions of the cube in number of voxels along each axis [x, y, z].

        Attributes
        ----------
        self._source_points : numpy.ndarray
            Source point(s) of ray tracing. Either a 1x3 array or an Nx3 matrix (with N rays to
            trace).
            If 1x3 with multiple target points, the one source point will be used for all rays.
        self._target_points : numpy.ndarray
            Target point(s) of ray tracing. An Nx3 matrix where N is the number of rays to trace.
        self._cube_dim : list or tuple
            Dimensions of the cube in number of voxels along each axis [x, y, z].
        self._resolution : dict
            Resolution of the cubes in millimeters per voxel. Keys should be 'x', 'y', and 'z'.
        self._num_planes : list
            Number of planes along each axis [x, y, z].
        self._x_planes : numpy.ndarray
            Positions of the planes along the x-axis in millimeters.
        self._y_planes : numpy.ndarray
            Positions of the planes along the y-axis in millimeters.
        self._z_planes : numpy.ndarray
            Positions of the planes along the z-axis in millimeters.
        self._ray_vec : numpy.ndarray
            Vector from source points to target points.

        Notes
        -----
        For a detailed description of the variables, see Siddon 1985 Medical Physics.
        """

        ref_cube = self._cubes[0]

        if ref_cube.GetDimension() != 3:
            raise ValueError("Only 3D cubes are supported by RayTracerSiddon!")

        origin = np.asarray(ref_cube.GetOrigin()).astype(self.precision)
        self._resolution = np.asarray(ref_cube.GetSpacing()).astype(self.precision)
        direction = np.asarray(ref_cube.GetDirection()).reshape(3, 3).astype(self.precision)
        self._cube_dim = np.asarray(ref_cube.GetSize())

        increment = np.zeros_like(origin)
        increment[0] = (direction @ np.array([1, 0, 0], dtype=self.precision))[
            0
        ] * self._resolution[0]
        increment[1] = (direction @ np.array([0, 1, 0], dtype=self.precision))[
            1
        ] * self._resolution[1]
        increment[2] = (direction @ np.array([0, 0, 1], dtype=self.precision))[
            2
        ] * self._resolution[2]

        self._x_planes = (
            origin[0]
            + (np.arange(self._cube_dim[0] + 1, dtype=self.precision) - 0.5) * increment[0]
        )
        self._y_planes = (
            origin[1]
            + (np.arange(self._cube_dim[1] + 1, dtype=self.precision) - 0.5) * increment[1]
        )
        self._z_planes = (
            origin[2]
            + (np.arange(self._cube_dim[2] + 1, dtype=self.precision) - 0.5) * increment[2]
        )

        self._num_planes = [len(self._x_planes), len(self._y_planes), len(self._z_planes)]
