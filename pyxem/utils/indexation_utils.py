# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.


from itertools import combinations
from operator import attrgetter

import numpy as np
import scipy

from pyxem.utils.expt_utils import _cart2polar
from pyxem.utils.vector_utils import get_rotation_matrix_between_vectors
from pyxem.utils.vector_utils import get_angle_cartesian
from pyxem.utils.cuda_utils import (
    is_cupy_array,
    to_numpy,
    get_array_module,
    _correlate_polar_image_to_library_gpu,
)

from transforms3d.euler import mat2euler

from collections import namedtuple

from pyxem.utils.dask_tools import _get_dask_array
import os
from dask.diagnostics import ProgressBar
from numba import njit, float64, float32, int32, prange

from pyxem.utils.polar_transform_utils import (
    _cartesian_positions_to_polar,
    get_polar_pattern_shape,
    image_to_polar,
    get_template_polar_coordinates,
    _chunk_to_polar,
)

try:
    import cupy as cp

    CUPY_INSTALLED = True
    import cupyx.scipy as spgpu
except ImportError:
    CUPY_INSTALLED = False


# container for OrientationResults
OrientationResult = namedtuple(
    "OrientationResult",
    "phase_index rotation_matrix match_rate error_hkls total_error scale center_x center_y".split(),
)


def get_nth_best_solution(
    single_match_result, mode, rank=0, key="match_rate", descending=True
):
    """Get the nth best solution by match_rate from a pool of solutions

    Parameters
    ----------
    single_match_result : VectorMatchingResults, TemplateMatchingResults
        Pool of solutions from the vector matching algorithm
    mode : str
        'vector' or 'template'
    rank : int
        The rank of the solution, i.e. rank=2 returns the third best solution
    key : str
        The key to sort the solutions by, default = match_rate
    descending : bool
        Rank the keys from large to small

    Returns
    -------
    VectorMatching:
        best_fit : `OrientationResult`
            Parameters for the best fitting orientation
            Library Number, rotation_matrix, match_rate, error_hkls, total_error
    TemplateMatching: np.array
            Parameters for the best fitting orientation
            Library Number , [z, x, z], Correlation Score
    """
    if mode == "vector":
        try:
            best_fit = sorted(
                single_match_result[0].tolist(), key=attrgetter(key), reverse=descending
            )[rank]
        except AttributeError:
            best_fit = sorted(
                single_match_result.tolist(), key=attrgetter(key), reverse=descending
            )[rank]
    if mode == "template":
        srt_idx = np.argsort(single_match_result[:, 2])[::-1][rank]
        best_fit = single_match_result[srt_idx]

    return best_fit


# Functions used in correlate_library.
def fast_correlation(image_intensities, int_local, pn_local, **kwargs):
    r"""Computes the correlation score between an image and a template

    Uses the formula

    .. math:: FastCorrelation
        \\frac{\\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)}{\\sqrt{\\sum_{j=1}^m T^2(x_j, y_j)}}

    Parameters
    ----------
    image_intensities: list
        list of intensity values in the image, for pixels where the template has a non-zero intensity
     int_local: list
        list of all non-zero intensities in the template
     pn_local: float
        pattern norm of the template

    Returns
    -------
    corr_local: float
        correlation score between template and image.

    See Also
    --------
    correlate_library, zero_mean_normalized_correlation

    """
    return (
        np.sum(np.multiply(image_intensities, int_local)) / pn_local
    )  # Correlation is the partially normalized dot product


def zero_mean_normalized_correlation(
    nb_pixels,
    image_std,
    average_image_intensity,
    image_intensities,
    int_local,
    **kwargs,
):
    r"""Computes the correlation score between an image and a template.

    Uses the formula

    .. math:: zero_mean_normalized_correlation
        \\frac{\\sum_{j=1}^m P(x_j, y_j) T(x_j, y_j)- avg(P)avg(T)}{\\sqrt{\\sum_{j=1}^m (T(x_j, y_j)-avg(T))^2+\\sum_{Not {j}} avg(T)}}
        for a template T and an experimental pattern P.

    Parameters
    ----------
    nb_pixels: int
        total number of pixels in the image
    image_std: float
        Standard deviation of intensities in the image.
    average_image_intensity: float
        average intensity for the image
    image_intensities: list
        list of intensity values in the image, for pixels where the template has a non-zero intensity
     int_local: list
        list of all non-zero intensities in the template
     pn_local: float
        pattern norm of the template

    Returns
    -------
    corr_local: float
        correlation score between template and image.

    See Also
    --------
    correlate_library, fast_correlation

    """

    nb_pixels_star = len(int_local)
    average_pattern_intensity = nb_pixels_star * np.average(int_local) / nb_pixels

    match_numerator = (
        np.sum(np.multiply(image_intensities, int_local))
        - nb_pixels * average_image_intensity * average_pattern_intensity
    )
    match_denominator = image_std * (
        np.linalg.norm(int_local - average_pattern_intensity)
        + (nb_pixels - nb_pixels_star) * pow(average_pattern_intensity, 2)
    )

    if match_denominator == 0:
        corr_local = 0
    else:
        corr_local = (
            match_numerator / match_denominator
        )  # Correlation is the normalized dot product

    return corr_local


def index_magnitudes(z, simulation, tolerance):
    """Assigns hkl indices to peaks in the diffraction profile.

    Parameters
    ----------
    simulation : DiffractionProfileSimulation
        Simulation of the diffraction profile.
    tolerance : float
        The n orientations with the highest correlation values are returned.

    Returns
    -------
    indexation : np.array()
        indexation results.

    """
    mags = z
    sim_mags = np.array(simulation.magnitudes)
    sim_hkls = np.array(simulation.hkls)
    indexation = np.zeros(len(mags), dtype=object)

    for i in np.arange(len(mags)):
        diff = np.absolute((sim_mags - mags.data[i]) / mags.data[i] * 100)

        hkls = sim_hkls[np.where(diff < tolerance)]
        diffs = diff[np.where(diff < tolerance)]

        indices = np.array((hkls, diffs))
        indexation[i] = np.array((mags.data[i], indices))

    return indexation


def _choose_peak_ids(peaks, n_peaks_to_index):
    """Choose `n_peaks_to_index` indices from `peaks`.

    This implementation sorts by angle and then picks every
    len(peaks)/n_peaks_to_index element to get an even distribution of angles.

    Parameters
    ----------
    peaks : array_like
        Array of peak positions.
    n_peaks_to_index : int
        Number of indices to return.

    Returns
    -------
    peak_ids : numpy.array
        Array of indices of the chosen peaks.
    """
    r, angles = _cart2polar(peaks[:, 0], peaks[:, 1])
    return angles.argsort()[
        np.linspace(0, angles.shape[0] - 1, n_peaks_to_index, dtype=np.int)
    ]


def match_vectors(
    peaks, library, mag_tol, angle_tol, index_error_tol, n_peaks_to_index, n_best
):
    # TODO: Sort peaks by intensity or SNR
    """Assigns hkl indices to pairs of diffraction vectors.

    Parameters
    ----------
    peaks : np.array()
        The experimentally measured diffraction vectors, associated with a
        particular probe position, to be indexed. In Cartesian coordinates.
    library : VectorLibrary
        Library of reciprocal space vectors to be matched to the vectors.
    mag_tol : float
        Max allowed magnitude difference when comparing vectors.
    angle_tol : float
        Max allowed angle difference in radians when comparing vector pairs.
    index_error_tol : float
        Max allowed error in peak indexation for classifying it as indexed,
        calculated as :math:`|hkl_calculated - round(hkl_calculated)|`.
    n_peaks_to_index : int
        The maximum number of peak to index.
    n_best : int
        The maximum number of good solutions to be retained for each phase.

    Returns
    -------
    indexation : np.array()
        A numpy array containing the indexation results, each result consisting of 5 entries:
            [phase index, rotation matrix, match rate, error hkls, total error]

    """
    if peaks.shape == (1,) and peaks.dtype == np.object:
        peaks = peaks[0]

    # Assign empty array to hold indexation results. The n_best best results
    # from each phase is returned.
    top_matches = np.empty(len(library) * n_best, dtype="object")
    res_rhkls = []

    # Iterate over phases in DiffractionVectorLibrary and perform indexation
    # on each phase, storing the best results in top_matches.
    for phase_index, (phase, structure) in enumerate(
        zip(library.values(), library.structures)
    ):
        solutions = []
        lattice_recip = structure.lattice.reciprocal()
        phase_indices = phase["indices"]
        phase_measurements = phase["measurements"]

        if peaks.shape[0] < 2:  # pragma: no cover
            continue

        # Choose up to n_peaks_to_index unindexed peaks to be paired in all
        # combinations.
        # TODO: Matching can be done iteratively where successfully indexed
        #       peaks are removed after each iteration. This can possibly
        #       handle overlapping patterns.
        # unindexed_peak_ids = range(min(peaks.shape[0], n_peaks_to_index))
        # TODO: Better choice of peaks (longest, highest SNR?)
        # TODO: Inline after choosing the best, and possibly require external sorting (if using sorted)?
        unindexed_peak_ids = _choose_peak_ids(peaks, n_peaks_to_index)

        # Find possible solutions for each pair of peaks.
        for vector_pair_index, peak_pair_indices in enumerate(
            list(combinations(unindexed_peak_ids, 2))
        ):
            # Consider a pair of experimental scattering vectors.
            q1, q2 = peaks[peak_pair_indices, :]
            q1_len, q2_len = np.linalg.norm(q1), np.linalg.norm(q2)

            # Ensure q1 is longer than q2 for consistent order.
            if q1_len < q2_len:  # pragma: no cover
                q1, q2 = q2, q1
                q1_len, q2_len = q2_len, q1_len

            # Calculate the angle between experimental scattering vectors.
            angle = get_angle_cartesian(q1, q2)

            # Get library indices for hkls matching peaks within tolerances.
            # TODO: phase are object arrays. Test performance of direct float arrays
            tolerance_mask = np.abs(phase_measurements[:, 0] - q1_len) < mag_tol
            tolerance_mask[tolerance_mask] &= (
                np.abs(phase_measurements[tolerance_mask, 1] - q2_len) < mag_tol
            )
            tolerance_mask[tolerance_mask] &= (
                np.abs(phase_measurements[tolerance_mask, 2] - angle) < angle_tol
            )

            # Iterate over matched library vectors determining the error in the
            # associated indexation.
            if np.count_nonzero(tolerance_mask) == 0:  # pragma: no cover
                continue

            # Reference vectors are cartesian coordinates of hkls
            reference_vectors = lattice_recip.cartesian(phase_indices[tolerance_mask])

            # Rotation from experimental to reference frame
            rotations = get_rotation_matrix_between_vectors(
                q1, q2, reference_vectors[:, 0], reference_vectors[:, 1]
            )

            # Index the peaks by rotating them to the reference coordinate
            # system. Use rotation directly since it is multiplied from the
            # right. Einsum gives list of peaks.dot(rotation).
            hklss = lattice_recip.fractional(np.einsum("ijk,lk->ilj", rotations, peaks))

            # Evaluate error of peak hkl indexation
            rhklss = np.rint(hklss)
            ehklss = np.abs(hklss - rhklss)
            valid_peak_mask = np.max(ehklss, axis=-1) < index_error_tol
            valid_peak_counts = np.count_nonzero(valid_peak_mask, axis=-1)
            error_means = ehklss.mean(axis=(1, 2))

            num_peaks = len(peaks)
            match_rates = (valid_peak_counts * (1 / num_peaks)) if num_peaks else 0

            possible_solution_mask = match_rates > 0
            solutions += [
                OrientationResult(
                    phase_index=phase_index,
                    rotation_matrix=R,
                    match_rate=match_rate,
                    error_hkls=ehkls,
                    total_error=error_mean,
                    scale=1.0,
                    center_x=0.0,
                    center_y=0.0,
                )
                for R, match_rate, ehkls, error_mean in zip(
                    rotations[possible_solution_mask],
                    match_rates[possible_solution_mask],
                    ehklss[possible_solution_mask],
                    error_means[possible_solution_mask],
                )
            ]

            res_rhkls += rhklss[possible_solution_mask].tolist()

        n_solutions = min(n_best, len(solutions))

        i = phase_index * n_best  # starting index in unfolded array

        if n_solutions > 0:
            top_n = sorted(solutions, key=attrgetter("match_rate"), reverse=True)[
                :n_solutions
            ]

            # Put the top n ranked solutions in the output array
            top_matches[i : i + n_solutions] = top_n

        if n_solutions < n_best:
            # Fill with dummy values
            top_matches[i + n_solutions : i + n_best] = [
                OrientationResult(
                    phase_index=0,
                    rotation_matrix=np.identity(3),
                    match_rate=0.0,
                    error_hkls=np.array([]),
                    total_error=1.0,
                    scale=1.0,
                    center_x=0.0,
                    center_y=0.0,
                )
                for x in range(n_best - n_solutions)
            ]

    # Because of a bug in numpy (https://github.com/numpy/numpy/issues/7453),
    # triggered by the way HyperSpy reads results (np.asarray(res), which fails
    # when the two tuple values have the same first dimension), we cannot
    # return a tuple directly, but instead have to format the result as an
    # array ourselves.
    res = np.empty(2, dtype=np.object)
    res[0] = top_matches
    res[1] = np.asarray(res_rhkls)
    return res


def _simulations_to_arrays(simulations, max_radius=None):
    """
    Convert simulation results to arrays of diffraction spots

    Parameters
    ----------
    simulations : list
        list of diffsims.sims.diffraction_simulation.DiffractionSimulation
        objects
    max_radius : float
        limit to g-vector length in pixel coordinates

    Returns
    -------
    positions : numpy.ndarray (N, 2, R)
        An array containing all (x,y) coordinates of reflections of N templates. R represents
        the maximum number of reflections; templates containing fewer
        reflections are padded with 0's at the end. In pixel units.
    intensities : numpy.ndarray (N, R)
        An array containing all intensities of reflections of N templates
    """
    num_spots = [i.intensities.shape[0] for i in simulations]
    max_spots = max(num_spots)
    positions = np.zeros((len(simulations), 2, max_spots), dtype=np.float64)
    intensities = np.zeros((len(simulations), max_spots), dtype=np.float64)
    for i, j in enumerate(simulations):
        x = j.calibrated_coordinates[:, 0]
        y = j.calibrated_coordinates[:, 1]
        intensity = j.intensities
        if max_radius is not None:
            condition = x ** 2 + y ** 2 < max_radius ** 2
            x = x[condition]
            y = y[condition]
            intensity = intensity[condition]
        positions[i, 0, : x.shape[0]] = x
        positions[i, 1, : y.shape[0]] = y
        intensities[i, : intensity.shape[0]] = intensity
    return positions, intensities


def _match_polar_to_polar_template(
    polar_image,
    r_template,
    theta_template,
    intensities,
):
    """
    Correlate a single polar template to a single polar image

    The template spots are shifted along the azimuthal axis by 1 pixel increments.
    A simple correlation index is calculated at each position.

    Parameters
    ----------
    polar_image : 2D ndarray
        the polar image
    r_template : 1D ndarray
        r coordinates of diffraction spots in template
    theta_template : 1D ndarray
        theta coordinates of diffraction spots in template
    intensities : 1D ndarray
        intensities of diffraction spots in template

    Returns
    -------
    correlation : 1D ndarray
        correlation index at each in-plane angle position
    """
    dispatcher = get_array_module(polar_image)
    sli = polar_image[:, r_template]
    rows, column_indices = dispatcher.ogrid[: sli.shape[0], : sli.shape[1]]
    rows = dispatcher.mod(rows + theta_template[None, :], polar_image.shape[0])
    extr = sli[rows, column_indices].astype(intensities.dtype)
    correlation = dispatcher.dot(extr, intensities)
    return correlation


@njit(parallel=True, nogil=True)
def _correlate_polar_to_library_cpu(
    polar_image,
    r_templates,
    theta_templates,
    intensities_templates,
):
    """
    Correlates a polar pattern to all polar templates at all in_plane angles

    This is a direct copy of the cuda kernel serving for comparison.

    Parameters
    ----------
    polar_image : (T, R) 2D numpy.ndarray
        The image converted to polar coordinates
    r_templates : (N, D) 2D numpy.ndarray
        r-coordinates of diffraction spots in templates.
    theta_templates : (N, D) 2D numpy ndarray
        theta-coordinates of diffraction spots in templates.
    intensities_templates : (N, D) 2D numpy.ndarray
        intensities of the spots in each template

    Returns
    -------
    correlations : (N, T) 2D numpy.ndarray
        the correlation index for each template at all in-plane angles with the image
    """
    correlation = np.empty(
        (r_templates.shape[0], polar_image.shape[0]), dtype=polar_image.dtype
    )
    for template in prange(r_templates.shape[0]):
        for shift in prange(polar_image.shape[0]):
            tmp = 0
            for spot in range(r_templates.shape[1]):
                tmp += (
                    polar_image[
                        (theta_templates[template, spot] + shift)
                        % polar_image.shape[0],
                        r_templates[template, spot],
                    ]
                    * intensities_templates[template, spot]
                )
            correlation[template, shift] = tmp
    return correlation


def _match_polar_to_library_cpu(
    polar_image,
    r_templates,
    theta_templates,
    intensities_templates,
):
    """
    Correlates a polar pattern to all polar templates on CPU

    Parameters
    ----------
    polar_image : 2D numpy.ndarray
        The image converted to polar coordinates
    r_templates : 2D numpy.ndarray
        r-coordinates of diffraction spots in templates.
    theta_templates : 2D numpy ndarray
        theta-coordinates of diffraction spots in templates.
    intensities_templates : 2D numpy.ndarray
        intensities of the spots in each template

    Returns
    -------
    best_in_plane_shift : (N) 1D numpy.ndarray
        Shift for all templates that yields best correlation
    best_in_plane_shift_m : (N) 1D numpy.ndarray
        Shift for all mirrored templates that yields best correlation
    best_in_plane_corr : (N) 1D numpy.ndarray
        Correlation at best match for each template
    best_in_plane_corr_m : (N) 1D numpy.ndarray
        Correlation at best match for each mirrored template

    Notes
    -----
    The dimensions of r_templates and theta_templates should be (N, R) where
    N is the number of templates and R the number of spots in the template
    with the maximum number of spots
    """
    correlations = _correlate_polar_to_library_cpu(
        polar_image, r_templates, theta_templates, intensities_templates
    )
    correlations_mirror = _correlate_polar_to_library_cpu(
        polar_image,
        r_templates,
        (polar_image.shape[0] - theta_templates) % polar_image.shape[0],
        intensities_templates,
    )
    return _get_best_correlations_and_angles(correlations, correlations_mirror)


def _match_polar_to_polar_library_gpu(
    pattern,
    r_templates,
    theta_templates,
    intensities_templates,
    blockspergrid,
    threadsperblock,
):
    """
    Correlates a polar pattern to all polar templates on GPU

    Parameters
    ----------
    polar_image : 2D cupy.ndarray
        The image converted to polar coordinates
    r_templates : 2D cupy.ndarray
        r-coordinates of diffraction spots in templates.
    theta_templates : 2D cupy ndarray
        theta-coordinates of diffraction spots in templates.
    intensities_templates : 2D cupy.ndarray
        intensities of the spots in each template

    Returns
    -------
    best_in_plane_shift : (N) 1D cupy.ndarray
        Shift for all templates that yields best correlation
    best_in_plane_shift_m : (N) 1D cupy.ndarray
        Shift for all mirrored templates that yields best correlation
    best_in_plane_corr : (N) 1D cupy.ndarray
        Correlation at best match for each template
    best_in_plane_corr_m : (N) 1D cupy.ndarray
        Correlation at best match for each mirrored template

    Notes
    -----
    The dimensions of r_templates and theta_templates should be (N, R) where
    N is the number of templates and R the number of spots in the template
    with the maximum number of spots
    """
    correlation = cp.empty((r_templates.shape[0], pattern.shape[0]), dtype=cp.float32)
    _correlate_polar_image_to_library_gpu[blockspergrid, threadsperblock](
        pattern,
        r_templates,
        theta_templates,
        intensities_templates,
        correlation,
    )
    correlation_m = cp.empty((r_templates.shape[0], pattern.shape[0]), dtype=cp.float32)
    _correlate_polar_image_to_library_gpu[blockspergrid, threadsperblock](
        pattern,
        r_templates,
        (pattern.shape[0] - theta_templates) % pattern.shape[0],
        intensities_templates,
        correlation_m,
    )
    return _get_best_correlations_and_angles(correlation, correlation_m)


def _get_best_correlations_and_angles(correlations, correlations_m):
    """
    Get the best correlations and in-plane angles from a set of correlation
    matrices obtained from _match_polar_to_polar_library_cpu and
    _match_polar_to_polar_library_gpu
    """
    dispatcher = get_array_module(correlations)
    # find the best in-plane angles and correlations
    best_in_plane_shift = dispatcher.argmax(correlations, axis=1).astype(np.int32)
    best_in_plane_shift_m = dispatcher.argmax(correlations_m, axis=1).astype(np.int32)
    rows = dispatcher.arange(correlations.shape[0], dtype=np.int32)
    best_in_plane_corr = correlations[rows, best_in_plane_shift]
    best_in_plane_corr_m = correlations_m[rows, best_in_plane_shift_m]
    return (
        best_in_plane_shift,
        best_in_plane_shift_m,
        best_in_plane_corr,
        best_in_plane_corr_m,
    )


def _get_row_norms(array):
    """Get the norm of all rows in a 2D array"""
    norms = ((array ** 2).sum(axis=1)) ** 0.5
    return norms


def _norm_rows(array):
    """Normalize all the rows in a 2D array"""
    norms = _get_row_norms(array)
    return array / norms[:, None]


def _get_integrated_polar_templates(
    r_max, r_templates, intensities_templates, normalize_templates
):
    """
    Get an azimuthally integrated representation of the templates.

    Parameters
    ----------
    r_max : float
        maximum radial distance to consider in pixel units. Typically the
        radial width of the polar images.
    r_templates : 2D numpy or cupy ndarray
        r-coordinate of all spots in the templates. Of shape (N, R) where
        N is the number of templates and R is the number of spots in the
        template with the maximum number of spots
    intensities_templates : 2D numpy or cupy ndarray
        intensities in all spots of the templates. Of shape (N, R)
    normalize_templates : bool
        Whether to normalize the integrated templates

    Returns
    -------
    integrated_templates : 2D numpy or cupy ndarray
        Templates integrated over the azimuthal axis of shape (N, r_max)
    """
    dispatcher = get_array_module(intensities_templates)
    data = intensities_templates.ravel()
    columns = r_templates.ravel()
    rows = dispatcher.arange(r_templates.shape[0]).repeat(r_templates.shape[1])
    out_shape = (r_templates.shape[0], r_max)
    if is_cupy_array(intensities_templates):
        integrated_templates = spgpu.sparse.coo_matrix(
            (data, (rows, columns)), shape=out_shape
        ).toarray()
    else:
        integrated_templates = scipy.sparse.coo_matrix(
            (data, (rows, columns)), shape=out_shape
        ).toarray()
    if normalize_templates:
        integrated_templates = _norm_rows(integrated_templates)
    return integrated_templates


def _match_library_to_polar_fast(polar_sum, integrated_templates):
    """
    Compare a polar image to azimuthally integrated templates

    Parameters
    ----------
    polar_sum : 1D numpy array or cupy array
        the image in polar coordinates integrated along the azimuthal axis
        (shape = (r_max,))
    integrated_templates : 2D numpy array or cupy array
        azimuthally integrated templates of shape (N, r_max) with N
        the number of templates and r_max the width of the polar image

    Returns
    -------
    correlations : 1D numpy array or cupy array
        the correlation between the integrated image and the integrated
        templates. (shape = (N,))
    """
    return (integrated_templates * polar_sum).sum(axis=1)


def _prepare_image_and_templates(
    image,
    simulations,
    delta_r,
    delta_theta,
    max_r,
    intensity_transform_function,
    find_direct_beam,
    direct_beam_position,
    normalize_image,
    normalize_templates,
):
    """
    Prepare a single cartesian coordinate image and a template library for comparison

    Parameters
    ----------
    image : 2D np or cp ndarray
        The diffraction pattern in cartesian coordinates
    simulations : list
        list of diffsims.sims.diffraction_simulation.DiffractionSimulation
    delta_r : float
        sampling interval for the r coordinate in the polar image in pixels
    delta_theta : float
        sampling interval for the theta coordinate in the polar image in degrees
    max_r : float
        maximum radius to consider in polar conversion, in pixels
    intensity_transform_function : Callable
        function to apply to both the image and template intensities. Must
        accept any dimensional numpy array as input and preferably operate
        independently on individual elements
    find_direct_beam : bool
        whether to refine the direct beam position in the image polar conversion
    direct_beam_position : 2-tuple of floats
        the (x, y) position of the direct beam in the image to override any
        defaults
    normalize_image : bool
        Whether to normalize the image
    normalize_templates : bool
        Whether to normalize the template intensities

    Returns
    -------
    polar_image : 2D np or cp ndarray
        The image in polar coordinates
    r : 2D np or cp ndarray
        The r coordinates in the polar image corresponding to template spots.
        shape = (N, R) with N the number of templates and R the number of spots
        contained by the template with the most spots
    theta : 2D np or cp ndarray
        The theta coordinates in the polar image corresponding to template spots.
        shape = (N, R) with N the number of templates and R the number of spots
        contained by the template with the most spots
    intensities :  2D np or cp ndarray
        The intensities corresponding to template spots.
        shape = (N, R) with N the number of templates and R the number of spots
        contained by the template with the most spots
    """
    polar_image = image_to_polar(
        image,
        delta_r,
        delta_theta,
        max_r=max_r,
        find_direct_beam=find_direct_beam,
        direct_beam_position=direct_beam_position,
    )
    dispatcher = get_array_module(polar_image)
    max_radius = polar_image.shape[1] * delta_r
    positions, intensities = _simulations_to_arrays(simulations, max_radius=max_radius)
    r, theta = _cartesian_positions_to_polar(
        positions[:, 0], positions[:, 1], delta_r, delta_theta
    )
    if is_cupy_array(polar_image):
        # send data to GPU
        r = cp.asarray(r)
        theta = cp.asarray(theta)
        intensities = cp.asarray(intensities)
    if intensity_transform_function is not None:
        intensities = intensity_transform_function(intensities)
        polar_image = intensity_transform_function(polar_image)
    if normalize_image:
        polar_image = polar_image / dispatcher.linalg.norm(polar_image)
    if normalize_templates:
        intensities = _norm_rows(intensities)
    return (polar_image, r, theta, intensities)


def _mixed_matching_lib_to_polar(
    polar_image,
    integrated_templates,
    r_templates,
    theta_templates,
    intensities_templates,
    n_keep,
    frac_keep,
    n_best,
    threadsperblock=(16, 16),
):
    """
    Match a polar image to a filtered subset of polar templates

    First does a fast matching basted on azimuthally integrated templates
    Then it takes the (1-fraction)*100% of patterns to do a full indexation on.
    Return the first n_best answers.

    Parameters
    ----------
    polar_image : 2D ndarray
        image in polar coordinates
    integrated_templates : 2D ndarray, (N, r_max)
        azimuthally integrated templates
    r_templates : 2D ndarray, (N, R)
        r coordinates of diffraction spots in all N templates
    theta_templates : 2D ndarray, (N, R)
        theta coordinates of diffraction spots in all N templates
    intensities_templates : 2D ndarray, (N, R)
        intensities of diffraction spots in all N templates
    frac_keep : float
        fraction of templates to pass on to the full indexation
    n_keep : float
        number of templates to pass to the full indexation
    n_best : int
        number of solutions to return in decending order of fit
    threadsperblock : 2-tuple of ints
        threads per block, only relevant for GPU implementation

    Return
    ------
    answer : 2D numpy array, (n_best, 4)
        in the colums are returned (template index, correlation, in-plane angle, factor)
        of the best fitting template, where factor is 1 if the direct template is
        matched and -1 if the mirror template is matched
    """
    dispatcher = get_array_module(polar_image)
    # b
    (
        template_indexes,
        r_templates,
        theta_templates,
        intensities_templates,
    ) = _prefilter_templates(
        polar_image,
        r_templates,
        theta_templates,
        intensities_templates,
        integrated_templates,
        frac_keep,
        n_keep,
    )
    # get a full match on the filtered data - we must branch for CPU/GPU
    (
        best_in_plane_shift,
        best_in_plane_corr,
        best_in_plane_shift_m,
        best_in_plane_corr_m,
    ) = _get_full_correlations(
        polar_image,
        r_templates,
        theta_templates,
        intensities_templates,
        threadsperblock=threadsperblock,
    )
    # compare positive and negative templates and combine
    positive_is_best = best_in_plane_corr >= best_in_plane_corr_m
    negative_is_best = ~positive_is_best
    # multiplication method is faster than dispatcher.choose
    best_sign = positive_is_best * 1 + negative_is_best * (-1)
    best_cors = (
        positive_is_best * best_in_plane_corr + negative_is_best * best_in_plane_corr_m
    )
    best_angles = (
        positive_is_best * best_in_plane_shift + negative_is_best * best_in_plane_shift
    )
    if n_best >= best_cors.shape[0]:
        n_best = best_cors.shape[0]
    if n_best < 1:
        nbest = 1
    answer = dispatcher.empty((n_best, 4), dtype=polar_image.dtype)
    if n_best == 1:
        max_index_filter = dispatcher.argmax(best_cors)
        max_cor = best_cors[max_index_filter]
        max_angle = best_angles[max_index_filter]
        max_index = template_indexes[max_index_filter]
        max_sign = best_sign[max_index_filter]
        answer[0] = dispatcher.array((max_index, max_cor, max_angle, max_sign))
    else:
        # a partial sort
        indices_nbest = dispatcher.argpartition(-best_cors, n_best - 1)[:n_best]
        nbest_cors = best_cors[indices_nbest]
        # a full sort on this subset
        indices_sorted = dispatcher.argsort(-nbest_cors)
        n_best_indices = indices_nbest[indices_sorted]
        answer[:, 0] = template_indexes[n_best_indices]
        answer[:, 1] = best_cors[n_best_indices]
        answer[:, 2] = best_angles[n_best_indices]
        answer[:, 3] = best_sign[n_best_indices]
    return answer


def _index_chunk(
    polar_images,
    integrated_templates,
    r_templates,
    theta_templates,
    intensities_templates,
    n_keep,
    frac_keep,
    n_best,
    norm_images,
    threadsperblock=(16, 16),
):
    dispatcher = get_array_module(polar_images)
    # prepare an empty results chunk
    indexation_result_chunk = dispatcher.empty(
        (polar_images.shape[0], polar_images.shape[1], n_best, 4),
        dtype=polar_images.dtype,
    )
    # norm the images if necessary
    if norm_images:
        pattern_norms = dispatcher.linalg.norm(polar_images, axis=(2, 3))
        polar_images = polar_images / pattern_norms[:, :, np.newaxis, np.newaxis]

    for index in np.ndindex(polar_images.shape[:2]):
        indexation_result_chunk[index] = _mixed_matching_lib_to_polar(
            polar_images[index],
            integrated_templates,
            r_templates,
            theta_templates,
            intensities_templates,
            n_keep,
            frac_keep,
            n_best,
            threadsperblock,
        )
    return indexation_result_chunk


def get_in_plane_rotation_correlation(
    image,
    simulation,
    intensity_transform_function=None,
    delta_r=1,
    delta_theta=1,
    max_r=None,
    find_direct_beam=False,
    direct_beam_position=None,
    normalize_image=True,
    normalize_template=True,
):
    """
    Correlate a single image and simulation over the in-plane rotation angle

    Parameters
    ----------
    image : 2D numpy.ndarray
        The image of the diffraction pattern
    simulation : diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern simulation
    intensity_transform_function : Callable, optional
        Function to apply to both image and template intensities on an
        element by element basis prior to comparison
    delta_r : float, optional
        The sampling interval of the radial coordinate in pixels
    delta_theta : float, optional
        The sampling interval of the azimuthal coordinate in degrees
    max_r : float, optional
        Maximum radius to consider in pixel units. By default it is the
        distance from the center of the image to a corner
    find_direct_beam : bool, optional
        Whether to optimize the direct beam, otherwise the center of the image
        is chosen
    direct_beam_position : 2-tuple of floats, optional
        (x, y) coordinate of the direct beam in pixel units. Overrides other
        settings for finding the direct beam
    normalize_image : bool, optional
        normalize the image to calculate the correlation coefficient
    normalize_template : bool, optional
        normalize the template to calculate the correlation coefficient

    Returns
    -------
    angle_array : 1D np.ndarray
        The in-plane angles at which the correlation is calculated in degrees
    correlation_array : 1D np.ndarray
        The correlation corresponding to these angles
    """
    polar_image = image_to_polar(
        image,
        delta_r,
        delta_theta,
        max_r=max_r,
        find_direct_beam=find_direct_beam,
        direct_beam_position=direct_beam_position,
    )
    r, theta, intensity = get_template_polar_coordinates(
        simulation,
        in_plane_angle=0.0,
        delta_r=delta_r,
        delta_theta=delta_theta,
        max_r=polar_image.shape[1],
    )
    if is_cupy_array(polar_image):
        dispatcher = cp
        r = cp.asarray(r)
        theta = cp.asarray(theta)
        intensity = cp.asarray(intensity)
    else:
        dispatcher = np
    if intensity_transform_function is not None:
        intensity = intensity_transform_function(intensity)
        polar_image = intensity_transform_function(polar_image)
    if normalize_image:
        polar_image = polar_image / dispatcher.linalg.norm(polar_image)
    if normalize_template:
        intensity = intensity / dispatcher.linalg.norm(intensity)
    correlation_array = _match_polar_to_polar_template(
        polar_image,
        r,
        theta,
        intensity,
    )
    angle_array = dispatcher.arange(correlation_array.shape[0]) * delta_theta
    return angle_array, correlation_array


def _get_fast_correlation_index(
    polar_image,
    r,
    intensities,
    normalize_image,
    normalize_templates,
):
    integrated_polar = polar_image.sum(axis=0)
    integrated_templates = _get_integrated_polar_templates(
        integrated_polar.shape[0],
        r,
        intensities,
        normalize_templates,
    )
    if normalize_image:
        integrated_polar = integrated_polar / np.linalg.norm(integrated_polar)
    correlations = _match_library_to_polar_fast(
        integrated_polar,
        integrated_templates,
    )
    return correlations


def correlate_library_to_pattern_fast(
    image,
    simulations,
    delta_r=1,
    delta_theta=1,
    max_r=None,
    intensity_transform_function=None,
    find_direct_beam=False,
    direct_beam_position=None,
    normalize_image=True,
    normalize_templates=True,
):
    """
    Get the correlation between azimuthally integrated templates and patterns

    Parameters
    ----------
    image : 2D numpy.ndarray
        The image of the diffraction pattern
    simulations : list of diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern simulation
    delta_r : float, optional
        The sampling interval of the radial coordinate in pixels
    delta_theta : float, optional
        The sampling interval of the azimuthal coordinate in degrees
    max_r : float, optional
        Maximum radius to consider in pixel units. By default it is the
        distance from the center of the image to a corner
    intensity_transform_function : Callable, optional
        Function to apply to both image and template intensities on an
        element by element basis prior to comparison
    find_direct_beam : bool, optional
        Whether to optimize the direct beam, otherwise the center of the image
        is chosen
    direct_beam_position : 2-tuple of floats, optional
        (x, y) coordinate of the direct beam in pixel units. Overrides other
        settings for finding the direct beam
    normalize_image : bool, optional
        normalize the image to calculate the correlation coefficient
    normalize_templates : bool, optional
        normalize the templates to calculate the correlation coefficient

    Returns
    -------
    correlations : 1D numpy.ndarray
        correlation between azimuthaly integrated template and each azimuthally integrated template

    Notes
    -----
    Mirrored templates have identical azimuthally integrated representations,
    so this only has to be done on the positive euler angle templates (0, Phi, phi2)
    """
    polar_image, r, theta, intensities = _prepare_image_and_templates(
        image,
        simulations,
        delta_r,
        delta_theta,
        max_r,
        intensity_transform_function,
        find_direct_beam,
        direct_beam_position,
        False,  # it is not necessary to normalize these here
        False,
    )
    return _get_fast_correlation_index(
        polar_image, r, intensities, normalize_image, normalize_templates
    )


def _get_max_n(N, n_keep, frac_keep):
    """
    Determine the number of templates to allow through
    """
    max_keep = N
    if frac_keep is not None:
        max_keep = max(round(frac_keep * N), 1)
    # n_keep overrides fraction
    if n_keep is not None:
        max_keep = max(n_keep, 1)
    return int(min(max_keep, N))


def _prefilter_templates(
    polar_image,
    r,
    theta,
    intensities,
    integrated_templates,
    frac_keep,
    n_keep,
):
    dispatcher = get_array_module(polar_image)
    max_keep = _get_max_n(r.shape[0], n_keep, frac_keep)
    template_indexes = dispatcher.arange(r.shape[0], dtype=np.int32)
    if max_keep != r.shape[0]:
        polar_sum = polar_image.sum(axis=0)
        correlations_fast = _match_library_to_polar_fast(
            polar_sum,
            integrated_templates,
        )
        sorted_cor_indx = dispatcher.argsort(-correlations_fast)[:max_keep]
        r = r[sorted_cor_indx]
        theta = theta[sorted_cor_indx]
        intensities = intensities[sorted_cor_indx]
        template_indexes = template_indexes[sorted_cor_indx]
    return template_indexes, r, theta, intensities


def _get_full_correlations(
    polar_image,
    r,
    theta,
    intensities,
    threadsperblock=(16, 16),
):
    # get a full match on the filtered data - we must branch for CPU/GPU
    if is_cupy_array(polar_image):
        blockspergrid_x = int(np.ceil(polar_image.shape[0] / threadsperblock[0]))
        blockspergrid_y = int(np.ceil(polar_image.shape[1] / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        (
            best_in_plane_shift,
            best_in_plane_shift_m,
            best_in_plane_corr,
            best_in_plane_corr_m,
        ) = _match_polar_to_polar_library_gpu(
            polar_image, r, theta, intensities, blockspergrid, threadsperblock
        )
    else:
        (
            best_in_plane_shift,
            best_in_plane_shift_m,
            best_in_plane_corr,
            best_in_plane_corr_m,
        ) = _match_polar_to_library_cpu(
            polar_image,
            r,
            theta,
            intensities,
        )
    return (
        best_in_plane_shift,
        best_in_plane_corr,
        best_in_plane_shift_m,
        best_in_plane_corr_m,
    )


def correlate_library_to_pattern(
    image,
    simulations,
    frac_keep=1.0,
    n_keep=None,
    delta_r=1.0,
    delta_theta=1.0,
    max_r=None,
    intensity_transform_function=None,
    find_direct_beam=False,
    direct_beam_position=None,
    normalize_image=True,
    normalize_templates=True,
    threadsperblock=(16, 16),
):
    """
    Get the best angle and associated correlation values, as well as the correlation with the inverted templates

    Parameters
    ----------
    image : 2D numpy.ndarray
        The image of the diffraction pattern
    simulations : list of diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern simulation
    frac_keep : float
        Fraction (between 0-1) of templates to do a full matching on. By default
        all patterns are fully matched.
    n_keep : int
        Number of templates to do a full matching on. When set frac_keep will be
        ignored
    delta_r : float, optional
        The sampling interval of the radial coordinate in pixels
    delta_theta : float, optional
        The sampling interval of the azimuthal coordinate in degrees
    max_r : float, optional
        Maximum radius to consider in pixel units. By default it is the
        distance from the center of the image to a corner
    intensity_transform_function : Callable, optional
        Function to apply to both image and template intensities on an
        element by element basis prior to comparison
    find_direct_beam : bool, optional
        Whether to optimize the direct beam, otherwise the center of the image
        is chosen
    direct_beam_position : 2-tuple of floats, optional
        (x, y) coordinate of the direct beam in pixel units. Overrides other
        settings for finding the direct beam
    normalize_image : bool, optional
        normalize the image to calculate the correlation coefficient
    normalize_templates : bool, optional
        normalize the templates to calculate the correlation coefficient
    threadsperblock : 2-Tuple of int
        Determines how the code is executed on GPU, only relevant if
        the polar image is a cupy array.

    Returns
    -------
    indexes : 1D numpy.ndarray
        indexes of templates on which a full calculation has been performed
    angles : 1D numpy.ndarray
        best fit in-plane angle for the top "keep" templates
    correlations : 1D numpy.ndarray
        best correlation for the top "keep" templates
    angles_mirrored : 1D numpy.ndarray
        best fit in-plane angle for the top mirrored "keep" templates
    correlations_mirrored : 1D numpy.ndarray
        best correlation for the top mirrored "keep" templates

    Notes
    -----
    Mirrored refers to the templates corresponding to the inverted orientations
    (0, -Phi, -phi/2)
    """
    polar_image, r, theta, intensities = _prepare_image_and_templates(
        image,
        simulations,
        delta_r,
        delta_theta,
        max_r,
        intensity_transform_function,
        find_direct_beam,
        direct_beam_position,
        normalize_image,
        normalize_templates,
    )
    integrated_templates = _get_integrated_polar_templates(
        polar_image.shape[1],
        r,
        intensities,
        normalize_templates,
    )
    indexes, r, theta, intensities = _prefilter_templates(
        polar_image, r, theta, intensities, integrated_templates, frac_keep, n_keep
    )
    angles, cor, angles_m, cor_m = _get_full_correlations(
        polar_image,
        r,
        theta,
        intensities,
        threadsperblock,
    )
    return (
        indexes,
        (angles * delta_theta).astype(polar_image.dtype),
        cor,
        (angles_m * delta_theta).astype(polar_image.dtype),
        cor_m,
    )


def get_n_best_matches(
    image,
    simulations,
    n_best=1,
    frac_keep=1.0,
    n_keep=None,
    delta_r=1.0,
    delta_theta=1.0,
    max_r=None,
    intensity_transform_function=None,
    find_direct_beam=False,
    direct_beam_position=None,
    normalize_image=True,
    normalize_templates=True,
    threadsperblock=(16, 16),
):
    """
    Get the n templates best matching an image in descending order

    Parameters
    ----------
    image : 2D numpy or cupy ndarray
        The image of the diffraction pattern
    simulations : list of diffsims.sims.diffraction_simulation.DiffractionSimulation
        The diffraction pattern simulation
    n_best : int, optional
        Number of best solutions to return, in order of descending match
    n_keep : int, optional
        Number of templates to do a full matching on
    frac_keep : float, optional
        Fraction (between 0-1) of templates to do a full matching on. When set
        n_keep will be ignored
    delta_r : float, optional
        The sampling interval of the radial coordinate in pixels
    delta_theta : float, optional
        The sampling interval of the azimuthal coordinate in degrees
    max_r : float, optional
        Maximum radius to consider in pixel units. By default it is the
        distance from the center of the image to a corner
    intensity_transform_function : Callable, optional
        Function to apply to both image and template intensities on an
        element by element basis prior to comparison
    find_direct_beam : bool, optional
        Whether to optimize the direct beam, otherwise the center of the image
        is chosen
    direct_beam_position : 2-tuple of floats, optional
        (x, y) coordinate of the direct beam in pixel units. Overrides other
        settings for finding the direct beam
    normalize_image : bool, optional
        normalize the image to calculate the correlation coefficient
    normalize_templates : bool, optional
        normalize the templates to calculate the correlation coefficient
    threadsperblock : 2-Tuple of int
        number of threads per block in the cuda matching operation. Only
        important when using a gpu.

    Returns
    -------
    indexes : 1D numpy or cupy ndarray
        indexes of best fit templates
    angles : 1D numpy or cupy ndarray
        corresponding best fit in-plane angles
    correlations : 1D numpy or cupy ndarray
        corresponding correlation values
    signs : 1D numpy or cupy ndarray
        1 if the positive template (0, Phi, phi2) is best matched, -1 if
        the negative template (0, -Phi, -phi2) is best matched
    """
    polar_image, r, theta, intensities = _prepare_image_and_templates(
        image,
        simulations,
        delta_r,
        delta_theta,
        max_r,
        intensity_transform_function,
        find_direct_beam,
        direct_beam_position,
        normalize_image,
        normalize_templates,
    )
    integrated_templates = _get_integrated_polar_templates(
        polar_image.shape[1], r, intensities, normalize_templates
    )
    answer = _mixed_matching_lib_to_polar(
        polar_image,
        integrated_templates,
        r,
        theta,
        intensities,
        n_keep,
        frac_keep,
        n_best,
        threadsperblock,
    )
    indices = answer[:, 0].astype(np.int32)
    cors = answer[:, 1]
    angles = (answer[:, 2] * delta_theta).astype(polar_image.dtype)
    sign = answer[:, 3]
    return indices, angles, cors, sign


def index_dataset_with_template_rotation(
    signal,
    library,
    phases=None,
    n_best=1,
    frac_keep=1.0,
    n_keep=None,
    delta_r=1.0,
    delta_theta=1.0,
    max_r=None,
    intensity_transform_function=None,
    normalize_images=True,
    normalize_templates=True,
    chunks="auto",
    parallel_workers="auto",
    target="cpu",
    threadsperblock=(16, 16),
    scheduler="threads",
    precision=np.float64,
):
    """
    Index a dataset with template_matching while simultaneously optimizing in-plane rotation angle of the templates

    Parameters
    ----------
    signal : hyperspy.signals.Signal2D
        The 4D-STEM dataset.
    library : diffsims.libraries.diffraction_library.DiffractionLibrary
        The library of simulated diffraction patterns.
    phases : list, optional
        Names of phases in the library to do an indexation for. By default this is
        all phases in the library.
    n_best : int, optional
        Number of best solutions to return, in order of descending match.
    frac_keep : float, optional
        Fraction (between 0-1) of templates to do a full matching on. By default
        all templates will be fully matched. See notes for details.
    n_keep : int, optional
        Number of templates to do a full matching on. Overrides frac_keep.
    delta_r : float, optional
        The sampling interval of the radial coordinate in pixels.
    delta_theta : float, optional
        The sampling interval of the azimuthal coordinate in degrees. This will
        determine the maximum accuracy of the in-plane correlation.
    max_r : float, optional
        Maximum radius to consider in pixel units. By default it is the
        distance from the center of the patterns to a corner of the image.
    intensity_transform_function : Callable, optional
        Function to apply to both image and template intensities on an
        element by element basis prior to comparison. Note that when using the
        gpu, the function must be gpu compatible.
    normalize_images : bool, optional
        Normalize the images in the correlation coefficient calculation
    normalize_templates : bool, optional
        Normalize the templates in the correlation coefficient calculation
    chunks : string or 4-tuple, optional
        Internally the work is done on dask arrays and this parameter determines
        the chunking. If set to None then no re-chunking will happen if the dataset
        was loaded lazily. If set to "auto" then dask attempts to find the optimal
        chunk size.
    parallel_workers: int, optional
        The number of workers to use in parallel. If set to "auto", the number
        will be determined from os.cpu_count()
    target: string, optional
        Use "cpu" or "gpu". If "gpu" is selected, the majority of the calculation
        intensive work will be performed on the CUDA enabled GPU. Fails if no
        such hardware is available.
    threadsperblock: 2-tuple of ints
        Only relevant when using GPU, determines how many threads in a block
        the cuda kernel is executed over when indexing patterns
    scheduler: string
        The scheduler used by dask to compute the result
    precision: np.float32 or np.float64
        The level of precision to work with on internal calculations

    Returns
    -------
    result : dict
        Results dictionary containing for each phase a dictionary that contains
        keys [template_index, orientation], each representing numpy arrays of
        shape (scan_x, scan_y, n_best) and (scan_x, scan_y, n_best, 3)
        respectively. Orientation is expressed in Bunge convention
        Euler angles.

    Notes
    -----
    It is possible to run the indexation using a subset of the templates. This
    two-stage procedure is controlled through `n_keep` or `frac_keep`. If one
    of these parameters is set, the azimuthally integrated patterns are
    compared to azimuthally integrated templates in a first stage, which is
    very fast. The top matching patterns are passed to a second stage of
    full matching, whereby the in-plane angle is determined. Setting these
    parameters can usually achieve the same answer faster, but it is also
    possible an incorrect match is found.
    """
    # get the dataset as a dask array
    data = _get_dask_array(signal)
    # check if we have a 4D dataset, and if not, make it
    navdim = signal.axes_manager.navigation_dimension
    if navdim == 0:
        # we assume we have a single image
        data = data[np.newaxis, np.newaxis, ...]
    elif navdim == 1:
        # we assume we have a line of images with the first dimension the line
        data = data[np.newaxis, ...]
    elif navdim == 2:
        # correct dimensions
        pass
    else:
        raise ValueError(f"Dataset has {navdim} navigation dimensions, max " "is 2")
    # change the chunking of the dataset if necessary
    if chunks is None:
        pass
    elif chunks == "auto":
        data = data.rechunk({0: "auto", 1: "auto", 2: None, 3: None})
    else:
        data = data.rechunk(chunks)

    if target == "gpu":
        from pyxem.utils.cuda_utils import dask_array_to_gpu

        # an error will be raised if cupy is not available
        data = dask_array_to_gpu(data)
        dispatcher = cp
    else:
        dispatcher = np

    # convert to polar dataset
    output_shape = get_polar_pattern_shape(
        data.shape[-2:], delta_r, delta_theta, max_r=max_r
    )
    theta_dim, r_dim = output_shape
    max_radius = r_dim * delta_r
    center = (data.shape[-2] / 2, data.shape[-1] / 2)
    polar_chunking = (data.chunks[0], data.chunks[1], theta_dim, r_dim)
    polar_data = data.map_blocks(
        _chunk_to_polar,
        center,
        max_radius,
        output_shape,
        precision,
        dtype=precision,
        drop_axis=signal.axes_manager.signal_indices_in_array,
        chunks=polar_chunking,
        new_axis=(2, 3),
    )
    # apply the intensity transform function to the images
    if intensity_transform_function is not None:
        polar_data = polar_data.map_blocks(intensity_transform_function)
    if phases is None:
        phases = library.keys()

    result = {}

    # calculate number of workers
    if parallel_workers == "auto":
        parallel_workers = os.cpu_count()
    for phase_key in phases:
        phase_library = library[phase_key]
        positions, intensities = _simulations_to_arrays(
            phase_library["simulations"], max_radius
        )
        x = positions[:, 0]
        y = positions[:, 1]
        if intensity_transform_function is not None:
            intensities = intensity_transform_function(intensities)
        r, theta = _cartesian_positions_to_polar(
            x, y, delta_r=delta_r, delta_theta=delta_theta
        )
        # integrated intensity library for fast comparison
        integrated_templates = _get_integrated_polar_templates(
            r_dim, r, intensities, normalize_templates
        )
        # normalize the templates if required
        if normalize_templates:
            integrated_templates = _norm_rows(integrated_templates)
            intensities = _norm_rows(intensities)
        # copy relevant data to GPU memory if necessary
        if target == "gpu":
            integrated_templates = cp.asarray(integrated_templates)
            r = cp.asarray(r)
            theta = cp.asarray(theta)
            intensities = cp.asarray(intensities)

        # put a limit on n_best
        max_n = _get_max_n(r.shape[0], n_keep, frac_keep)
        if n_best > max_n:
            n_best = max_n

        indexation = polar_data.map_blocks(
            _index_chunk,
            integrated_templates,
            r,
            theta,
            intensities,
            n_keep,
            frac_keep,
            n_best,
            normalize_images,
            threadsperblock,
            dtype=precision,
            drop_axis=signal.axes_manager.signal_indices_in_array,
            chunks=(polar_data.chunks[0], polar_data.chunks[1], n_best, 4),
            new_axis=(2, 3),
        )
        # TODO: there is some duplication here as the polar transform is re-calculated for each loop iteration
        # over the phases
        with ProgressBar():
            res_index = indexation.compute(
                scheduler=scheduler, num_workers=parallel_workers, optimize_graph=True
            )
        # in case the data is on the GPU, retrieve it
        res_index = to_numpy(res_index)
        # wrangle data to (template_index), (orientation), (correlation)
        result[phase_key] = {}
        result[phase_key]["template_index"] = res_index[:, :, :, 0].astype(np.int32)
        oris = phase_library["orientations"]
        orimap = oris[res_index[:, :, :, 0].astype(np.int32)]
        orimap[:, :, :, 1] = (
            orimap[:, :, :, 1] * res_index[:, :, :, 3]
        )  # multiply by the sign
        orimap[:, :, :, 2] = (
            orimap[:, :, :, 2] * res_index[:, :, :, 3]
        )  # multiply by the sign
        orimap[:, :, :, 0] = res_index[:, :, :, 2] * delta_theta
        result[phase_key]["orientation"] = orimap
        result[phase_key]["correlation"] = res_index[:, :, :, 1]
        result[phase_key]["mirrored_template"] = res_index[:, :, :, 3] == -1
    return result
