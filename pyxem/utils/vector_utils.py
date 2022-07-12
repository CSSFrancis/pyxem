# -*- coding: utf-8 -*-
# Copyright 2016-2022 The pyXem developers
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

import numpy as np
import dask.array as da
import math
from itertools import product

from transforms3d.axangles import axangle2mat
from skimage.morphology import convex_hull_image,flood
from skimage.draw import disk
from skimage.metrics import structural_similarity
from scipy.ndimage import center_of_mass
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance_matrix


def trim_duplicates(vectors, label):
    return np.squeeze(vectors)[label != -1]


def center_and_crop(image, center):
    extent = np.argwhere(image > 0)
    if len(extent) == 0:
        return [], tuple([slice(None) for c in center])
    else:
        rounded_center = np.array(np.round(center), dtype=int)
        max_values = np.max(extent, axis=0)-rounded_center
        min_values = rounded_center-np.min(extent, axis=0)
        max_extent = np.max([max_values, min_values])
        slices = tuple([slice(c-max_extent, c+max_extent+1) for c in rounded_center])
        return image[slices]


def get_chunk_offsets(img):
    """Returns the extent of some chunk.

     For determining if a vector is inside of the chunk.
     """
    offset = []
    for block_id in product(*(range(len(c)) for c in img.chunks)):
        offset.append(np.transpose([np.multiply(block_id, img.chunksize),
                                    np.multiply(np.add(block_id, 1), img.chunksize)]))
    offset = np.array(offset, dtype=object)
    offset = np.reshape(offset, [len(c) for c in img.chunks] + [4, 2])  # this might fail eventually
    offset = np.squeeze(offset)
    if np.shape(offset) == (4, 2):
        offset = np.reshape(offset, (1, 4, 2))
    return offset


def get_vectors_chunkwise(vectors, offsets, extra_vectors=None):
    chunks = offsets.shape[:-2]
    if chunks == ():
        chunks = (1,)
    vectors = vectors[0]
    new_vectors = np.empty(shape=chunks, dtype=object)
    if extra_vectors is not None:
        extra = [np.empty(shape=chunks, dtype=object) for v in extra_vectors]
    for i in np.ndindex(chunks):
        sliced_offsets = offsets[i]
        vector_in_block = np.prod([np.greater(vectors[:, i], s[0]) &
                                   np.less(vectors[:, i], s[1]) for i, s in enumerate(sliced_offsets)], axis=0,
                                  dtype=bool)
        new_vectors[i] = vectors[vector_in_block]
        if extra_vectors is not None:
            for j, e in enumerate(extra_vectors):
                extra[j][i] = e[vector_in_block]
    if extra_vectors is not None:
        return da.from_array(new_vectors, chunks=1), da.from_array(extra, chunks=1)
    else:
        return da.from_array(new_vectors, chunks=1)


"""
Functions for refining the position of vectors
"""


def refine(data, vectors, extents, offset=None, threshold=0.7):
    ind = (0,)*len(vectors.shape)
    vectors = vectors[ind]
    if offset is None:
        offset = np.zeros(vectors.shape[1])
    else:
        offset = np.squeeze(offset)[:, 0]
    refined = []
    for v, e in zip(vectors, extents[ind]):
        shifted_vector = v - offset
        ref = refine_position(shifted_vector, data, extent=e, threshold=threshold)
        ref = np.add(offset, ref)
        refined.append(ref)
    if len(refined) == 0:
        return np.empty(1, dtype=object)
    ref_data = np.empty(1, dtype=object)
    ref_data[0] = np.array(refined, dtype=object)
    return ref_data


def refine_position(vector, data, extent, threshold=0.5):
    extent = np.array(extent, dtype=float)
    mask = extent > 0
    real_pos = center_of_mass(mask)
    mean_image = np.mean(data[mask, :, :], axis=0)
    max_val = mean_image[int(vector[2]), int(vector[3])]
    abs_threshold = max_val * threshold
    threshold_image = mean_image > abs_threshold
    ex = flood(threshold_image, seed_point=(int(vector[2]), int(vector[3])))
    recip_pos = center_of_mass(ex)
    new_vector = list(tuple(real_pos) + tuple(recip_pos))
    if any(np.isnan(new_vector)):
        new_vector = vector
    return new_vector


"""
Functions for getting the VDF image from some dataset
"""


def get_extents(img, vectors, offset=None, **kwargs):
    ind = (0,)*len(vectors.shape)
    vectors = vectors[ind]
    if offset is None:
        offset = np.zeros(vectors.shape[1])
    else:
        offset = np.squeeze(offset)[:, 0]
    vectors = vectors - offset
    extents = np.array([_get_vdf(v,
                                 img,
                                 **kwargs) for v in vectors])

    if len(extents) == 0:
        return np.empty(1, dtype=object)
    ext_data = np.empty(1, dtype=object)
    ext_data[0] = np.array(extents, dtype=object)
    return ext_data


from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import linkage, fcluster


def combine_vectors(vectors,
                    distance,
                    duplicate_distance=None,
                    include_k=True,
                    extents=None,
                    ss=None,
                    ):
    import itertools
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=distance)
    if include_k:
        agg.fit(vectors[:, :3])
    else:
        agg.fit(vectors[:, :2])
    labels = agg.labels_
    duplicates = np.zeros(len(labels), dtype=bool)
    for l in range(max(labels + 1)):
        grouped_vectors = vectors[labels == l]
        if duplicate_distance is not None:
            dup = distance_matrix(grouped_vectors[:, 2:], grouped_vectors[:, 2:]) < duplicate_distance
            not_first = np.sum(np.tril(dup), axis=1) > 1
            duplicates[labels == l] = not_first
    labels = np.array(labels)
    labels[duplicates] = -1

    max_l = max(labels) + 1

    if ss is not None:
        if extents is None:
            raise ValueError("extents must be passed if structural similarity is compared")
        else:
            for l in range(max(labels + 1)):
                grouped_extents = extents[labels == l]
                if len(grouped_extents) == 1:
                    continue
                grouped_extents = [np.array(ex, dtype=float) for ex in grouped_extents]

                comb = itertools.combinations(grouped_extents, 2)
                indexes = np.argwhere(labels == l)
                index_combo = itertools.combinations(indexes, 2)
                sim = np.array([crop_ss(c[0], c[1]) for c in comb])
                sim[sim < 0] = 0
                above_ss = np.array(list(index_combo))[sim > ss]

                above_ss = above_ss[..., 0]
                new_groups = merge_it(above_ss)
                if len(new_groups) > 1:
                    for n in new_groups[1:]:
                        for i in n:
                            labels[i] = max_l
                            max_l += 1
                for i in indexes:
                    if i[0] not in set().union(*new_groups):
                        labels[i] = max_l
                        max_l += 1
    return labels


def merge_it(lot):
    import itertools
    merged = [set(x) for x in lot]  # operate on sets only
    finished = False
    while not finished:
        finished = True
        for a, b in itertools.combinations(merged, 2):
            if a & b:
                # we merged in this iteration, we may have to do one more
                finished = False
                if a in merged: merged.remove(a)
                if b in merged: merged.remove(b)
                merged.append(a.union(b))
                break  # don't inflate 'merged' with intermediate results
    return merged


def crop_extent(ex1, ex2):
    try:
        non_zero = np.where(ex1 > 0)
        non_zero2 = np.where(ex2 > 0)

        min1 = np.min(non_zero, axis=1)
        max1 = np.max(non_zero, axis=1)

        min2 = np.min(non_zero2, axis=1)
        max2 = np.max(non_zero2, axis=1)

        mins = [np.min([min1[0], min2[0]]), np.min([min1[1], min2[1]])]
        maxes = [np.max([max1[0], max2[0]]), np.max([max1[1], max2[1]])]
        return ex1[mins[0]:maxes[0], mins[1]:maxes[1]], ex2[mins[0]:maxes[0], mins[1]:maxes[1]]
    except ValueError:
        return ex1, ex2


def crop_ss(im1, im2, **kwargs):
    e1, e2 = crop_extent(im1, im2)

    try:
        ss = structural_similarity(e1, e2, **kwargs)
        return ss
    except ValueError:
        return 0


def _get_vdf(vector,
             img,
             threshold=None,
             radius=2,
             fill=True,
             crop=True,
             ):
    shape = img.shape[-2:]
    rr, cc = disk(center=(int(vector[-2]), int(vector[-1])),
                  radius=int(radius),
                  shape=shape)
    nav_dims = len(np.shape(img))-2
    vdf = np.sum(np.squeeze(img)[..., rr, cc], axis=-1)
    if threshold is not None:
        center = np.array(vector[:nav_dims], dtype=int)
        maximum = vdf[tuple(center)]
        minimum = np.mean(vdf)
        difference = maximum - minimum
        thresh = minimum + threshold * difference
        mask = vdf > thresh
        mask = flood(mask, seed_point=tuple(center))
        if fill is True and len(mask.shape) > 1:
            mask = convex_hull_image(mask)
        if np.sum(mask) > (np.product(vdf.shape) / 2):
            vdf = np.zeros(vdf.shape)
        else:
            vdf[np.logical_not(mask)] = 0

    return vdf


def detector_to_fourier(k_xy, wavelength, camera_length):
    """Maps two-dimensional Cartesian coordinates in the detector plane to
    three-dimensional coordinates in reciprocal space, with origo in [000].

    The detector uses a left-handed coordinate system, while the reciprocal
    space uses a right-handed coordinate system.

    Parameters
    ----------
    k_xy : np.array()
        Cartesian coordinates in detector plane, in reciprocal Ångström.
    wavelength : float
        Electron wavelength in Ångström.
    camera_length : float
        Camera length in metres.

    Returns
    -------
    k : np.array()
        Array of Cartesian coordinates in reciprocal space relative to [000].

    """

    if k_xy.shape == (1,) and k_xy.dtype == "object":
        # From ragged array
        k_xy = k_xy

    # The calibrated positions of the diffraction spots are already the x and y
    # coordinates of the k vector on the Ewald sphere. The radius is given by
    # the wavelength. k_z is calculated courtesy of Pythagoras, then offset by
    # the Ewald sphere radius.

    k_z = np.sqrt(1 / (wavelength ** 2) - np.sum(k_xy ** 2, axis=1)) - 1 / wavelength

    # Stack the xy-vector and the z vector to get the full k
    k = np.hstack((k_xy, k_z[:, np.newaxis]))
    return k


def calculate_norms(z):
    """Calculates the norm of an array of cartesian vectors. For use with map().

    Parameters
    ----------
    z : np.array()
        Array of cartesian vectors.

    Returns
    -------
    norms : np.array()
        Array of vector norms.
    """
    return np.linalg.norm(z, axis=1 )


def calculate_norms_ragged(z):
    """Calculates the norm of an array of cartesian vectors. For use with map()
    when applied to a ragged array.

    Parameters
    ----------
    z : np.array()
        Array of cartesian vectors.

    Returns
    -------
    norms : np.array()
        Array of vector norms.
    """
    norms = []
    for i in z:
        norms.append(np.linalg.norm(i))
    return np.asarray(norms)


def filter_vectors_ragged(z, min_magnitude, max_magnitude):
    """Filters the diffraction vectors to accept only those with magnitudes
    within a user specified range.

    Parameters
    ----------
    min_magnitude : float
        Minimum allowed vector magnitude.
    max_magnitude : float
        Maximum allowed vector magnitude.

    Returns
    -------
    filtered_vectors : np.array()
        Diffraction vectors within allowed magnitude tolerances.
    """
    # Calculate norms
    norms = []
    for i in z:
        norms.append(np.linalg.norm(i))
    norms = np.asarray(norms)
    # Filter based on norms
    norms[norms < min_magnitude] = 0
    norms[norms > max_magnitude] = 0
    filtered_vectors = z[np.where(norms)]

    return filtered_vectors


def filter_vectors_edge_ragged(z, x_threshold, y_threshold):
    """Filters the diffraction vectors to accept only those not within a user
    specified proximity to detector edge.

    Parameters
    ----------
    x_threshold : float
        Maximum x-coordinate in calibrated units.
    y_threshold : float
        Maximum y-coordinate in calibrated units.

    Returns
    -------
    filtered_vectors : np.array()
        Diffraction vectors within allowed tolerances.
    """
    # Filter x / y coordinates
    z[np.absolute(z.T[0]) > x_threshold] = 0
    z[np.absolute(z.T[1]) > y_threshold] = 0
    filtered_vectors = z[np.where(z.T[0])]

    return filtered_vectors


def normalize_or_zero(v):
    """Normalize `v`, or return the vector directly if it has zero length.

    Parameters
    ----------
    v : np.array()
        Single vector or array of vectors to be normalized.
    """
    norms = np.linalg.norm(v, axis=-1)
    nonzero_mask = norms > 0
    if np.any(nonzero_mask):
        v[nonzero_mask] /= norms[nonzero_mask].reshape(-1, 1)


def get_rotation_matrix_between_vectors(from_v1, from_v2, to_v1, to_v2):
    """Calculates the rotation matrix from one pair of vectors to the other.
    Handles multiple to-vectors from a single from-vector.

    Find `R` such that `v_to = R @ v_from`.

    Parameters
    ----------
    from_v1, from_v2 : np.array()
        Vector to rotate _from_.
    to_v1, to_v2 : np.array()
        Nx3 array of vectors to rotate _to_.

    Returns
    -------
    R : np.array()
        Nx3x3 list of rotation matrices between the vector pairs.
    """
    # Find normals to rotate around
    plane_normal_from = np.cross(from_v1, from_v2, axis=-1)
    plane_normal_to = np.cross(to_v1, to_v2, axis=-1)
    plane_common_axes = np.cross(plane_normal_from, plane_normal_to, axis=-1)

    # Try to remove normals from degenerate to-planes by replacing them with
    # the rotation axes between from and to vectors.
    to_degenerate = np.isclose(np.sum(np.abs(plane_normal_to), axis=-1), 0.0)
    plane_normal_to[to_degenerate] = np.cross(from_v1, to_v1[to_degenerate], axis=-1)
    to_degenerate = np.isclose(np.sum(np.abs(plane_normal_to), axis=-1), 0.0)
    plane_normal_to[to_degenerate] = np.cross(from_v2, to_v2[to_degenerate], axis=-1)

    # Normalize the axes used for rotation
    normalize_or_zero(plane_normal_to)
    normalize_or_zero(plane_common_axes)

    # Create rotation from-plane -> to-plane
    common_valid = ~np.isclose(np.sum(np.abs(plane_common_axes), axis=-1), 0.0)
    angles = get_angle_cartesian_vec(
        np.broadcast_to(plane_normal_from, plane_normal_to.shape), plane_normal_to
    )
    R1 = np.empty((angles.shape[0], 3, 3))
    if np.any(common_valid):
        R1[common_valid] = np.array(
            [
                axangle2mat(axis, angle, is_normalized=True)
                for axis, angle in zip(
                    plane_common_axes[common_valid], angles[common_valid]
                )
            ]
        )
    R1[~common_valid] = np.identity(3)

    # Rotate from-plane into to-plane
    rot_from_v1 = np.matmul(R1, from_v1)
    rot_from_v2 = np.matmul(R1, from_v2)

    # Create rotation in the now common plane

    # Find the average angle
    angle1 = get_angle_cartesian_vec(rot_from_v1, to_v1)
    angle2 = get_angle_cartesian_vec(rot_from_v2, to_v2)
    angles = 0.5 * (angle1 + angle2)
    # Negate angles where the rotation where the rotation axis points the
    # opposite way of the to-plane normal. Einsum gives list of dot
    # products.
    neg_angle_mask = (
        np.einsum("ij,ij->i", np.cross(rot_from_v1, to_v1, axis=-1), plane_normal_to)
        < 0
    )
    np.negative(angles, out=angles, where=neg_angle_mask)

    # To-plane normal still the same
    R2 = np.array(
        [
            axangle2mat(axis, angle, is_normalized=True)
            for axis, angle in zip(plane_normal_to, angles)
        ]
    )

    # Total rotation is the combination of to plane R1 and in plane R2
    R = np.matmul(R2, R1)

    return R


def get_npeaks(found_peaks):
    """Returns the number of entries in a list. For use with map().

    Parameters
    ----------
    found_peaks : np.array()
        Array of found peaks.

    Returns
    -------
    len : int
        The number of peaks in the array.
    """
    return len(found_peaks)


def get_angle_cartesian_vec(a, b):
    """Compute the angles between two lists of vectors in a cartesian
    coordinate system.

    Parameters
    ----------
    a, b : np.array()
        The two lists of directions to compute the angle between in Nx3 float
        arrays.

    Returns
    -------
    angles : np.array()
        List of angles between `a` and `b` in radians.
    """
    if a.shape != b.shape:
        raise ValueError(
            "The shape of a {} and b {} must be the same.".format(a.shape, b.shape)
        )

    denom = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    denom_nonzero = denom != 0.0
    angles = np.zeros(a.shape[0])
    angles[denom_nonzero] = np.arccos(
        np.clip(
            np.sum(a[denom_nonzero] * b[denom_nonzero], axis=-1) / denom[denom_nonzero],
            -1.0,
            1.0,
        )
    ).ravel()
    return angles


def get_angle_cartesian(a, b):
    """Compute the angle between two vectors in a cartesian coordinate system.

    Parameters
    ----------
    a, b : array-like with 3 floats
        The two directions to compute the angle between.

    Returns
    -------
    angle : float
        Angle between `a` and `b` in radians.
    """
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return math.acos(max(-1.0, min(1.0, np.dot(a, b) / denom)))
