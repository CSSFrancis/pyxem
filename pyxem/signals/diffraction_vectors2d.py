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

import numpy as np
from warnings import warn


from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from hyperspy.signals import BaseVectorSignal

from pyxem.utils.vector_utils import (
    detector_to_fourier,
    calculate_norms,
    calculate_norms_ragged,
    get_npeaks,
    filter_vectors_ragged,
    filter_vectors_edge_ragged,
)
from pyxem.utils.signal import (
    transfer_navigation_axes,
    transfer_navigation_axes_to_signal_axes,
)


class DiffractionVectors2D(BaseVectorSignal):
    """Crystallographic mapping results containing the best matching crystal
    phase and orientation at each navigation position with associated metrics.

    """
    _signal_dimension = 2
    _signal_type = "diffraction_vectors"

    def plot_on_signal(self):
        return

    def get_magnitudes(self, *args, **kwargs):
        """Calculate the magnitude of diffraction vectors.

        Parameters
        ----------
        *args:
            Arguments to be passed to map().
        **kwargs:
            Keyword arguments to map().

        Returns
        -------
        magnitudes : BaseSignal
            A signal with navigation dimensions as the original diffraction
            vectors containing an array of gvector magnitudes at each
            navigation position.

        """
        magnitudes = self.map(calculate_norms,
                              inplace=False,
                              *args, **kwargs)

        return magnitudes

    def get_unique_vectors(
        self,
        distance_threshold=0.01,
        method="distance_comparison",
        min_samples=1,
        return_clusters=False,
    ):
        """Returns diffraction vectors considered unique by:
        strict comparison, distance comparison with a specified
        threshold, or by clustering using DBSCAN [1].

        Parameters
        ----------
        distance_threshold : float
            The minimum distance between diffraction vectors for them to
            be considered unique diffraction vectors. If
            distance_threshold==0, the unique vectors will be determined
            by strict comparison.
        method : str
            The method to use to determine unique vectors. Valid methods
            are 'strict', 'distance_comparison' and 'DBSCAN'.
            'strict' returns all vectors that are strictly unique and
            corresponds to distance_threshold=0.
            'distance_comparison' checks the distance between vectors to
            determine if some should belong to the same unique vector,
            and if so, the unique vector is iteratively updated to the
            average value.
            'DBSCAN' relies on the DBSCAN [1] clustering algorithm, and
            uses the Eucledian distance metric.
        min_samples : int, optional
            The minimum number of not strictly identical vectors within
            one cluster for the cluster to be considered a core sample,
            i.e. to not be considered noise. Only used for method='DBSCAN'.
        return_clusters : bool, optional
            If True (False is default), the DBSCAN clustering result is
            returned. Only used for method='DBSCAN'.

        References
        ----------
        [1] https://scikit-learn.org/stable/modules/generated/sklearn.
            cluster.DBSCAN.html

        Returns
        -------
        unique_peaks : DiffractionVectors
            The unique diffraction vectors.
        clusters : DBSCAN
            The results from the clustering, given as class DBSCAN.
            Only returned if method='DBSCAN' and return_clusters=True.
        """
        # Flatten the array of peaks to reach dimension (n, 2), where n
        # is the number of peaks.
        peaks_all = np.concatenate([peaks.ravel() for peaks in self.data.flat]).reshape(
            -1, 2
        )

        # A distance_threshold of 0 implies a strict comparison. So in that
        # case, a warning is raised unless the specified method is 'strict'.
        if distance_threshold == 0:
            if method != "strict":
                warn(
                    "distance_threshold=0 was given, and therefore "
                    "a strict comparison is used, even though the "
                    "specified method was {}".format(method)
                )
                method = "strict"

        if method == "strict":
            unique_peaks = np.unique(peaks_all, axis=0)

        elif method == "distance_comparison":
            unique_vectors, unique_counts = np.unique(
                peaks_all, axis=0, return_counts=True
            )

            unique_peaks = np.array([[0, 0]])
            unique_peaks_counts = np.array([0])

            while unique_vectors.shape[0] > 0:
                unique_vector = unique_vectors[0]
                distances = distance_matrix(np.array([unique_vector]), unique_vectors)
                indices = np.where(distances < distance_threshold)[1]

                new_count = indices.size
                new_unique_peak = np.array(
                    [
                        np.average(
                            unique_vectors[indices],
                            weights=unique_counts[indices],
                            axis=0,
                        )
                    ]
                )

                unique_peaks = np.append(unique_peaks, new_unique_peak, axis=0)

                unique_peaks_counts = np.append(unique_peaks_counts, new_count)
                unique_vectors = np.delete(unique_vectors, indices, axis=0)
                unique_counts = np.delete(unique_counts, indices, axis=0)
            unique_peaks = np.delete(unique_peaks, [0], axis=0)

        elif method == "DBSCAN":
            # All peaks are clustered by DBSCAN so that peaks within
            # one cluster are separated by distance_threshold or less.
            unique_vectors, unique_vectors_counts = np.unique(
                peaks_all, axis=0, return_counts=True
            )
            clusters = DBSCAN(
                eps=distance_threshold, min_samples=min_samples, metric="euclidean"
            ).fit(unique_vectors, sample_weight=unique_vectors_counts)
            unique_labels, unique_labels_count = np.unique(
                clusters.labels_, return_counts=True
            )
            unique_peaks = np.zeros((unique_labels.max() + 1, 2))

            # For each cluster, a center of mass is calculated based
            # on all the peaks within the cluster, and the center of
            # mass is taken as the final unique vector position.
            for n in np.arange(unique_labels.max() + 1):
                peaks_n_temp = unique_vectors[clusters.labels_ == n]
                peaks_n_counts_temp = unique_vectors_counts[clusters.labels_ == n]
                unique_peaks[n] = np.average(
                    peaks_n_temp, weights=peaks_n_counts_temp, axis=0
                )

        # Manipulate into DiffractionVectors class
        if unique_peaks.size > 0:
            unique_peaks = DiffractionVectors2D(unique_peaks)
            unique_peaks.axes_manager.set_signal_dimension(1)
        if return_clusters and method == "DBSCAN":
            return unique_peaks, clusters
        else:
            return unique_peaks

    def get_diffracting_pixels_map(self, in_range=None, binary=False):
        """Map of the number of vectors at each navigation position.

        Parameters
        ----------
        in_range : tuple
            Tuple (min_magnitude, max_magnitude) the minimum and maximum
            magnitude of vectors to be used to form the map.
        binary : boolean
            If True a binary image with diffracting pixels taking value == 1 is
            returned.

        Returns
        -------
        crystim : Signal2D
            2D map of diffracting pixels.
        """
        if in_range:
            filtered = self.filter_magnitude(in_range[0], in_range[1])
            xim = filtered.map(get_npeaks, inplace=False).as_signal2D((0, 1))
        else:
            xim = self.map(get_npeaks, inplace=False).as_signal2D((0, 1))
        # Make binary if specified
        if binary is True:
            xim = xim >= 1.0
        # Set properties
        xim = transfer_navigation_axes_to_signal_axes(xim, self)
        xim.change_dtype("float")
        xim.set_signal_type("signal2d")
        xim.metadata.General.title = "Diffracting Pixels Map"

        return xim
