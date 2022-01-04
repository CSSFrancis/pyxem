from pyxem.decomposition.vector_decomposition import VectorDecomposition2D
import numpy as np
from skimage.draw import disk
from skimage.morphology import flood
from sklearn.cluster import AgglomerativeClustering
from scipy.ndimage import center_of_mass
from scipy.spatial import distance_matrix


def get_vdf(data,
            vector,
            threshold=None,
            radius=2,
            search=5,
            extent=None,
            ):
    shape = tuple(reversed(data.axes_manager.signal_shape))
    rr, cc = disk(center=(vector[-2], vector[-1]),
                  radius=radius,
                  shape=shape)
    if extent is None:
        xslice, yslice = (slice(None), slice(None))

    vdf = np.sum(data.data[xslice, yslice, rr, cc], axis=(2))

    if threshold is not None:
        start0 = int(vector[0] - search)
        start1 = int(vector[1] - search)
        stop0 = int(vector[0] + search)
        stop1 = int(vector[1] + search)
        if start0 < 0:
            start0 = 0
        if start1 < 0:
            start1 = 0
        if stop0 > vdf.shape[0]:
            stop0 = vdf.shape[0] - 1
        if stop1 > vdf.shape[1]:
            stop1 = vdf.shape[1] - 1
        sl0 = slice(start0, stop0)
        sl1 = slice(start1, stop1)

        change = np.unravel_index(np.argmax(vdf[sl0, sl1]),
                                  (stop0 - start0, stop1 - start1))
        center = np.array([start0, start1]) + change
        maximum = vdf[int(center[0]), int(center[1])]
        minimum = np.min(vdf)

        difference = maximum - minimum
        thresh = minimum + threshold * difference
        mask = vdf > thresh
        mask = flood(mask, seed_point=(int(vector[0]), int(vector[1])))
        if np.sum(mask) > (np.product(vdf.shape) / 2):
            vdf = np.zeros(vdf.shape)
        else:
            vdf[np.logical_not(mask)] = 0
        center = center_of_mass(vdf)
    else:
        center = (vector[0], vector[1])
    return center, vdf


def refine_reciporical_position(data, mask, vector, threshold=0.5):
    mean_image = np.mean(data[mask, :, :], axis=(0))
    max_val = mean_image[int(vector[2]), int(vector[3])]
    abs_threshold = max_val*threshold
    threshold_image = mean_image > abs_threshold
    ex = flood(threshold_image, seed_point=(int(vector[2]), int(vector[3])))
    center = center_of_mass(ex)
    return center, ex


def center_and_crop(image, center):
    extent = np.argwhere(image > 0)
    if len(extent) == 0:
        return [], tuple([slice() for c in center])
    else:
        rounded_center = np.array(np.round(center), dtype=int)
        max_values = np.max(extent, axis=0)-rounded_center
        min_values = rounded_center-np.min(extent, axis=0)
        max_extent = np.max([max_values, min_values])
        slices = tuple([slice(c-max_extent, c+max_extent+1) for c in rounded_center])
        return image[slices], slices


class DiffractionVector(VectorDecomposition2D):
    """
    Diffraction Vector Object.  A collection of diffraction vectors.
    Extends the vector decomposition class
    """

    def get_extents(self,
                    data,
                    threshold=0.9,
                    inplace=False,
                    crop=True,
                    **kwargs):
        """Get the extent of each diffraction vector using some dataset.  Finds the extent given
        some seed point and a threshold.

        Parameters
        ----------
        data: Array-like
            An n-dimensional array to use to determine the extent of the vector.
            Should have the same dimensions as one of the diffraction vectors.
        threshold: float
            The relative threshold to use to determine the extent of some feature.
        inplace: bool
            Return a copy of the data or act on the dataset
        crop: bool
            Crop the vdf to only be the extent of the vector.  Saves on memory if you have a large
            number of vectors.
        **kwargs:
            radius: The radius of to use to create the vdf
            search: The area around the center point to look to higher values.
        """
        if inplace:
            vectors = self
        else:
            vectors = self._deepcopy_with_new_data(new_data=np.copy(
                self.data.array))
        for i, v in enumerate(vectors.vectors):
            center, vdf = get_vdf(data,
                                  v,
                                  threshold=threshold,
                                  **kwargs,)
            if crop:
                vdf, slices = center_and_crop(vdf, center)
                vectors.slices[i] = slices
                vectors.cropped = True

            vectors.extents[i] = vdf
            if not any(np.isnan(center)):
                vectors.vectors[i][:2] = center

        return vectors

    def refine_positions(self,
                         data,
                         threshold=0.5,
                         inplace=False,
                         ):
        """Refine the position of the diffraction vector the signal space

        Parameters
        ----------
        data: Array-like
            An n-dimensional array to use to determine the extent of the vector.
            Should have the same dimensions as one of the diffraction vectors.
        threshold: float
            The relative threshold to use to determine the extent of some feature.
        inplace: bool
            Return a copy of the data or act on the dataset
        crop: bool
            Crop the vdf to only be the extent of the vector.  Saves on memory if you have a large
            number of vectors.
        """
        if inplace:
            vectors = self
        else:
            vectors = self._deepcopy_with_new_data(new_data=np.copy(self.data.array))
        for i, (v, e) in enumerate(zip(vectors.vectors, vectors.extents)):
            if self.cropped:
                d = data[self.slices[i]]
            else:
                d = data
            if len(e) != 0:
                center, ex = refine_reciporical_position(d,
                                                         e > 0,
                                                         v,
                                                         threshold=threshold)
                vectors.vectors[i][2:] = center
        return vectors

    def combine_vectors(self,
                        distance,
                        remove_duplicates=True,
                        symmetries=None,
                        structural_similarity=False,
                        inplace=False
                        ):
        if inplace:
            vectors = self
        else:
            vectors = self._deepcopy_with_new_data(new_data=np.copy(
                self.data.array))
        agg = AgglomerativeClustering(n_clusters=None, distance_threshold=distance)
        agg.fit(vectors.vectors[:, :2])
        labels = agg.labels_
        new_vectors = []
        new_labels = []
        new_extents = []
        vectors.extents = np.array(vectors.extents)
        for l in range(max(labels)):
            grouped_vectors = vectors.vectors[labels == l]
            extents = vectors.extents[labels == l]
            if remove_duplicates:
                dist_mat = distance_matrix(grouped_vectors[:, 2:], grouped_vectors[:, 2:]) < 5
                is_first = np.sum(np.tril(dist_mat), axis=1) == 1
                grouped_vectors = grouped_vectors[is_first]
                extents = extents[is_first]
            for v, e in zip(grouped_vectors, extents):
                new_vectors.append(v)
                new_labels.append(l)
                new_extents.append(e)

        vectors.data.array = np.array(new_vectors)
        vectors.labels = np.array(new_labels)
        vectors.extents = np.array(new_extents)

        return vectors

