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
            extent=None, ):
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


class DiffractionVector(VectorDecomposition2D):

    def get_extents(self,
                    data,
                    trim_duplicates=True,
                    threshold=0.9,
                    inplace=False,
                    **kwargs):
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
            vectors.extents.append(vdf)
            vectors.vectors[i][:2] = center
        return vectors

    def combine_vectors(self,
                        distance,
                        remove_duplicates=True,
                        symmetries=None,
                        ):
        agg = AgglomerativeClustering(n_clusters=None,
                                      distance_threshold=distance)
        agg.fit(self.vectors[:, :2])
        labels = agg.labels_
        new_vectors = []
        new_labels = []
        for l in range(max(labels)):
            grouped_vectors = self.vectors[labels == l]
            if remove_duplicates:
                dist_mat = distance_matrix(grouped_vectors,
                                           grouped_vectors) < 5
                grouped_vectors = grouped_vectors[
                    np.sum(np.tril(dist_mat), axis=1) == 1]
            for v in grouped_vectors:
                new_vectors.append(v)
                new_labels.append(l)
        return new_vectors, labels