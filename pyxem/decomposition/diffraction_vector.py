from pyxem.decomposition.vector_decomposition import VectorDecomposition2D
import numpy as np
from skimage.draw import disk
from skimage.morphology import flood


def get_vdf(data, vector, threshold=None, return_vectors=True, reduce=True,
            radius=2, extent=None, ):
    shape = tuple(reversed(data.axes_manager.signal_shape))
    rr, cc = disk(center=(vector[-2], vector[-3]),
                  radius=radius,
                  shape=shape)
    if extent is None:
        xslice, yslice = (slice(None), slice(None))

    vdf = np.sum(data.data[xslice, yslice, rr, cc], axis=(2))

    if threshold is not None:
        max = np.max(vdf)
        minimum = np.min(vdf)
        difference = max - minimum
        thresh = minimum + (threshold * difference)
        vectors = vdf > thresh

    if reduce:
        # Yeah so this doens't work if the seed point is outside of the actual extent...
        vectors = flood(vectors, seed_point=(int(vector[1]), int(vector[0])))

    if return_vectors:
        return [np.concatenate([v, vector[2:]]) for v in np.argwhere(vectors)]

    return vectors


class DiffractionVector(VectorDecomposition2D):

    def get_extents(self,
                    data,
                    threshold=0.9,
                    **kwargs):
        new_vectors = []
        labels = np.arange(0, len(self.data.array))
        for v, l in zip(self.data.array, labels):
            new_vector = get_vdf(data,
                                 vector=v,
                                 threshold=threshold,
                                 reduce=True,
                                 return_vectors=True,
                                 **kwargs,
                                 )
            self.data.array = np.concatenate([self.data.array, new_vector])
            labels = np.concatenate([labels, np.repeat(l, len(new_vector))])
        self.labels = labels
        return
