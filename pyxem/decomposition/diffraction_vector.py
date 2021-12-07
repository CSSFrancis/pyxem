from pyxem.decomposition.vector_decomposition import VectorDecomposition2D
import numpy as np
from skimage.draw import disk
from skimage.morphology import flood


def get_vdf(data, vector, threshold=None, return_vectors=True, reduce=True,
            radius=2, extent=None, ):
    shape = tuple(reversed(data.axes_manager.signal_shape))
    rr, cc = disk((vector[-1], vector[2]),
                  radius=radius,
                  shape=shape)
    if extent is None:
        xslice, yslice = (slice(None), slice(None))

    vdf = np.sum(data.data[xslice, yslice, rr, cc], axis=(2))

    if threshold is not None:
        center = vdf[int(vector[0]), int(vector[1])]
        minimum = np.min(vdf)
        difference = center - minimum
        thresh = minimum + threshold * difference
        vectors = vdf > thresh

    if reduce:
        vectors = flood(vectors, seed_point=(int(vector[0]), int(vector[1])))

    if return_vectors:
        return [np.concatenate([v, vector[2:]]) for v in np.argwhere(vectors)]

    return vectors



class DiffractionVector(VectorDecomposition2D):

    def get_extents(self,
                    data,
                    trim_duplicates=True,
                    threshold=0.9,
                    **kwargs):
        new_vectors = []
        while self.data.array:
            new_vectors = get_vdf(data,
                                  vector=self.data.array.pop(0),
                                  threshold=threshold,
                                  reduce=True,
                                  return_vectors=True,
                                  **kwargs,)
            if trim_duplicates:
                self.data.array