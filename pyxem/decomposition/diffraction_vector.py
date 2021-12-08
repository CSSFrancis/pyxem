from pyxem.decomposition.vector_decomposition import VectorDecomposition2D
import numpy as np
from skimage.draw import disk
from skimage.morphology import flood

from scipy.ndimage import center_of_mass


def get_vdf(data,
            vector,
            threshold=None,
            radius=2,
            extent=None, ):
    shape = tuple(reversed(data.axes_manager.signal_shape))
    rr, cc = disk(center=(vector[-2], vector[-3]),
                  radius=radius,
                  shape=shape)
    if extent is None:
        xslice, yslice = (slice(None), slice(None))

    vdf = np.sum(data.data[xslice, yslice, rr, cc], axis=(2))

    if threshold is not None:
        search = 5
        start0 = int(vector[0] - search)
        start1 = int(vector[1] - search)
        stop0 = int(vector[0] + search)
        stop1 = int(vector[1] + search)
        if start0 < 0:
            start0 = 0
        if start1 < 0:
            start1 = 0
        if stop0 > vdf.shape[0]:
            stop0 = vdf.shape[0]-1
        if stop1 > vdf.shape[1]:
            stop1 = vdf.shape[1]-1
        sl0 = slice(start0, stop0)
        sl1 = slice(start1, stop1)

        change = np.unravel_index(np.argmax(vdf[sl0, sl1]), (stop0-start0, stop1-start1))
        center = np.array([start0, start1]) + change
        maximum = vdf[int(center[0]), int(center[1])]
        minimum = np.min(vdf)
        difference = maximum - minimum
        thresh = minimum + threshold * difference
        mask = vdf > thresh
        mask = flood(mask, seed_point=(int(vector[0]), int(vector[1])))
        if np.sum(mask) > (np.product(vdf.shape)/2):
            vdf = np.zeros(vdf.shape)
        else:
            vdf[np.logical_not(mask)] = 0
        center = center_of_mass(vdf)
    return center, vdf


class DiffractionVector(VectorDecomposition2D):

    def get_extents(self,
                    data,
                    trim_duplicates=True,
                    threshold=0.9,
                    **kwargs):
        for i, v in enumerate(self.vectors):
            center, vdf = get_vdf(data,
                                  v,
                                  threshold=threshold,
                                  **kwargs,)
            self.extents.append(vdf)
            print(self.vectors[i])
            self.vectors[i][:2] = center