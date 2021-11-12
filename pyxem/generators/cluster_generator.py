import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_laplace as sci_gaussian_laplace
from dask.array.overlap import overlap, trim_internal
from dask.graph_manipulation import clone
from dask.array import concatenate

from pyxem.utils.cluster_tools import find_peaks
from pyxem.utils.cluster_roi import Clusters


class PeakGenerator:
    """A workflow for finding peaks in diffraction patterns.

    This method is designed to work on many different signals and for identifying
    important features in a dataset.
    """
    def __init__(self,
                 signal,
                 mask=None):
        """
        Parameters
        ------------
        signal : hyperspy.Signal2D
            Some ndimensional signal which is to be analyzed.
        """
        self.signal = signal
        self.signal_mask = mask
        self.space_scale_rep = None
        self.lazy_space_scale_rep = None
        self.clusters = None
        self.sigma = None
        self.lazy = self.signal._lazy

    def __str__(self):
        if self.lazy:
            return "<Lazy Peak Generator-- Peaks:" + str(len(self.clusters)) + "; Signal" + str(self.signal)
        else:
            return "<Peak Generator-- Peaks:" + str(len(self.clusters)) + "; Signal" + str(self.signal)

    def get_gaussian_laplace(self,
                            sigma=(1, 1, 1, 1),
                            beam_size=None,
                            **kwargs,
                            ):
        """ This method returns a space scale representation of the data. This is a
            common data form for finding objects that have some specific spatial frequency.

            Parameters
            -----------
            signal: signal
                The signal to be filtered
            min_sigma: tuple
                The minimum sigma to be applied to the dataset
            max_sigma: tuple
                The max sigma to be applied to the dataset
            num_sigma: int
                The umber of sigma to be analyzed
            log_scale: bool
                If the step size for the min --> max sigma should be in a log scale
            kwargs: dict
                Aby additional parameters passed to the laplaican of Gaussian Function
            For more information https://en.wikipedia.org/wiki/Scale_space.
        """
        # if both min and max sigma are scalar, function returns only one sigma
        if np.isscalar(sigma):
            sigma = np.full(self.signal.data.ndim,
                                dtype=float)
        # computing gaussian laplace
        if self.signal._lazy:
            overlap_dist = sigma*2
            overlap_dist = (overlap_dist[0], overlap_dist[1], 0, 0)
            overlapped = overlap(self.signal.data, overlap_dist, None)
            clones = concatenate([clone(b, omit=overlapped) for b in overlapped.blocks])
            lazy_lap = clones.map_blocks(sci_gaussian_laplace, sigma=sigma, dtype=self.signal.dtype)
            self.lazy_space_scale_rep = lazy_lap
            lap = trim_internal(lazy_lap, overlap_dist)
            self.signal._deepcopy_with_new_data(data=lap)
        else:
            lap = self.signal._deepcopy_with_new_data(data=-sci_gaussian_laplace(self.signal.data,
                                                                                 sigma,
                                                                                 **kwargs))
        self.space_scale_rep = lap
        self.sigma = sigma
        return

    def get_clusters(self,
                     mask=None,
                     **kwargs):
        if self.space_scale_rep is None:
            print("The space scale representation must first be intialized."
                  "Please run the `get_space_scale_rep` function. ")
            return
        self.clusters = Clusters(find_peaks(signal=self.space_scale_rep,
                                            mask=mask,
                                            **kwargs,
                                            obj=self),
                                 obj=self)
