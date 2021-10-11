import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_laplace as sci_gaussian_laplace

from hyperspy._signals.signal2d import Signal2D

from pyxem.utils.cluster_tools import find_peaks
from pyxem.utils.cluster_roi import Clusters


class ClusterGenerator:
    """A workflow for finding clusters in diffraction patterns.

    This method is designed to work on many different signals and for identifying
     local clusters
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
        self.space_scale_rep = None
        self.clusters = None
        self.signal_mask = mask
        self.sigma = None

    def get_space_scale_rep(self,
                            sigma=(1, 1, 1, 1),
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

    def plot_symmetries(self,
                        mask,
                        ax=None,
                        fig_size=None,
                        **kwargs):
        symmetries = self.clusters.get_symmetries(mask=mask)
        if ax is not None:
            ax, f = plt.subplot(1, 1, fig_size)
        ax.bar(symmetries, **kwargs)
        return

    def plot_k_distribution(self):
        pass

    def refine_clusters(self):
        pass

