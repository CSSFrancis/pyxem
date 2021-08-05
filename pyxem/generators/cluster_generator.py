import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_laplace as sci_gaussian_laplace

from hyperspy._signals.signal2d import Signal2D

from pyxem.utils.cluster_tools import find_peaks


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
                            min_sigma=(1, 1, 1, 1),
                            max_sigma=(10, 10, 1, 1),
                            num_sigma=5,
                            log_scale=False,
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
                Aby additional parmaters passed to the laplaican of Gaussian Function
            For more information https://en.wikipedia.org/wiki/Scale_space.
        """
        # if both min and max sigma are scalar, function returns only one sigma
        if np.isscalar(max_sigma):
            max_sigma = np.full(self.signal.data.ndim,
                                max_sigma,
                                dtype=float)
        if np.isscalar(min_sigma):
            min_sigma = np.full(self.signal.data.ndim,
                                min_sigma,
                                dtype=float)
        min_sigma = np.asarray(min_sigma, dtype=float)
        max_sigma = np.asarray(max_sigma, dtype=float)
        if log_scale:
            # for anisotropic data, we use the "highest resolution/variance" axis
            standard_axis = np.argmax(min_sigma)
            start = np.log10(min_sigma[standard_axis])
            stop = np.log10(max_sigma[standard_axis])
            scale = np.logspace(start, stop, num_sigma)[:, np.newaxis]
            sigma_list = scale * min_sigma / np.max(min_sigma)
        else:
            scale = np.linspace(0, 1, num_sigma)[:, np.newaxis]
            sigma_list = scale * (max_sigma - min_sigma) + min_sigma
        # computing gaussian laplace
        # average s**2 provides scale invariance
        gl_images = [-sci_gaussian_laplace(self.signal.data, s, **kwargs) * s[0] ** 2
                     for s in sigma_list]
        gl_images = Signal2D(data=gl_images)
        gl_images.axes_manager.navigation_axes[-1].name = "Sigma"
        gl_images.axes_manager.navigation_axes[-1].scale = (max_sigma[0] - min_sigma[0]) / num_sigma
        gl_images.axes_manager.navigation_axes[-1].offset = min_sigma[0]

        for ax1, ax2 in zip(gl_images.axes_manager.navigation_axes[:-1],
                            self.signal.axes_manager.navigation_axes):
            ax1.scale = ax2.scale
            #ax1.units = ax2.units
            ax1.offset = ax2.offset
            ax1.name = ax2.name
        for ax1, ax2 in zip(gl_images.axes_manager.signal_axes[:-1],
                            self.signal.axes_manager.signal_axes):
            ax1.scale = ax2.scale
            #ax1.units = ax2.units
            ax1.offset = ax2.offset
            ax1.name = ax2.name

        self.space_scale_rep = gl_images
        self.sigma = sigma_list
        return

    def get_clusters(self,
                     **kwargs):
        if self.space_scale_rep is None:
            print("The space scale representation must first be intialized."
                  "Please run the `get_space_scale_rep` function. ")
            return
        self.clusters = find_peaks(signal=self.space_scale_rep,
                                   **kwargs)

    def get_correlations(self,
                         radius=3,
                         mask=None):
        self.clusters.get_correlations(self.signal,
                                       radius=radius,
                                       mask=mask)

    def get_symmetries(self, **kwargs):
        self.clusters.get_symmetries(**kwargs)

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

    def plot_cluster_radius(self, mask=None):
        symmetries = self.clusters.get_symmetries(mask=mask)
        if ax is not None:
            ax, f = plt.subplot(1, 1, fig_size)
        ax.bar(symmetries, **kwargs)

    def plot_k_distribution(self):
        pass

    def refine_clusters(self):
        pass

