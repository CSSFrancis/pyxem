from abc import ABC

from hyperspy.roi import CircleROI
from matplotlib.pyplot import Circle
import hyperspy.api as hs
from skimage.draw import circle
from pyxem.utils.correlation_utils import _cross_correlate_masked
from scipy.ndimage import gaussian_filter as sci_gaussian_filter
import numpy as np

from hyperspy.signals import Signal1D, Signal2D


class Cluster(CircleROI):
    """This is a real space cluster of like symmetry diffraction
    patterns in a sample. Eventually it may allow for more diverse
    shapes than circles but this is a useful tool to getting cluster
    information from different signals by allowing the user to easily
    slice all forms of some dataset.
    """

    def __init__(self,
                 indexes,
                 cluster_indexes=[0,1],
                 speckle_indexes=[2,3],
                 radius=None,
                 **kwargs):
        """ Initializes some cluster. The coordinates x and y are the real space
        coordinates defining the circle of interest
        Parameters
        -----------
        x: float
            The real space x position of the cluster
        y: float
            The real space y position of the clusters
        r: float
            The radius of the cluster
        k: float
            The inverse spacing for the cluster
        correlation: Array-like
            The saved angular correlation for the center of the cluster at some k.
        """
        super().__init__(cx=indexes[cluster_indexes[0]],
                         cy=indexes[cluster_indexes[1]],
                         r=radius,
                         **kwargs)
        self.cluster_indexes =cluster_indexes
        self.speckle_indexes = speckle_indexes
        self.indexes = indexes
        self.symmetry = None
        self.correlation = None
        self.intensities = None

    def __str__(self):
        return ("Position: < "+ self.indexes +" >" +
                " radius: " + str(self.r) +
                " Symmetry: " + str(self.symmetry))

    def to_circle(self,
                  linewidth=2,
                  fill=False,
                  color="blue",
                  alpha=None,
                  **kwargs):
        """This takes the object and turns it into a matplotlib.Circle object
        """
        return Circle(xy=(self.cy, self.cx),
                      radius=self.r,
                      linewidth=linewidth,
                      fill=fill,
                      color=color,
                      alpha=alpha,
                      **kwargs)

    def get_mean(self,
                 signal):
        return self(signal, axes=self.cluster_indexes).nansum()

    def get_kernel(self,
                   signal,
                   radius,
                   ):
        shape = tuple(reversed(signal.axes_manager.signal_shape))
        mask = np.zeros(shape, dtype=bool)
        rr, cc = circle(self.indexes[self.speckle_indexes[1]],
                        self.indexes[self.speckle_indexes[0]],
                        radius,
                        shape)
        mask[rr, cc] = True
        data = signal.data  # might just pass the reference
        data[~mask] = 0
        return data

    def get_correlation(self,
                        signal,
                        radius,
                        mask=None,
                        summed=True,
                        ):
        mean = self.get_mean(signal)
        kernel = self.get_kernel(signal=signal, radius=radius)
        if mask is None:
            mask = np.zeros(kernel.shape,
                            dtype=bool)
        mask2 = np.zeros(kernel.shape,
                         dtype=bool)
        cor = _cross_correlate_masked(z1=mean.data,
                                      z2=kernel,
                                      mask1=mask,
                                      mask2=mask2,
                                      axis=1,
                                      )
        if summed:
            cor = cor.sum(axis=0)
            cor = Signal1D(cor)
            cor.axes_manager[0].scale = len(cor.data)/(np.pi*2)
            cor.axes_manager[0].unit = "Radians"
            cor.axes_manager[0].name = "Correlation, $\phi$ "
        else:
            cor = Signal2D(cor)
            cor.axes_manager[1].scale = len(cor.data) / (np.pi * 2)
            cor.axes_manager[1].unit = "Radians"
            cor.axes_manager[1].name = "Correlation, $\phi$ "
            cor.axes_manager[0].scale = mean.axes_manager[0].scale
            cor.axes_manager[0].unit = mean.axes_manager[0].unit
            cor.axes_manager[0].name = mean.axes_manager[0].name
        self.correlation = cor

    def get_symmetry(self):
        pass
    
    def get_intensities(self):
        pass

    def plot(self, **kwargs):
        markers = [hs.markers.vertical_line(j * 6.28 / self.symmetry) for j in range(self.symmetry)]
        self.correlation.add_marker(markers, permanent=True, plot_marker=True)


class Clusters(list):
    """This is a group of clusters for some experiment.  This class is designed to organize clusters into a unit for
    easier processing.
    """

    def __init__(self,
                 cluster_list):
        """ Initializes a cluster based on a list of clusters.
        """
        super().__init__(cluster_list)

    def __str__(self):
        return "Number of Clusters: <" + len(self) + " >"

    def to_markers(self,
                   navigation_shape,
                   **kwargs):
        """This takes the object and turns it into a matplotlib.Circle object
        """
        xindexes = np.zeros(navigation_shape)
        yindexes = np.zeros(navigation_shape)

        for cluster in self:
            xindexes[int(cluster.cy), int(cluster.cx)] = cluster.kx
            yindexes[int(cluster.cy), int(cluster.cx)] = cluster.ky

        markers = hs.plot.markers.point(xindexes, yindexes, **kwargs)
        xx, yy = [int(c.cx) for c in self], [int(c.cy) for c in self]
        nav_markers = hs.plot.markers.point(yy, xx, **kwargs)

        return markers, nav_markers

    def to_signal(self,
                  shape,
                  ):
        # add in symmetry plotting
        data = np.zeros(shape, dtype=bool)
        for c in self:
            rr, cc = circle(c.ky, c.kx, 4, shape=shape[-2:])
            data[int(c.cx), int(c.cy), rr, cc] = True
        data = sci_gaussian_filter(data, (1, 1, 0, 0))
        return hs.signals.Signal2D(data)

    def get_correlations(self,
                         signal,
                         mask):
        for cluster in self:
            cluster.get_correlation(signal=signal,
                                    mask=mask)

    def get_symmetries(self, mask):
        return [cluster.symmetry for cluster, m in zip(self, mask) if m]

    def get_radius(self, mask):
        return [cluster.r for cluster, m in zip(self, mask) if m]

    def get_index(self, index, mask):
        return [cluster.index[index] for cluster, m in zip(self, mask) if m]

    def get_intensities(self, mask):
        return [cluster.intensities for cluster, m in zip(self, mask) if m]
