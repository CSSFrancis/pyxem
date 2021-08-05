from fractions import Fraction as frac

from matplotlib.pyplot import Circle
from skimage.draw import disk
from scipy.ndimage import gaussian_filter as sci_gaussian_filter
import numpy as np

import hyperspy.api as hs
from hyperspy.roi import CircleROI
from hyperspy.signals import Signal1D, Signal2D
from pyxem.utils.correlation_utils import _cross_correlate_masked, get_interpolation_matrix


class Cluster(CircleROI):
    """This is a real space cluster of like symmetry diffraction
    patterns in a sample. Eventually it may allow for more diverse
    shapes than circles but this is a useful tool to getting cluster
    information from different signals by allowing the user to easily
    slice all forms of some dataset.

    All of the units for the ClusterROI are in pixels rather than in real
    units to make things easier...
    """

    def __init__(self,
                 real_indexes,
                 pixel_indexes,
                 cluster_indexes=[1, 2],
                 speckle_indexes=[3, 4],
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
        radius = radius/2
        super().__init__(cx=real_indexes[cluster_indexes[0]],
                         cy=real_indexes[cluster_indexes[1]],
                         r=radius,
                         **kwargs)
        self.cluster_indexes = cluster_indexes
        self.speckle_indexes = speckle_indexes
        self.real_indexes = real_indexes
        self.pixel_indexes = pixel_indexes
        self.symmetry = None
        self.correlation = None
        self.intensities = None

    def __str__(self):
        return ("Position: < " + str(self.real_indexes) +" >" +
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
        ind = np.flip(np.array(self.pixel_indexes[:3], dtype=int))
        return signal.inav[ind]

    def get_kernel(self,
                   signal,
                   radius,
                   ):
        shape = tuple(reversed(signal.axes_manager.signal_shape))
        mask = np.zeros(shape, dtype=bool)
        rr, cc = disk((self.pixel_indexes[self.speckle_indexes[0]],
                       self.pixel_indexes[self.speckle_indexes[1]]),
                      radius=radius,
                      shape=shape)
        mask[rr, cc] = True
        data = np.copy(signal.data.data)
        data[~mask] = 0
        # Not sure why this is necessary... Need to double Check
        #data = np.flip(data, axis=1)
        return data, mask

    def get_correlation(self,
                        signal,
                        radius,
                        mask=None,
                        summed=True,
                        ):
        mean = self.get_mean(signal)
        kernel, mask2 = self.get_kernel(signal=mean, radius=radius)
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
            cor.axes_manager[0].scale = (np.pi*2)/len(cor.data)
            cor.axes_manager[0].unit = "Radians"
            cor.axes_manager[0].name = "Correlation, $\phi$ "
        else:
            cor = Signal2D(cor)
            cor.axes_manager[1].scale = (np.pi * 2)/len(cor.data)
            cor.axes_manager[1].unit = "Radians"
            cor.axes_manager[1].name = "Correlation, $\phi$ "
            cor.axes_manager[0].scale = mean.axes_manager[0].scale
            cor.axes_manager[0].unit = mean.axes_manager[0].unit
            cor.axes_manager[0].name = mean.axes_manager[0].name
        self.correlation = cor

    def get_symmetry(self,
                     symmetries=(1, 2, 4, 6, 8, 10),
                     include_duplicates=False,
                     angular_range=0,
                     method="sum"):
        if method is "average":
            normalize = True
            method = "sum"
        else:
            normalize=False
        angles = [set(frac(j, i) for j in range(0, i)) for i in symmetries]
        if not include_duplicates:
            already_used = set()
            new_angles = []
            for a in angles:
                new_angles.append(a.difference(already_used))
                already_used = already_used.union(a)
            angles = new_angles
        num_angles = [len(a) for a in angles]
        interp = np.array([get_interpolation_matrix(a,
                                           angular_range,
                                           num_points=len(self.correlation.data),
                                           method=method)
                  for a in angles])
        symmetries = np.matmul(self.correlation.data, np.transpose(interp))
        if normalize:
            np.divide(symmetries, num_angles)
        self.symmetry=symmetries
    
    def get_intensities(self):
        pass

    def plot(self,
             **kwargs):
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
        return "<Collection of:" + str(len(self)) + " Clusters>"

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
            kx = c.pixel_indexes[c.speckle_indexes[0]]
            ky = c.pixel_indexes[c.speckle_indexes[1]]
            rr, cc = disk((kx, ky), 3, shape=shape)
            cx = c.pixel_indexes[c.cluster_indexes[0]]
            cy = c.pixel_indexes[c.cluster_indexes[1]]
            data[int(cx-1):int(cx+1), int(cy-1):int(cy+1), rr, cc] = True
        #data = sci_gaussian_filter(data, (1, 1, 0, 0))
        return hs.signals.Signal2D(data)

    def get_correlations(self,
                         signal,
                         radius=3,
                         mask=None):
        for cluster in self:
            cluster.get_correlation(signal=signal,
                                    radius=radius,
                                    mask=mask)

    def get_symmetries(self, **kwargs):
        for cluster in self:
            cluster.get_symmetry(**kwargs)

    @property
    def symmetries(self):
        return [c.symmetry for c in self]

    def get_radius(self, mask):
        return [cluster.r for cluster, m in zip(self, mask) if m]

    def get_index(self, index, mask):
        return [cluster.index[index] for cluster, m in zip(self, mask) if m]

    def get_intensities(self, mask):
        return [cluster.intensities for cluster, m in zip(self, mask) if m]
