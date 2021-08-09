from fractions import Fraction as frac

from matplotlib.pyplot import Circle
from skimage.draw import disk
from scipy.ndimage import gaussian_filter as sci_gaussian_filter
import numpy as np
from dask import delayed

import hyperspy.api as hs
from hyperspy.roi import CircleROI
from hyperspy.signals import Signal1D, Signal2D
from pyxem.utils.correlation_utils import _cross_correlate_masked, get_interpolation_matrix, symmetry_stem


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
                 obj=None,
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
        self.max_sym =None
        self.correlation = None
        self.intensities = None
        self.obj = obj

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

    @property
    def mean(self):
        if self.obj is None:
            print("Set a cluster generator object")
        ind = np.flip(np.array(self.pixel_indexes[:3], dtype=int))
        mean = self.obj.space_scale_rep.inav[ind]
        point = hs.plot.markers.point(self.real_indexes[-1],
                                      self.real_indexes[-2],
                                      color="blue",
                                      size=100)
        mean.add_marker(point,
                        permanent=True,
                        render_figure=False,
                        plot_marker=False)
        return mean

    def get_mean(self,
                 signal):
        ind = np.flip(np.array(self.pixel_indexes[:3], dtype=int))
        mean = signal.inav[ind]
        return mean

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
            normalize = False
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
        symmetries = symmetry_stem(self.correlation.data,
                                   interpolation=interp,
                                   method=method)
        if normalize:
            np.divide(symmetries, num_angles)
        self.symmetry = symmetries

    def get_max_symmetry(self,
                         syms=(1,2,4,6,10),
                         ):
        self.max_sym = syms[np.argmax(self.symmetry[1:]) + 1]

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
                 cluster_list,
                 obj=None):
        """ Initializes a cluster based on a list of clusters.
        """
        super().__init__(cluster_list)
        self.obj = obj

    def __str__(self):
        return "<Collection of:" + str(len(self)) + " Clusters>"

    @property
    def symmetries(self):
        return [c.symmetry for c in self]



    def to_markers(self,
                   navigation_shape=None,
                   **kwargs):
        """This takes the object and turns it into a matplotlib.Circle object
        """
        if (navigation_shape is None) and (self.obj is not None):
            navigation_shape = self.obj.signal.axes_manager.navigation_shape
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
                  shape=None,
                  ):
        if (shape is None) and (self.obj is not None):
            shape = np.shape(self.obj.signal.data)
        data = np.zeros(shape, dtype=bool)
        for c in self:
            kx = c.pixel_indexes[c.speckle_indexes[0]]
            ky = c.pixel_indexes[c.speckle_indexes[1]]
            rr, cc = disk((kx, ky), 3, shape=shape[-2:])
            cx = c.pixel_indexes[c.cluster_indexes[0]]
            cy = c.pixel_indexes[c.cluster_indexes[1]]
            data[int(cx-1):int(cx+1), int(cy-1):int(cy+1), rr, cc] = True
        #data = sci_gaussian_filter(data, (1, 1, 0, 0))
        s = hs.signals.Signal2D(data)
        if self.obj is not None:
            for ax1, ax2 in zip(s.axes_manager.navigation_axes,
                                self.obj.signal.axes_manager.navigation_axes):
                ax1.scale = ax2.scale
                # ax1.units = ax2.units
                ax1.offset = ax2.offset
                ax1.name = ax2.name
            for ax1, ax2 in zip(s.axes_manager.signal_axes,
                                self.obj.signal.axes_manager.signal_axes):
                ax1.scale = ax2.scale
                # ax1.units = ax2.units
                ax1.offset = ax2.offset
                ax1.name = ax2.name
        return s

    def get_correlations(self,
                         signal=None,
                         radius=3,
                         mask=None):
        if signal is None and self.obj is not None:
            signal = self.obj.space_scale_rep
        else:
            print("Either a signal or a cluster generator obj must be intialized")
        for cluster in self:
            cluster.get_correlation(signal=signal,
                                    radius=radius,
                                    mask=mask)

    def get_symmetries(self,
                       **kwargs):
        for cluster in self:
            cluster.get_symmetry(**kwargs)

    def get_means(self,
                  signal=None,
                  **kwargs):
        if signal is None and self.obj is not None:
            signal = self.obj.space_scale_rep
        for cluster in self:
            cluster.get_mean(signal=signal)


    def get_radius(self, mask):
        return [cluster.r for cluster, m in zip(self, mask) if m]

    def get_index(self, index, mask):
        return [cluster.index[index] for cluster, m in zip(self, mask) if m]

    def get_intensities(self, mask):
        return [cluster.intensities for cluster, m in zip(self, mask) if m]
