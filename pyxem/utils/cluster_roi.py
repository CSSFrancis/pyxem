from fractions import Fraction as frac

import matplotlib.pyplot as plt
from matplotlib.pyplot import Circle
from skimage.draw import disk
from scipy.spatial import distance_matrix
import numpy as np

import hyperspy.api as hs
from hyperspy.roi import CircleROI
from hyperspy.signals import Signal1D, Signal2D
from pyxem.utils.correlation_utils import _cross_correlate_masked, get_interpolation_matrix, symmetry_stem

class DiffractionFeature:

    def __init__(self,
                 vector,
                 signal_extent=None,
                 navigation_extent=None,
                 obj=None):
        """ Initalizes the Diffraction Feature Object.
        """
        self.vector = vector
        self.signal_extent
        self.navigation_extent = navigation_extent
        self.obj = obj
        self.chunk_index=None

    @property
    def real_position(self):
        if self.obj is not None:
            return [a.index2value(i) for a, i in zip(self.obj.axes_manager, self.vector)]
        else:
            return



class DiffractionFeatures(list):

    def group_features(self, max_distance=1, symetries=(2,4,6,10),):
        navigation_positions = [c.vector[0:2] for c in self]
        distance = distance_matrix(navigation_positions, navigation_positions)
        num_inside = np.sum([distance < max_distance], axis=0)
        nearest = np.transpose(np.argsort(distance, axis=0))
        nearest = [nearest[1:num] for n, num in zip(nearest, num_inside)]
        grouped_clusters = [[self[n] for n in near] for near in nearest]




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
                 cluster_indexes=[0, 1],
                 speckle_indexes=[2, 3],
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
        self.extent = None

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
        ind = np.flip(np.array(self.pixel_indexes[:2], dtype=int))
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
        ind = np.flip(np.array(self.pixel_indexes[:2], dtype=int))
        mean = signal.inav[ind]
        return mean

    def get_slice(self, sl_extent=0.5):
        """Gets a slice from the signal around the center of the cluster

        Parameters
        ---------------
        sl_extent: float
            The extent of the area in real space to slice. This should be based on your probe size,
            Size of clusters and the size of the spacing between probes.
        """
        if self.obj is None:
            print("Set a cluster generator object")
        sl = self.obj.signal.inav[self.real_indexes[1] - sl_extent:self.real_indexes[1] + sl_extent,
                                  self.real_indexes[0] - sl_extent:self.real_indexes[0] + sl_extent]
        return sl

    def get_extent(self, radius=3, test_region=5):
        sl = self.get_slice(sl_extent=test_region)
        mask = self.get_kernel_mask(radius=radius)
        return np.tensordot(sl, mask)

    def get_kernel_mask(self,
                        radius=3):
        if self.obj is None:
            print("Set a cluster generator object")
            return
        shape = tuple(reversed(self.obj.signal.axes_manager.signal_shape))
        mask = np.zeros(shape, dtype=bool)
        rr, cc = disk((self.pixel_indexes[self.speckle_indexes[0]],
                       self.pixel_indexes[self.speckle_indexes[1]]),
                      radius=radius,
                      shape=shape)
        mask[rr, cc] = True
        return mask


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
                        radius=2,
                        extent=0.2,
                        mask=None,
                        summed=True,
                        method = "auto",
                        **kwargs,
                        ):
        if method is "auto":
            sl = self.get_slice(sl_extent=extent)
            cor = sl.get_angular_correlation(mask=mask).mean(axis=(0, 1))
            cor = cor.sum(axis=1)
        elif method is "cross":
            sl = self.get_slice(sl_extent=extent)
            mean = self.get_mean(signal)
            kernel, kern_mask = self.get_kernel(signal=mean, radius=radius)
            mask2 = np.zeros(kernel.shape,
                             dtype=bool)
            cor = sl.map(_cross_correlate_masked,
                         z2=kernel,
                         mask1=mask,
                         mask2=mask2,
                         axs=1,
                         pad_axis=None,
                         inplace=False,
                         **kwargs).mean(axis=(0, 1))
            cor = cor.sum(axis=1)

        elif method is "cross_kernel":
            sl = self.get_slice(sl_extent=extent)
            mean = self.get_mean(signal)
            kernel, kern_mask = self.get_kernel(signal=mean, radius=radius)
            mask2 = np.zeros(kernel.shape,
                             dtype=bool)
            cor = sl.map(_cross_correlate_masked,
                         z2=kern_mask,
                         mask1=mask,
                         mask2=mask2,
                         axs=1,
                         pad_axis=None,
                         inplace=False,
                         **kwargs).mean(axis=(0, 1))
            cor = cor.sum(axis=1)

        else:
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
                                          axs=1,
                                          pad_axis=None,
                                          **kwargs,
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
                #cor.axes_manager[1].unit = "Radians"
                cor.axes_manager[1].name = "Correlation, $\phi$ "
                cor.axes_manager[0].scale = mean.axes_manager[0].scale
                #cor.axes_manager[0].unit = mean.axes_manager[0].unit
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
                         symmetries=(1, 2, 4, 6, 10),
                         ):
        self.max_sym = symmetries[np.argmax(self.symmetry[1:]) + 1]
        self.intensities = [self.symmetry[0], np.max(self.symmetry[1:])]

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

    @property
    def max_symmetries(self):
        return [c.max_sym for c in self]

    def trim_clusters(self, max_self_cor=True, ratio_trim=0.2):
        """Trims any clusters which have a mask cluster
        """
        if max_self_cor and ratio_trim is not None:
            clus = [c for c in self if np.argmax(c.symmetry) == 0 and
                    (np.max(c.intensities[1:])/c.intensities[0]) > ratio_trim]
            self.clear()
            self.extend(clus)
        elif max_self_cor:
            clus = [c for c in self if (np.argmax(c.symmetry) == 0)]
            self.clear()
            self.extend(clus)
        elif ratio_trim is not None:
            clus = [c for c in self if (c.intensities[0] / np.max(c.intensities[1:]) < ratio_trim)]
            self.clear()
            self.extend(clus)
        else:
            "Nothing done"
            return


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
                         mask=None,
                         **kwargs):
        if signal is None and self.obj is not None:
            signal = self.obj.space_scale_rep
        else:
            print("Either a signal or a cluster generator obj must be intialized")
        for cluster in self:
            cluster.get_correlation(signal=signal,
                                    radius=radius,
                                    mask=mask,
                                    **kwargs)

    def get_symmetries(self,
                       symmetries=(1, 2, 4, 6, 10),
                       advanced=True,
                       **kwargs,
                       ):
        for cluster in self:
            cluster.get_symmetry(symmetries=symmetries, **kwargs)
        for cluster in self:
            cluster.get_max_symmetry(syms=symmetries)

        if advanced:
            for cluster in self:
                self.get_extent

    def get_max_sym_indexes(self, symmetries=(1, 2, 4, 6, 10)):
        syms = np.array([c.max_sym for c in self])
        return [np.where(syms == s) for s in symmetries]

    def get_means(self,
                  signal=None,
                  ):
        if signal is None and self.obj is not None:
            signal = self.obj.space_scale_rep
        for cluster in self:
            cluster.get_mean(signal=signal)

    def get_index(self, index, mask):
        return [cluster.index[index] for cluster, m in zip(self, mask) if m]

    def get_intensities(self, mask):
        return [cluster.intensities for cluster, m in zip(self, mask) if m]

    def plot_intensities(self, symmetries=(2, 4, 6, 10),figsize=(10,13), **kwargs):
        ind = self.get_max_sym_indexes(symmetries)
        intensities = np.array([cluster.intensities for cluster in self])
        int0 = [intensities[i, 0] for i in ind]
        int1 = [intensities[i, 1] for i in ind]
        f, axs = plt.subplots(len(symmetries), 2, figsize=figsize)
        for i0, i1, ax in zip(int0, int1,  axs):
            ax[0].hist(i0[0], **kwargs)
            ax[0].set(xlabel="Correlation Intensity", ylabel="Number of Clusters")
            ax[1].hist(i1[0], **kwargs)
            ax[1].set(xlabel="Correlation Intensity", ylabel="Number of Clusters")

    def plot_k_range(self,
                     symmetries=(2, 4, 6, 10),
                     figsize=(10,13),
                     **kwargs):
        ind = self.get_max_sym_indexes(symmetries)
        intensities = np.array([cluster.real_indexes[-2] for cluster in self])
        k = [intensities[i] for i in ind]
        f, axs = plt.subplots(len(symmetries), 1, figsize=figsize)
        for k1, ax in zip(k, axs):
            ax.hist(k1, **kwargs)
            ax.set(xlabel="k, nm$^-1$", ylabel="Number of Clusters")

    def plot_extent(self,
                    symmetries=(2, 4, 6, 10),
                    figsize=(10,13),
                    test_region=5,
                    out_shape=(12,12),
                    slice=None,
                    **kwargs,
                    ):
        ind = self.get_max_sym_indexes(symmetries)
        extents = np.array([c.get_extent(test_region=test_region) for c in self])
        sym_ext = [extents[i] for i in ind]
        sym_ext = [[e for e in s if e.shape == out_shape] for s in sym_ext]
        mean = [np.mean(e, axis=0) for e in sym_ext]

        if slice is None:
            f, axs = plt.subplots(int(np.ceil(len(symmetries)/2)), 2, figsize=figsize)
            axs = np.ndarray.flatten(axs)
            for m, ax in zip(mean, axs):
                ax.imshow(m, **kwargs)
        else:
            deviation = [np.std(e, axis=0) for e in sym_ext]
            f, axs = plt.subplots(1, 2, figsize=figsize)
            for m,s,st in zip(mean, symmetries,deviation):
                axs[0].plot(m[:, slice], label=str(s)+"-fold Symmetry", **kwargs)
                axs[1].plot(st[:, slice], label=str(s) + "-fold Symmetry", **kwargs)
            plt.legend()




