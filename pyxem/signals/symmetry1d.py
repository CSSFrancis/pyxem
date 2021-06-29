from hyperspy.utils.plot import plot_images
from hyperspy.signals import Signal1D
import hyperspy.api as hs

import matplotlib.pyplot as plt
import numpy as np
from pyxem.utils.cluster_roi import Cluster
from pyxem.utils.correlation_utils import blob_finding,peak_finding
from scipy.ndimage import gaussian_filter as sci_gaussian_filter
from dask_image.ndfilters import gaussian_filter as lazy_gaussian_filter
from scipy.ndimage import gaussian_laplace as sci_gaussian_laplace
from dask_image.ndfilters import gaussian_laplace as lazy_gaussian_laplace
from hyperspy.api import stack

colors = ["black", "blue", "red", "green", "yellow", "orange", "purple", "gray", "pink", "cyan",
          "olive", "brown", "ivory", "lime", "gold"]
class Symmetry1D(Signal1D):
    _signal_type = "symmetry"

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.clusters = None
        self.sigma = None

    @property
    def symmetries(self):
        try:
            return self.metadata.Signal["symmetries"]
        except (AttributeError):
            return None

    @symmetries.setter
    def symmetries(self, symmetries):
        self.metadata.set_item("Signal.symmetries", symmetries)

    def get_clusters(self,
                     method="log",
                     k_range=None,
                     **kwargs):
        """This method uses the three different blob finding algorithms from
         `skimage.feature` and then returns those blobs as hyperspy ROI's.

        Parameters
        ------------
        method: one of 'dog', 'log' or 'doh'
            The skimage method to use to find the blobs (clusters in most case)
        k-range: None or list
            The region of interest to search for clusters. If k-range= None
            then 3-dimensional blobs are looked for and the k-range is used to
            further seprate blobs.
        kwargs: dict
            Any additional keyword arguements to pass to the skimage blob finding
             algorithms or any additonal kwargs for the `BaseSignal.map` function
        Returns
        -----------
        cluster_list: list
            A list of the clusters in the form of `hs.roi.CircularROI` objects
        """
        if k_range is not None:

            s = self.isig[k_range[0]:k_range[1]].sum(axis=-1).transpose(navigation_axes=(0,))
        else:
            s = self.transpose(navigation_axes=(0,))
            if method is "doh":
                raise ValueError("k_range can't be equal to None as the 'doh'"
                                 " (difference of Hessians) method doesn't "
                                 "currently support 3 dimensional data")
                return
        s = s.map(blob_finding,
                  method=method,
                  inplace=False,
                  ragged=True,
                  **kwargs)
        cluster_list = []
        for clusters, symmetry in zip(s.data, self.symmetries):
            if k_range is None:
                cluster_list.append([Cluster(x=cluster[0]*self.axes_manager.navigation_axes[-1].scale,
                                             y=cluster[1]*self.axes_manager.navigation_axes[-1].scale,
                                             radius=cluster[3] * np.sqrt(2)*self.axes_manager.navigation_axes[-1].scale,
                                             k=cluster[2]*self.axes_manager.signal_axes[-1].scale,
                                             symmetry=symmetry)
                                     for cluster in clusters])
            else:
                cluster_list.append([Cluster(x=cluster[0]*self.axes_manager.navigation_axes[-1].scale,
                                             y=cluster[1]*self.axes_manager.navigation_axes[-1].scale,
                                             radius=cluster[2] * np.sqrt(2)*self.axes_manager.navigation_axes[-1].scale,
                                             k=k_range,
                                             symmetry=symmetry)
                                     for cluster in clusters])
        self.clusters = cluster_list
        return cluster_list

    def get_space_scale_representation(self,
                                       min_sigma= (1,1,0,0),
                                       max_sigma=(5,5,0, 0),
                                       log_scale=False,
                                       num_sigma=5,
                                       **kwargs):

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

        gl_images = [self.laplacian(sigma=s) for s in sigma_list]

        gl_images = stack(gl_images, axis=None)
        gl_images.axes_manager.navigation_axes[-1].name = "Sigma"
        gl_images.axes_manager.navigation_axes[-1].scale = (max_sigma[0]+1-min_sigma[0])/num_sigma
        gl_images.axes_manager.navigation_axes[-1].offset = min_sigma[0]
        gl_images.sigma = sigma_list[:, 0]
        return gl_images


    def find_peaks(self,
                   overlap=0.5,
                   correlation=None,
                   trim_edges=True,
                   **kwargs):
        """
        This method takes a library of SymmetrySTEM Objects and finds peaks
        in the library.  This method might be moved to a different SymmetrySTEMLibrary Class
        which better handles the different Sigma applied to the dataset.

        :param overlap:
        :param kwargs:
        :return:
        """
        s = self.transpose(navigation_axes=(0,)).map(peak_finding,
                                                     sigma=self.sigma,
                                                     overlap=overlap,
                                                     inplace=False,
                                                     **kwargs)
        cluster_list = []
        print("sigma:", self.sigma)
        for clusters, symmetry in zip(s.data, self.symmetries):
            cluster_sym = [Cluster(x=cluster[2] * self.axes_manager.navigation_axes[2].scale,
                                   y=cluster[1] * self.axes_manager.navigation_axes[2].scale,
                                   radius=self.sigma[int(cluster[0])] * np.sqrt(2) * self.axes_manager.navigation_axes[2].scale,
                                   k=((cluster[3] * self.axes_manager.signal_axes[-1].scale) +
                                   self.axes_manager.signal_axes[-1].offset),
                                   symmetry=symmetry,
                                   correlation=correlation)
                           for cluster in clusters if (cluster[0] > 0 and trim_edges)]
            cluster_list.append(cluster_sym)
        self.clusters = cluster_list
        return cluster_list

    def plot_all(self,
                 k_range,
                 include_clusters=True,
                 fig_size=(10, 10),
                 **kwargs):
        f = plt.figure(figsize=fig_size)
        signals = self.isig[k_range[0]:k_range[1]].sum(axis=-1).split(axis=0)
        print(signals)
        labels = [str(s) + "-Fold Symmetry" for s in self.symmetries]
        plot_images(signals,
                    label=labels,
                    fig=f,
                    **kwargs)
        if include_clusters:
            axs = f.get_axes()
            if self.clusters is not None:
                for ax, clusters in zip(axs[::2], self.clusters):
                    for cluster in clusters:
                        ax.add_patch(cluster.to_circle())

    def get_blurred_library(self,
                            min_cluster_size=0.5,
                            max_cluster_size=5.0,
                            sigma_ratio=1.6,
                            k_sigma=2):
        gaussian_symmetry_stem = []
        if isinstance(k_sigma, float):
            k_sigma = k_sigma / self.axes_manager.signal_axes[0].scale
        if isinstance(min_cluster_size, float):
            min_cluster_size = min_cluster_size / self.axes_manager["x"].scale
        if isinstance(max_cluster_size, float):
            max_cluster_size = max_cluster_size / self.axes_manager["x"].scale

        # k such that min_sigma*(sigma_ratio**k) > max_sigma
        k = int(np.mean(np.log(max_cluster_size / min_cluster_size) / np.log(sigma_ratio) + 1))

        # a geometric progression of standard deviations for gaussian kernels
        sigma_list = np.array([[min_cluster_size * (sigma_ratio ** i),
                                min_cluster_size * (sigma_ratio ** i),
                                0,
                                k_sigma]
                               for i in range(k + 1)])

        for s in sigma_list:
            filtered = self.gaussian_filter(sigma=s, inplace=False)
            gaussian_symmetry_stem.append(filtered)

        print(sigma_list)
        dog_images = [(gaussian_symmetry_stem[i] - gaussian_symmetry_stem[i + 1]) * np.mean(sigma_list[i]) for i in range(k)]
        image_cube = stack(dog_images, axis=None)
        image_cube.axes_manager.navigation_axes[-1].name ="Sigma"
        image_cube.sigma = sigma_list[:, 0]
        return image_cube

    def plot_clusters(self, ax=None, k_range=None, symmetries=None):
        if ax is None:
            fig, ax = plt.subplots()
        extent = self.axes_manager.navigation_extent
        ax.set_xlim(extent[2], extent[3])
        ax.set_ylim(extent[4], extent[5])

        for symmetry, color in zip(self.clusters, colors[:len(self.clusters)]):
            for c in symmetry:
                if k_range is None or (c.k is None or (c.k <k_range[1] and c.k > k_range[0])):
                    if symmetries is None or (c.symmetry in symmetries):
                        ax.add_patch(c.to_circle(fill=True, color=color, alpha=0.5))
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker='o', color=colors[i], label=str(sym) + " fold symmetry",
                      markerfacecolor=colors[i], markersize=15) for i, sym in enumerate(self.symmetries)]

        ax.legend(handles=leg, loc='upper right')
        return ax

    def gaussian_filter(self,
                        sigma,
                        inplace=False):
        if inplace:
            if self._lazy:
                self.data = lazy_gaussian_filter(self.data,
                                                 sigma)
            else:
                self.data = sci_gaussian_filter(self.data, sigma)
        else:
            if self._lazy:
                return self._deepcopy_with_new_data(data=lazy_gaussian_filter(self.data,
                                                                              sigma))
            else:
                return self._deepcopy_with_new_data(data=sci_gaussian_filter(self.data,
                                                                             sigma))
    def laplacian(self, sigma, inplace=False):
        if inplace:
            if self._lazy:
                self.data = [lazy_gaussian_laplace(d, sigma) * np.mean(sigma)for d in self.data]
            else:
                self.data = [sci_gaussian_laplace(d, sigma) * np.mean(sigma) for d in self.data]
        else:
            if self._lazy:
                return self._deepcopy_with_new_data(data=[lazy_gaussian_laplace(d, sigma)*np.mean(sigma)**2
                                                          for d in self.data])
            else:
                return self._deepcopy_with_new_data(data=[sci_gaussian_laplace(d, sigma) * np.mean(sigma) ** 2
                                                          for d in self.data])

    def get_cluster_size_distribution(self):
        radii = [[cluster.r for cluster in symmetry]for symmetry in self.clusters]
        return radii

    def plot_cluster_size_distribution(self,
                                       ax=None,
                                       normalize=False):
        if ax is None:
            fig, ax = plt.subplots()
        size = self.get_cluster_size_distribution()
        rad = self.sigma * np.sqrt(2) * self.axes_manager.navigation_axes[2].scale
        leg = [str(sym) + " fold symmetry" for sym in self.symmetries]
        num = [[s.count(sig) for sig in rad] for s in size]
        ax.set_xlabel("Cluster Size, nm", fontsize=14)
        ax.set_ylabel("Number of Clusters", fontsize=14)
        for s, l in zip(num, leg):
            if not normalize:
                ax.plot(rad, s, label=l)
            else:
                max_val = np.sum(s)
                ax.set_ylabel("Fraction of Clusters", fontsize=14)
                if max_val == 0:
                    max_val = 1
                ax.plot(rad, np.array(s)/max_val, label=l)
        ax.legend(loc='upper right')

    def get_k_range_distribution(self, min_cluster_size=None):
        if min_cluster_size is None:
            k_range = [[cluster.k for cluster in symmetry]for symmetry in self.clusters]
        else:
            k_range = [[cluster.k for cluster in symmetry if cluster.r>min_cluster_size] for symmetry in self.clusters]
        return k_range

    def plot_k_range_distribution(self,
                                  min_cluster_size=None,
                                  ax=None,
                                  normalize=False):
        if ax is None:
            fig, ax = plt.subplots()
        k_range = self.get_k_range_distribution(min_cluster_size)
        s_extent = self.axes_manager.signal_extent
        histogram_k = np.array([np.histogram(ks,
                                             np.shape(self)[-1],
                                             s_extent) for ks in k_range])
        leg = [str(sym) + " fold symmetry" for sym in self.symmetries]
        ax.set_xlabel("k, nm$^{-1}$", fontsize=14)
        ax.set_ylabel("Number of Clusters", fontsize=14)
        for k, l in zip(histogram_k, leg):
            if not normalize:
                ax.plot(k[1][0:-1], k[0], label=l)
            else:
                ax.set_ylabel("Fraction of Clusters", fontsize=14)
                max_val = np.sum(k[0])
                if max_val == 0:
                    max_val = 1
                ax.plot(k[1][0:-1], k[0]/max_val, label=l)
        ax.legend(loc='upper right')

    def plot_d_range_distribution(self,
                                  ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        k_range = self.get_k_range_distribution()
        s_extent = self.axes_manager.signal_extent
        histogram_k = np.array([np.histogram(ks,
                                             np.shape(self)[-1],
                                             s_extent) for ks in k_range])
        leg = [str(sym) + " fold symmetry" for sym in self.symmetries]
        ax.set_xlabel("d-spacing, nm", fontsize=14)
        ax.set_ylabel("Number of Clusters", fontsize=14)

        for k, l in zip(histogram_k, leg):
            d = k[1][0:-1] ** -1
            ax.plot(d, k[0], label=l)
        ax.legend(loc='upper left')
        return ax

    def plot_cluster_stats(self,
                           k_range=True,
                           size=True,
                           spatial=True):
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[:2, :2])
        ax1.set_title('Spatial Clusters')
        self.plot_clusters(ax=ax1)
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.set_title('Size Distribution')
        ax3 = fig.add_subplot(gs[2:, 1])
        ax3.set_title('K Distribution')

    def plot_cluster_numbers(self,
                             ax=None,
                             horizontal=False):
        if ax is None:
            fig, ax = plt.subplots()
        if horizontal:
            ax.barh(range(len(self.clusters)),
                   [len(c) for c in self.clusters],
                   tick_label=[sym for sym in self.symmetries])
        else:
            ax.bar(range(len(self.clusters)),
                   [len(c) for c in self.clusters],
                   tick_label=[sym for sym in self.symmetries])

        ax.set_xlabel("Symmetry,n-Fold", fontsize=14)
        ax.set_ylabel("Number of Clusters", fontsize=14)


