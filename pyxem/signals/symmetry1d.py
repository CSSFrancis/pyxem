from hyperspy.utils.plot import plot_images
from hyperspy.signals import Signal1D
import hyperspy.api as hs

import matplotlib.pyplot as plt
import numpy as np
from pyxem.utils.cluster_roi import Cluster
from pyxem.utils.correlation_utils import blob_finding,peak_finding

colors = ["black", "blue", "red", "green", "yellow", "orange", "purple"]
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
        print(np.shape(s.data))
        cluster_list = []
        for clusters, symmetry in zip(s.data, self.symmetries):
            print(clusters)
            if k_range is None:
                cluster_list.append([Cluster(x=cluster[0]*self.axes_manager.navigation_axes[-1].scale,
                                             y=cluster[1]*self.axes_manager.navigation_axes[-1].scale,
                                             radius=cluster[3] * np.sqrt(2)*self.axes_manager.navigation_axes[-1].scale,
                                             k=cluster[2*self.axes_manager.signal_axes[-1].scale],
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

    def find_peaks(self,
                   overlap=0.5,
                   **kwargs):
        """ This method takes a library of SymmetrySTEM Objects and finds peaks
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
            cluster_sym = [Cluster(x=cluster[1] * self.axes_manager.navigation_axes[2].scale,
                                   y=cluster[2] * self.axes_manager.navigation_axes[2].scale,
                                   radius=self.sigma[int(cluster[0])] * np.sqrt(2) * self.axes_manager.navigation_axes[2].scale,
                                   k=((cluster[3] * self.axes_manager.signal_axes[-1].scale) +
                                      self.axes_manager.signal_axes[-1].offset),
                                   symmetry=symmetry)
                           for cluster in clusters]
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

    def plot_clusters(self, ax=None, k_range=None):
        if ax is None:
            fig, ax = plt.subplots()
        extent = self.axes_manager.navigation_extent
        ax.set_xlim(extent[2]-5, extent[3]+5)
        ax.set_ylim(extent[4]-5, extent[5]+5)
        for symmetry, color in zip(self.clusters, colors[:len(self.clusters)]):
            for c in symmetry:
                if k_range is None or (c.k is None or (c.k <k_range[1] and c.k > k_range[0])):
                    ax.add_patch(c.to_circle(fill=True, color=color, alpha=0.5))
        from matplotlib.lines import Line2D
        leg = [Line2D([0], [0], marker='o', color=colors[i], label=str(sym) + " fold symmetry",
                      markerfacecolor=colors[i], markersize=15) for i, sym in enumerate(self.symmetries)]

        ax.legend(handles=leg, loc='upper right')
        return ax

    def get_cluster_size_distribution(self):
        radii = [[cluster.r for cluster in symmetry]for symmetry in self.clusters]
        return radii

    def plot_cluster_size_distribution(self,
                                       nbins=5,
                                       ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        size = self.get_cluster_size_distribution()
        leg = [str(sym)+" fold symmetry" for sym in self.symmetries]
        s_extent = (min(self.sigma) * np.sqrt(2) * self.axes_manager.navigation_axes[2].scale,
                    max(self.sigma) * np.sqrt(2) * self.axes_manager.navigation_axes[2].scale)
        histogram_s = np.array([np.histogram(s,
                                             nbins,
                                             s_extent) for s in size])
        for s, l in zip(histogram_s, leg):
            ax.plot(s[1][0:-1], s[0], label=l)
        ax.legend(loc='upper right')

    def get_k_range_distribution(self):
        k_range = [[cluster.k for cluster in symmetry]for symmetry in self.clusters]
        return k_range

    def plot_k_range_distribution(self,
                                  nbins=5,
                                  ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        k_range = self.get_k_range_distribution()
        s_extent = self.axes_manager.signal_extent
        histogram_k = np.array([np.histogram(ks,
                                             nbins,
                                             s_extent) for ks in k_range])
        leg = [str(sym)+" fold symmetry" for sym in self.symmetries]
        for k, l in zip(histogram_k, leg):
            ax.plot(k[1][0:-1], k[0], label=l)
        ax.legend(loc='upper right')


    def plot_cluster_stats(self, k_range=True, size=True, spatial=True):
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(3, 2)
        ax1 = fig.add_subplot(gs[:2, :2])
        ax1.set_title('Spatial Clusters')
        self.plot_clusters(ax=ax1)
        ax2 = fig.add_subplot(gs[2, 0])
        ax2.set_title('Size Distribution')
        ax3 = fig.add_subplot(gs[2:, 1])
        ax3.set_title('K Distribution')
        


