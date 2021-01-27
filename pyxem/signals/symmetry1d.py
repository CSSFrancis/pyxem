from hyperspy.utils.plot import plot_images
from hyperspy.signals import Signal1D
import hyperspy.api as hs

import matplotlib.pyplot as plt
import numpy as np
from pyxem.utils.cluster_roi import Cluster
from pyxem.utils.correlation_utils import blob_finding
from skimage.feature.peak import peak_local_max
from skimage.feature.blob import _prune_blobs


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
        self.transpose()
        for s in self.symmetries:
            local_maxima = peak_local_max(self.data, **kwargs)
            # Catch no peaks
            if local_maxima.size == 0:
                return np.empty((0, 3))
                # Convert local_maxima to float64
            lm = local_maxima.astype(np.float64)

            # translate final column of lm, which contains the index of the
            # sigma that produced the maximum intensity value, into the sigma
            sigmas_of_peaks = self.sigma[local_maxima[:, 0]]
            # Remove sigma index and replace with sigmas
            lm = np.hstack([lm[:, :-1], sigmas_of_peaks])
            pruned = _prune_blobs(lm, overlap, sigma_dim=3)
            self.clusters.append([Cluster(x=cluster[0] * self.axes_manager.navigation_axes[-1].scale,
                                          y=cluster[1] * self.axes_manager.navigation_axes[-1].scale,
                                          radius=cluster[3] * np.sqrt(2) * self.axes_manager.navigation_axes[-1].scale,
                                          k=cluster[2 * self.axes_manager.signal_axes[-1].scale],
                                          symmetry=s)
                                  for cluster in pruned])

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
                    print(ax)
                    for cluster in clusters:
                        ax.add_patch(cluster.to_circle())


