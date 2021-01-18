from hyperspy.utils.plot import plot_images
from hyperspy.signals import Signal1D
import hyperspy.api as hs
import numpy as np
from pyxem.utils.cluster_roi import Cluster
from pyxem.utils.correlation_utils import blob_finding


class Symmetry1D(Signal1D):
    _signal_type = "symmetry"

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.clusters = None

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
        print(s)
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
                cluster_list.append([Cluster(x=cluster[0],
                                             y=cluster[1],
                                             radius=cluster[3] * np.sqrt(2),
                                             k=cluster[2],
                                             symmetry=symmetry)
                                     for cluster in clusters])
            else:
                cluster_list.append([Cluster(x=cluster[0],
                                             y=cluster[1],
                                             radius=cluster[3] * np.sqrt(2),
                                             k=k_range,
                                             symmetry=symmetry)
                                     for cluster in clusters])
        self.clusters = cluster_list
        return cluster_list

    def plot_all(self,
                 k_range,
                 include_clusters=True):
        signals = self.isig[k_range[0]:k_range[1]].sum(axis=-1).split()
        plot_images(signals)



