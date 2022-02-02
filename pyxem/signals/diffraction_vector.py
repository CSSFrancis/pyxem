from pyxem.decomposition.vector_decomposition import VectorDecomposition2D
import numpy as np
from skimage.morphology import flood
from sklearn.cluster import AgglomerativeClustering

from scipy.spatial import distance_matrix


from hyperspy._signals.vector_signal import BaseVectorSignal
from hyperspy._signals.lazy import LazySignal
from pyxem.utils.vector_utils import get_extents as _get_extents
from pyxem.utils.vector_utils import refine
from hyperspy.axes import create_axis




def center_and_crop(image, center):
    extent = np.argwhere(image > 0)
    if len(extent) == 0:
        return [], tuple([slice(None) for c in center])
    else:
        rounded_center = np.array(np.round(center), dtype=int)
        max_values = np.max(extent, axis=0)-rounded_center
        min_values = rounded_center-np.min(extent, axis=0)
        max_extent = np.max([max_values, min_values])
        slices = tuple([slice(c-max_extent, c+max_extent+1) for c in rounded_center])
        return image[slices], slices


class DiffractionVector(BaseVectorSignal):
    """
    Diffraction Vector Object.  A collection of diffraction vectors.
    Extends the vector decomposition class
    """
    def __init__(self,
                 data,
                 **kwds):
        """

        Parmeters
        ---------
        navigation_components: list
            The list of navigation components.  If navigation components is
            a vector then each component is a vector of dimension equal
            to the navigation dimensions. Otherwise the navigation component
            spans the navigation dimension.

        signal_components:
            The list of signal components.  If signal components is
            a vector then each component is a vector of dimension equal
            to the signal dimensions. Otherwise the signal component
            spans the navigation dimension.
        """
        super().__init__(data, **kwds)
        self.metadata.add_node("Vectors")
        self.metadata.Vectors["extents"] = np.empty(self.data.shape, dtype=object)
        self.metadata.Vectors["slices"] = np.empty(self.data.shape, dtype=object)

    def get_extents(self,
                    img,
                    threshold=0.9,
                    **kwargs):
        """Get the extent of each diffraction vector using some dataset.  Finds the extent given
        some seed point and a threshold.

        Parameters
        ----------
        data: Array-like
            An n-dimensional array to use to determine the extent of the vector.
            Should have the same dimensions as one of the diffraction vectors.
        threshold: float
            The relative threshold to use to determine the extent of some feature.
        inplace: bool
            Return a copy of the data or act on the dataset
        crop: bool
            Crop the vdf to only be the extent of the vector.  Saves on memory if you have a large
            number of vectors.
        **kwargs:
            radius: The radius of to use to create the vdf
            search: The area around the center point to look to higher values.
        """
        extents = img.map(_get_extents,
                           vectors=self,
                           threshold=threshold,
                           inplace=False,
                           output_dtype=object,
                           ragged=True,
                           **kwargs)
        if len(extents.axes_manager.navigation_axes)==0:
            ax = (create_axis(size=1, scale=1, offset=0),)
            extents.axes_manager.navigation_axes = ax
        self.extents = extents
        return


    @property
    def extents(self):
        return self.metadata.Vectors.extents

    @extents.setter
    def extents(self, extents):
        self.metadata.Vectors.extents = extents

    @property
    def slices(self):
        return self.metadata.Vectors.slices

    @slices.setter
    def slices(self, slices):
        self.metadata.Vectors.slices = slices

    def refine_positions(self,
                         data=None,
                         threshold=0.5,
                         inplace=False,
                         ):
        """Refine the position of the diffraction vector the signal space

        Parameters
        ----------
        data: Array-like
            An n-dimensional array to use to determine the extent of the vector.
            Should have the same dimensions as one of the diffraction vectors.
        threshold: float
            The relative threshold to use to determine the extent of some feature.
        inplace: bool
            Return a copy of the data or act on the dataset
        crop: bool
            Crop the vdf to only be the extent of the vector.  Saves on memory if you have a large
            number of vectors.
        """
        refined = self.map(refine,
                           data=data,
                           extents=self.extents,
                           threshold=threshold,
                           output_dtype=object,
                           inplace=False)

        return refined

    def combine_vectors(self,
                        distance,
                        remove_duplicates=True,
                        symmetries=None,
                        structural_similarity=False,
                        inplace=False
                        ):
        if inplace:
            vectors = self
        else:
            vectors = self.deepcopy()
        agg = AgglomerativeClustering(n_clusters=None, distance_threshold=distance)
        agg.fit(vectors.vectors[:, :2])
        labels = agg.labels_
        new_vectors = []
        new_labels = []
        new_extents = []
        vectors.extents = np.array(vectors.extents)
        for l in range(max(labels)):
            grouped_vectors = vectors.vectors[labels == l]
            extents = vectors.extents[labels == l]
            if remove_duplicates:
                dist_mat = distance_matrix(grouped_vectors[:, 2:], grouped_vectors[:, 2:]) < 5
                is_first = np.sum(np.tril(dist_mat), axis=1) == 1
                grouped_vectors = grouped_vectors[is_first]
                extents = extents[is_first]
            for v, e in zip(grouped_vectors, extents):
                new_vectors.append(v)
                new_labels.append(l)
                new_extents.append(e)

        vectors.data = np.array(new_vectors)
        vectors.labels = np.array(new_labels)
        vectors.extents = np.array(new_extents)

        return vectors


class LazyDiffractionVector(LazySignal,DiffractionVector):
    _lazy = True

def spatial_cluster(vectors,distance,**kwargs):
    agg = AgglomerativeClustering(n_clusters=None, distance_threshold=distance)
    agg.fit(vectors)
    labels = agg.labels_
    return labels