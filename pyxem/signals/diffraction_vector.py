from pyxem.decomposition.vector_decomposition import VectorDecomposition2D
import numpy as np
from skimage.morphology import flood





from hyperspy._signals.vector_signal import BaseVectorSignal
from hyperspy._signals.lazy import LazySignal
from pyxem.utils.vector_utils import get_extents as _get_extents
from pyxem.utils.vector_utils import refine, combine
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
        self.metadata.Vectors["extents"] = None
        self.metadata.Vectors["labels"] = None

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
                           output_signal_size=(),
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
    def labels(self):
        return self.metadata.Vectors.labels

    @labels.setter
    def labels(self, labels):
        self.metadata.Vectors.labels = labels

    def refine_positions(self,
                         data=None,
                         threshold=0.5,
                         inplace=False,
                         **kwargs,
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
                           img=data,
                           extents=self.extents,
                           threshold=threshold,
                           output_dtype=object,
                           ragged=True,
                           inplace=False,
                           output_signal_size=(),
                           **kwargs)
        refined.axes_manager = self.axes_manager
        refined.set_signal_type("vector")
        refined.vector = True
        print(refined)

        return refined

    def combine_vectors(self,
                        distance,
                        remove_duplicates=True,
                        symmetries=None,
                        structural_similarity=False,
                        inplace=False
                        ):
        labels = self.map(combine,
                           distance=distance,
                           remove_duplicates=remove_duplicates,
                           inplace=False,
                           output_dtype=object,
                           output_signal_size=(),
                           )

        self.labels = labels

        return


class LazyDiffractionVector(LazySignal,DiffractionVector):
    _lazy = True

