from pyxem.decomposition.vector_decomposition import VectorDecomposition2D
import numpy as np
from skimage.morphology import flood





from hyperspy._signals.vector_signal import BaseVectorSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy.signals import BaseSignal
from pyxem.utils.vector_utils import get_extents as _get_extents
from pyxem.utils.vector_utils import refine, combine_vectors,trim_duplicates
from hyperspy.axes import create_axis
import dask.array as da



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
                    chunkwise=False,
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
                           data=data,
                           extents=self.extents,
                           threshold=threshold,
                           output_dtype=object,
                           ragged=True,
                           inplace=inplace, **kwargs)
        if not inplace:
            if self._lazy:
                refined = LazyDiffractionVector(refined.data)
            else:
                refined = DiffractionVector(refined.data)
            refined.axes_manager = self.axes_manager.deepcopy()
            refined.vector = True
            refined.axes_manager._ragged = True
            refined.extents = self.extents
        return refined

    def combine_vectors(self,
                        distance,
                        duplicate_distance=None,
                        trim=True,
                        symmetries=None,
                        structural_similarity=False,
                        ):

        labels = self.map(combine_vectors,
                          distance=distance,
                          duplicate_distance=duplicate_distance,
                          symmetries=symmetries,
                          structural_similarity=structural_similarity,
                          ragged=True,
                          inplace=False)
        self.labels = labels

        if trim:
            new_labels = labels.map(trim_duplicates,label=labels, inplace=False)
            am = self.axes_manager.deepcopy()
            new_extents = self.extents.map(trim_duplicates,label=self.extents, inplace=False)
            self.map(trim_duplicates, label=labels)
            self.axes_manager = am
            self.set_signal_type("vector")
            self.axes_manager._ragged=True
            self.labels = new_labels
            self.extents = new_extents

        else:
            self.labels = labels

        return

    """def to_polar_markers(self,
                         index,
                         x_axis=(-2,),
                         y_axis=(-1,),
                         **kwargs):
        """""" Converts the vector to a set of markers given the axes.

        Parameters
        ----------
        x_axis: int
            The index for the x axis
        y_axis: int
            The index for the y axis
        """"""
        azim = self.get_real_vectors(axis=(1,))
        rad = self.get_real_vectors(axis=(2,))
        x = [np.sin(a)for a, r in zip(azim, rad)]
        y = [np.cos(a) for a, r in zip(azim, rad)]
        [x[labels=l] for l in range(max(self.labels)):
            

        if isinstance(x_axis, int):
            x_axis = (x_axis,)
        if isinstance(y_axis, int):
            y_axis = (y_axis,)
        x_vectors = self.get_real_vectors(axis=x_axis).T
        print(x_vectors[0])
        y_vectors = self.get_real_vectors(axis=y_axis).T
        return Point(x_vectors, y_vectors,**kwargs)"""

    def plot_patterns(self,):
        pass


class LazyDiffractionVector(LazySignal,DiffractionVector):
    _lazy = True

