from pyxem.decomposition.vector_decomposition import VectorDecomposition2D
import numpy as np
from skimage.morphology import flood





from hyperspy._signals.vector_signal import BaseVectorSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy.signals import BaseSignal
from pyxem.utils.vector_utils import (get_extents as _get_extents,
                                      _get_extents_lazy,
                                      _lazy_refine,
                                      refine_position as _refine_position,
                                      )
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


class DiffractionVector4D(BaseVectorSignal):
    """
    Diffraction Vector Object.  A collection of diffraction vectors.
    Extends the vector decomposition class
    """

    _signal_dimension = 4
    _signal_type = "diffraction_vectors"

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
        self.metadata.Vectors["slice"] = None

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
        if isinstance(img, da.Array):
            spanned = np.equal(img.chunks, img.shape)
            drop_axes = np.squeeze(np.argwhere(spanned))
            adjust_chunks = {}
            for i in range(len(img.shape)):
                if i not in drop_axes:
                    adjust_chunks[i] = 1
                else:
                    adjust_chunks[i] = -1
            pattern = np.squeeze(np.argwhere(np.logical_not(spanned)))
            from itertools import product
            offset = []
            for block_id in product(*(range(len(c)) for c in img.chunks)):
                offset.append(np.transpose([np.multiply(block_id, img.chunksize),
                              np.multiply(np.add(block_id, 1), img.chunksize)]))
            offset = np.array(offset, dtype=object)
            offset = np.reshape(offset, [len(c) for c in img.chunks] + [4, 2])
            offset = da.from_array(offset, chunks=(1,) * len(offset.shape))
            extents = da.reshape(da.blockwise(_get_extents_lazy,
                                              pattern,
                                              img,
                                              pattern,
                                              offset,
                                              [0, 1, 2, 3, 4,5],
                                              threshold=threshold,
                                              vectors=self.data,
                                              adjust_chunks=adjust_chunks,
                                              dtype=object,
                                              concatenate=True,
                                              align_arrays=False,
                                              **kwargs), (-1,))
            extents = extents.compute()
            extents = np.array([np.array(e, dtype=float) for extent in extents for e in extent], dtype=object)
            self.extents = extents
            self.slices = offset
        else:
            extents = _get_extents(img=img, vectors=self.data)
        return extents

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

    @property
    def slices(self):
        return self.metadata.Vectors.slices

    @slices.setter
    def slices(self, slices):
        self.metadata.Vectors.slices = slices

    def refine_positions(self,
                         img=None,
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
        if isinstance(img, da.Array):

            spanned = np.equal(img.chunks, img.shape)
            drop_axes = np.squeeze(np.argwhere(spanned))
            adjust_chunks = {}
            for i in range(len(img.shape)):
                if i not in drop_axes:
                    adjust_chunks[i] = 1
                else:
                    adjust_chunks[i] = -1

            pattern = np.squeeze(np.argwhere(np.logical_not(spanned)))
            from itertools import product
            offset = []
            for block_id in product(*(range(len(c)) for c in img.chunks)):
                offset.append(np.transpose([np.multiply(block_id, img.chunksize),
                              np.multiply(np.add(block_id, 1), img.chunksize)]))
            offset = np.array(offset, dtype=object)
            offset = np.reshape(offset, [len(c) for c in img.chunks] + [4, 2])
            offset = da.from_array(offset, chunks=(1,) * len(offset.shape))
            ref = da.reshape(da.blockwise(_lazy_refine,
                               pattern,
                               img, pattern,
                               offset, [0, 1, 2, 3, 4,5],
                               vectors=self.data,
                               vdf=self.extents,
                               threshold=threshold,
                               adjust_chunks=adjust_chunks,
                               dtype=object,
                               concatenate=True,
                               align_arrays=False,
                               **kwargs), (-1,))
            ref = ref.compute()
            refined = np.vstack([p for p in ref if p is not None])
            if inplace:
                self.data = refined
            else:
                refined = self._deepcopy_with_new_data(data=refined)
                refined.extents = self.extents
        else:
            refined = _refine_position(self.data,
                                       data=img,
                                       threshold=threshold,
                                       **kwargs)
            if inplace:
                self.data = refined
            else:
                refined = self._deepcopy_with_new_data(data=refined)
                refined.extents = self.extents
        return refined

    def combine_vectors2(self,
                        distance,
                        duplicate_distance=None,
                        trim=True,
                        symmetries=None,
                        structural_similarity=False,
                        ):

        labels = combine_vectors(self.data,
                                 distance=distance,
                                 duplicate_distance=duplicate_distance,
                                 symmetries=symmetries,
                                 structural_similarity=structural_similarity)
        #self.labels = labels
        clusters = np.array([np.array(self.data[labels == l],dtype=float)
                             for l in range(0, max(labels))],dtype=object)
        extents = np.array([self.extents[labels == l]
                   for l in range(0, max(labels))], dtype=object)
        s = BaseSignal(clusters)
        s = s.T
        s.vector = True
        s.set_signal_type("diffraction_vector")
        s.extents = extents
        for ax_new, ax_old in zip(s.axes_manager.signal_axes,
                                  self.axes_manager.signal_axes):
            ax_new.scale = ax_old.scale
            ax_new.units = ax_old.units
            ax_new.offset = ax_old.offset
            ax_new.name = ax_old.name
        s.axes_manager.navigation_axes[0].name = "label"
        return s

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
            new_labels = labels.map(trim_duplicates, label=labels, inplace=False)
            am = self.axes_manager.deepcopy()
            new_extents = self.extents.map(trim_duplicates,label=self.extents, inplace=False)
            self.map(trim_duplicates, label=labels)
            self.axes_manager = am
            self.set_signal_type("vector")
            self.axes_manager._ragged = True
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


class LazyDiffractionVector(LazySignal,DiffractionVector4D):
    _lazy = True

