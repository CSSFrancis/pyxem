from pyxem.decomposition.vector_decomposition import VectorDecomposition2D
import numpy as np
from skimage.morphology import flood





from hyperspy._signals.vector_signal import BaseVectorSignal
from hyperspy._signals.lazy import LazySignal
from hyperspy.signals import BaseSignal
from pyxem.utils.vector_utils import get_extents as _get_extents
from pyxem.utils.vector_utils import refine, combine_vectors, trim_duplicates, get_vectors_chunkwise, get_chunk_offsets
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

    def vector_signal_map(self,
                          func,
                          signal,
                          cleanup_func=None,
                          extra_vectors=None,
                          **kwargs
                          ):

        """ A generic function to be applied to a set of vectors and a signal.  The vectors are assumed to be
        related to the signal and the function is applied to the signal where every vector which is inside of
        some chunk is applied to that `chunk`.

        Extra Vectors can be passed which are signals with dtype object and the same length as `self`.

        """
        if not signal._lazy:
            signal = signal.as_lazy()
            signal.rechunk(nav_chunks=signal.axes_manager.navigation_shape)
        spanned = [c == s or c == (s,) for c, s in zip(signal.data.chunks, signal.data.shape)]

        drop_axes = np.squeeze(np.argwhere(spanned))
        adjust_chunks = {}
        for i in range(len(signal.data.shape)):
            if i not in drop_axes:
                adjust_chunks[i] = 1
            else:
                adjust_chunks[i] = -1
        pattern = np.argwhere(np.logical_not(spanned))[:, 0]
        if len(pattern) == 0:
            pattern = [True, ]
        offsets = get_chunk_offsets(signal.data)
        offsets = da.from_array(offsets, chunks=(1,)*len(pattern)+(-1, -1))

        new_args = (signal.data, range(len(signal.data.shape)))

        if extra_vectors is not None:
            vectors, extra = get_vectors_chunkwise(self.data,
                                                   offsets=offsets,
                                                   extra_vectors=[extra_vectors,]
                                                   )

            new_args += (vectors, pattern)
            for e in extra:
                new_args += (e, pattern)
        else:
            vectors = get_vectors_chunkwise(self.data,
                                            offsets=offsets)
            new_args += (vectors, pattern)

        # Applying the function blockwise
        new_args += (offsets, range(len(offsets.shape)))
        ref = da.reshape(da.blockwise(func, pattern,
                                      *new_args,
                                      adjust_chunks=adjust_chunks,
                                      dtype=object,
                                      concatenate=True,
                                      align_arrays=False,
                                      **kwargs),
                         (-1,)
                         )
        ref = ref.compute()
        ref = np.vstack(r for r in ref if r is not None)
        return ref

    def get_extents(self,
                    img,
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
        extents = self.vector_signal_map(_get_extents,
                                         img,
                                         **kwargs)
        self.extents = extents

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

    def refine_position(self, img, inplace=False, **kwargs):
        """Refine the position of the features based on an image.  Uses the VDF to
        find the center rather than the maximum."""
        refined = self.vector_signal_map(refine, img, extra_vectors=self.extents, **kwargs)
        if inplace:
            self.data = refined
        else:
            refined = self._deepcopy_with_new_data(data=refined)
            refined.extents = self.extents

        return refined

    def combine_vectors(self,
                        distance,
                        duplicate_distance=None,
                        structural_similarity=None,
                        cross_correlation=None,
                        **kwargs
                        ):

        labels = combine_vectors(self.data,
                                 distance=distance,
                                 duplicate_distance=duplicate_distance,
                                 extents=self.extents,
                                 ss=structural_similarity,
                                 cc=cross_correlation,
                                 **kwargs)
        clusters = np.array([np.array(self.data[labels == l], dtype=float)
                             for l in range(0, max(labels))], dtype=object)
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

    def get_separation(self):
        """Return the separation between diffraction vectors that are clustered in space"""
        separation = []
        for v in self.data:
            angles = v[:, 3]
            all_angles = np.abs(np.unique(np.triu(np.subtract.outer(angles, angles))) * 4)
            all_angles = all_angles[all_angles != 0]
            all_angles[all_angles > 180] = np.abs(all_angles[all_angles > 180] - 360)
            separation.append(all_angles)
        separation = [s for sep in separation for s in sep]
        return separation




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

