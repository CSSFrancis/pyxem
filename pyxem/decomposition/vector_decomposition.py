import numpy as np
from hyperspy.axes import AxesManager
import copy
from hyperspy.misc.slicing import SpecialSlicers, FancySlicing
from hyperspy.signal import BaseSignal
import matplotlib.pyplot as plt


class Vectors:
    def __init__(self, array):
        self.array = array
        self.shape = array.shape

    def __array__(self, dtype=None):
        return self.array

    def __getitem__(self, items):
        print(items)
        if isinstance(items, tuple):
            slices = [[s.start, s.stop] for s in items]
            for i, s in enumerate(slices):
                if s[0] is None:
                    slices[i][0] = -np.inf
                if s[1] is None:
                    slices[i][1] = np.inf
            is_returned = np.prod([np.multiply(self.array[:, i] >= s[0],
                                               self.array[:, i] <= s[1])
                                   for i, s in enumerate(slices)], axis=0)
        new_vectors = self.array[np.where(is_returned)]
        return new_vectors


class VectorDecomposition2D(BaseSignal):
    """ The decomposition class is a vectorized representation of some signal.

    This means that the a representation of the signal can be recreated from
    the sum of the matrix multiplication of each component.

    For pure vector representations this

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
        if not isinstance(data,Vectors):
            vectors = Vectors(np.array(data))

        super().__init__(data, **kwds)
        print(self.data)
        self.metadata.add_node("Vectors")
        self.metadata.Vectors["labels"] = np.empty(len(vectors.array))
        self.metadata.Vectors["extents"] = np.empty(len(vectors.array), dtype=object)
        self.metadata.Vectors["slices"] = np.empty(len(vectors.array), dtype=object)

        if "axes" not in kwds:
            axes = [{"size": 1} for a in range(np.shape(vectors)[1])]
            self.axes_manager = AxesManager(axes)
            if self.axes_manager.signal_dimension != 2:
                self.axes_manager.set_signal_dimension(2)

    def deepcopy(self):
        new = super().deepcopy()
        new.metadata.Vectors["labels"] = self.labels
        new.metadata.Vectors["extents"] = self.extents
        new.metadata.Vectors["slices"] = self.slices
        return new


    def __repr__(self):
        return str("<Collection of: " + str(len(self.data))
                   + "vectors>")

    def __getitem__(self, item):
        new_data = self.data[item]
        new_extents = self.extents[item]
        new_labels = self.labels[item]
        new_slices = self.slices[item]
        new = self._deepcopy_with_new_data(Vectors(new_data))
        new.labels = new_labels
        new.extents = new_extents
        new.slices = new_slices
        new.axes_manager = self.axes_manager
        return new


    @property
    def labels(self):
        return self.metadata.Vectors.labels

    @labels.setter
    def labels(self, labels):
        self.metadata.Vectors.labels=labels

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
    @property
    def vectors(self):
        return self.data

    @property
    def real_space_vectors(self):
        scales = [a.scale for a in self.axes_manager._axes]
        offsets = [a.offset for a in self.axes_manager._axes]
        return np.add(offsets, np.multiply(self.data, scales))

    def label_vectors(self, method, **kwargs):
        """Using a pre-defined or custom method.  The vectors
        are assigned a label which is used to group the different
        vector.

        :param method:
        :return:
        """

        return

    def refine_labels(self, method):
        return

    def __call__(self, *args, **kwargs):
        """Slice some signal target using a set of vectors.

        The extent of the signal which is sliced """
        return

    def get_label(self, label):
        return

    def plot_label(self, label):
        return

    def plot(self):
        """Plots the paired navigation and signal components
        """
        return

    def to_signal(self, lazy=False):
        """Returns the signal created from the decomposition.  If lazy then the signal is
        only calculated when necessary.
        """
        return

    def plot_navigation(self):
        nav_indexes = self.axes_manager.navigation_indices_in_array
        navigation = self.data[:, nav_indexes]
        plt.figure()
        plt.scatter(navigation[:, 0], navigation[:, 1])
        return

    def plot_signal(self):
        sig_indexes = self.axes_manager.signal_indices_in_array
        signal = self.data[:, sig_indexes]
        plt.figure()
        plt.scatter(signal[:, 0], signal[:, 1])
        return

    def add_to_signal(self):
        return

    def sum(self, axis):
        """Sum of the data along some axis. If the data is vectorized then this gives the number of
        vectors for every pixel"""
        return

    def mean(self, axis):
        """mean of the data along some axis. If the data is vectorized then this gives the
        average number of vectors for every pixel"""
        return
