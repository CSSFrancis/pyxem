import numpy as np
from hyperspy.axes import AxesManager
import copy
from hyperspy.misc.slicing import SpecialSlicers, FancySlicing
import matplotlib.pyplot as plt


class Vectors:
    def __init__(self, array):
        self.array = array
        self.shape = ((2), (2), (2), (2))

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


class VectorDecomposition2D(FancySlicing):
    """ The decomposition class is a vectorized representation of some signal.

    This means that the a representation of the signal can be recreated from
    the sum of the matrix multiplication of each component.

    For pure vector representations this

    """
    def __init__(self,
                 vectors,
                 metadata=None,
                 axes=None):
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
        self.data = Vectors(vectors)
        self.isig = SpecialSlicers(self, isNavigation=False)
        self.inav = SpecialSlicers(self, isNavigation=True)
        self.labels = None
        if axes is None:
            axes = [{"size": 1} for a in range(np.shape(vectors)[1])]
        self.axes_manager = AxesManager(axes)
        if self.axes_manager.signal_dimension != 2:
            self.axes_manager.set_signal_dimension(2)

    def __repr__(self):
        return str("<Collection of: " + str(len(self.data.array))
                   + "vectors>")

    @property
    def vectors(self):
        return self.data.array

    @property
    def real_space_vectors(self):
        scales = [a.scale for a in self.axes_manager._axes]
        offsets = [a.offset for a in self.axes_manager._axes]
        print(offsets)
        return np.add(offsets, np.multiply(self.data.array, scales))

    def _deepcopy_with_new_data(self, new_data, **kwargs):
        copied = copy.deepcopy(self)
        copied.data = Vectors(new_data)
        return copied

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
        navigation = self.data.array[:, nav_indexes]
        plt.figure()
        plt.scatter(navigation[:, 0], navigation[:, 1])
        return

    def plot_signal(self):
        sig_indexes = self.axes_manager.signal_indices_in_array
        signal = self.data.array[:, sig_indexes]
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

    def save(self):
        """Saves the data using the .hspy format as a decomposition"""
        return
