from hyperspy.utils.plot import plot_images
from hyperspy.signals import Signal2D
import numpy as np

class SymmetrySTEM(Signal2D):
    def __init__(self, *args, **kwargs):
        """Create a PolarDiffraction2D object from a numpy.ndarray.
        *** Note that this class should change when hyperspy allows for non-uniform axes.***

        Parameters
        ----------
        *args :
            Passed to the __init__ of Signal2D. The first arg should be
            a numpy.ndarray
        **kwargs :
            Passed to the __init__ of Signal2D
        """
        super().__init__(*args, **kwargs)

    @angles.setter
    def set_angles(self, angles):
        self.metadata.set_item("Signal.angles", angles)

    @property
    def angles(self):
        try:
            return self.metadata.Signal["angles"]
        except (AttributeError):
            return None

    def get_symmetry(self,
                     symmetry,
                     include_odd_duplicates=False):
        if self.angles is not None:
            if not include_odd_duplicates:
                sym_angles = [i / symmetry for i in range(1, symmetry, 2)if i != int(symmetry/2)]
            else:
                sym_angles = [i / symmetry for i in range(1, symmetry) if i != int(symmetry/2)]
            indexes = np.where(np.isin(self.angles, sym_angles))
            return self.isig[indexes, :].sum(axis=1)
        else:
            raise ValueError("self.angles needs to be set before you can get the symmetry.")

    def plot_symmetries(self, symmetries=None, **kwargs):
        """Plots the symmetry STEM for some symmetry STEM object.
         """
        plot_images(self[symmetries], **kwargs)

    def simple_separate(self,
                        even=True,
                        unique=False):
        """ This function separates the symmetries by decomposing them
        allowing for better and more faithful reconstruction of the symmetriesc

        Parameters
        even: bool
            Only account for even symmetries.  This
        """
        pass