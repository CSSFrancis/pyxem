from hyperspy.utils.plot import plot_images
from hyperspy.signals import Signal2D
import numpy as np
from fractions import Fraction as frac

class Symmetry2D(Signal2D):
    _signal_type = "symmetry"

    @property
    def angles(self):
        try:
            return self.metadata.Signal["angles"]
        except (AttributeError):
            return None

    @angles.setter
    def angles(self, angles):
        self.metadata.set_item("Signal.angles", angles)

    def get_symmetry(self,
                     symmetry,
                     include_odd_duplicates=False):
        if self.angles is not None:
            if not include_odd_duplicates and symmetry!=2:
                if symmetry % 2 == 1:
                    sym_angles = [frac(i, symmetry) for i in range(1, symmetry) if i != int(symmetry / 2)]
                else:
                    sym_angles = [frac(i, symmetry) for i in range(1, symmetry, 2)if i != int(symmetry/2)]
            elif symmetry == 2:
                sym_angles = 0.5
            else:
                sym_angles = [frac(i, symmetry) for i in range(1, symmetry) if i != int(symmetry/2)]
            indexes = np.where(np.isin(self.angles, sym_angles))[0]
            ans = self.isig[indexes[0], :]
            if len(indexes) == 1:
                return ans
            else:
                for i in indexes[1:]:
                    ans = ans + self.isig[i, :]
                return ans
        else:
            raise ValueError("self.angles needs to be set before you can get the symmetry.")

    def plot_symmetries(self, symmetries=None,
                        include_odd_duplicates=False,
                        k_range=[4.0, 6.5],
                        **kwargs):
        """Plots the symmetry STEM for some symmetry STEM object.
         """
        if symmetries is None:
            symmetries = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        syms = [self.get_symmetry(sym, include_odd_duplicates).isig[k_range[0]:k_range[1]].sum(axis=-1).T for sym in symmetries]
        sym_labels = [str(s) + "-Fold Symmetry" for s in symmetries]
        plot_images(syms, label=sym_labels, **kwargs)

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