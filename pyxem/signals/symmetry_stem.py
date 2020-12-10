from hyperspy.utils.plot import plot_images

class SymmetrySTEM(dict):
    def __init__(self, *args, **kwargs):
        """Create a PolarDiffraction2D object from a numpy.ndarray.

        Parameters
        ----------
        *args :
            Passed to the __init__ of Signal2D. The first arg should be
            a numpy.ndarray
        **kwargs :
            Passed to the __init__ of Signal2D
        """
        super().__init__(*args, **kwargs)

    def plot(self, symmetries=None, **kwargs):
        """Plots the symmetry STEM for some symmetry STEM objct.
         """
        plot_images(self[symmetries],**kwargs)

    def simple_separate(self,
                        even=True,
                        unique=False):
        """ This function separates the symmetries by decomposing them
        allowing for better and more faithful reconstruction of the symmetries

        Parameters
        even: bool
            Only account for even symmetries.  This
        """
        pass

    def cluster
