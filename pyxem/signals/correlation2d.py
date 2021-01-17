# -*- coding: utf-8 -*-
# Copyright 2016-2020 The pyXem developers
#
# This file is part of pyXem.
#
# pyXem is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyXem is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyXem.  If not, see <http://www.gnu.org/licenses/>.

"""Signal class for two-dimensional diffraction data in polar coordinates."""

from hyperspy.signals import Signal2D, BaseSignal
from hyperspy._signals.lazy import LazySignal
import hyperspy.drawing._markers.

from pyxem.utils.correlation_utils import corr_to_power,\
    get_interpolation_matrix,\
    symmetry_stem
from pyxem.signals.common_diffraction import CommonDiffraction
import numpy as np
from fractions import Fraction as frac
from skimage.feature import blob_dog, blob_log, blob_doh


class Correlation2D(Signal2D, CommonDiffraction):
    _signal_type = "correlation"

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

        self.decomposition.__func__.__doc__ = BaseSignal.decomposition.__doc__

    def get_angular_power(self, inplace=False, **kwargs):
        """Returns the power spectrum of the angular auto-correlation function
        in the form of a Signal2D class.

        This gives the fourier decomposition of the radial correlation. Due to
        nyquist sampling the number of fourier coefficients will be equal to the
        angular range.

        Parameters
        ----------
        mask: Numpy array or Signal2D
            A bool mask of values to ignore of shape equal to the signal shape.  If the mask
            is a BaseSignal than it is iterated with the polar signal
         normalize: bool
             Normalize the radial correlation by the average value at some radius.
        inplace: bool
            From hyperspy.signal.map(). inplace=True means the signal is
            overwritten.

        Returns
        -------
        power: Signal2D
            The power spectrum of the Signal2D
        """
        power = self.map(corr_to_power, inplace=inplace, **kwargs)
        if inplace:
            self.set_signal_type("power")
            fourier_axis = self.axes_manager.signal_axes[0]
        else:
            power.set_signal_type("power")
            fourier_axis = self.axes_manager.signal_axes[0]
        fourier_axis.name = "Fourier Coefficient"
        fourier_axis.units = "a.u"
        fourier_axis.offset = 0.5
        fourier_axis.scale = 1
        return power

    def get_summed_angular_power(self, inplace=False, **kwargs):
        """Returns the power spectrum of the summed angular auto-correlation function
        over all real space positions.  Averages the angular correlation.

        Parameters
        ----------
        inplace: bool
            From hyperspy.signal.map(). inplace=True means the signal is
            overwritten.

        Returns
        -------
        power: Power2D
            The power spectrum of summed angular correlation
        """
        power = self.sum().map(corr_to_power, inplace=inplace, **kwargs)
        if inplace:
            self.set_signal_type("power")
            fourier_axis = self.axes_manager.signal_axes[0]
        else:
            power.set_signal_type("power")
            fourier_axis = self.axes_manager.signal_axes[0]
        fourier_axis.name = "Fourier Coefficient"
        fourier_axis.units = "a.u"
        fourier_axis.offset = 0.5
        fourier_axis.scale = 1
        return power

    def find_clusters(self,
                      symmetries=[1,2,4,6,8,10],
                      angular_range=0,
                      k_range=None,
                      include_duplicates=False,
                      method="log",
                      **kwargs):
        """ This function is for finding and extracting information about clusters
        based on the angular symmetries. This a pretty catch all method which has
        a couple of different operating principles.

        If k_range=None the function uses a 3 dimensional version of blob finding and attempts to
        find blobs in the 3 dimensional space, x,y,k for each symmetry.

        Parameters
        ------------
        symmetries: list
            The symmetries to calculate
        include_duplicates: bool
            Include duplicates like 2 and 4
        method: one of "log", "dog" or "doh"
            The `skimage.features` method for finding blobs.
        :return:
        """
        method_dict = {"log": blob_log, "dog": blob_dog, "doh": blob_doh}
        angles = [set(frac(j, i) for j in range(0, i))for i in symmetries]
        if not include_duplicates:
            already_used = set()
            new_angles = []
            for a in angles:
                new_angles.append(a.difference(already_used))
                already_used = already_used.union(a)
            angles = new_angles
        interp = [get_interpolation_matrix(a,
                                           angular_range,
                                           num_points=self.axes_manager.signal_axes[0].size)
                  for a in angles]
        signals = self.map(symmetry_stem, interpolation=interp, show_progressbar=True, inplace=False)
        if k_range is None:
            # 3-D signal (x,y,k) for each symmetry
            signals = signals.transpose(navigation_axes=(2,))
        else:
            signals = signals.isig[:, k_range[0]:k_range[1]]
            signals = signals.T
        s = signals.map(method_dict[method],
                        **kwargs,
                        inplace=False)
        for i in np.ndindex(s.axes_manager.navigation_shape):
            blobs = s.inav[i].data
            [marker. indefor b in blobs]





    def get_symmetry_stem(self,
                          angular_range=0,
                          symmetry=10,
                          **kwargs,
                          ):
        """ Get the symmetry stem for some angular range.
        Parameters
        angular_range: float
            The range of angles to integrate over.
        symmetries: float
            The different symmetries to test
        duplicates: bool
            Remove any lower order which is repeated. ie. pi/2 would only be included in the 4
            fold intensity and not the 8 fold intensity"""
        angles = np.unique([frac(j, i) for i in range(1, max_symmetry+1) for j in range(0, i)])
        print(len(angles))
        interp = get_interpolation_matrix(angles,
                                          angular_range,
                                          num_points=self.axes_manager.signal_axes[0].size)
        print(len(interp))
        sym = self.map(symmetry_stem,  intepolation=interp, inplace=False)
        sym.set_signal_type("symmetry")
        sym.angles = angles
        return sym


class LazyCorrelation2D(LazySignal, Correlation2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
