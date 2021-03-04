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

from pyxem.utils.correlation_utils import corr_to_power,\
    get_interpolation_matrix,\
    symmetry_stem
from pyxem.signals.polar_diffraction2d import PolarDiffraction2D
import numpy as np
from fractions import Fraction as frac
from hyperspy.api import stack


class Correlation2D(PolarDiffraction2D):
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

    def get_symmetry_stem(self,
                          symmetries=[1, 2,3, 4,5, 6,7, 8,9, 10],
                          angular_range=0,
                          method="average",
                          include_duplicates=False,
                          normalize=True,
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
        method: str
            One of max or average
        include_duplicates: bool
            Include duplicates like 2 and 4

        :return:
        """
        angles = [set(frac(j, i) for j in range(0, i))for i in symmetries]
        if not include_duplicates:
            already_used = set()
            new_angles = []
            for a in angles:
                new_angles.append(a.difference(already_used))
                already_used = already_used.union(a)
            angles = new_angles
        num_angles = [len(a) for a in angles]
        interp = [get_interpolation_matrix(a,
                                           angular_range,
                                           num_points=self.axes_manager.signal_axes[0].size,
                                           method=method)
                  for a in angles]
        signals = self.map(symmetry_stem,
                           interpolation=interp,
                           show_progressbar=True,
                           inplace=False,
                           method=method)
        if normalize & (method is not "max" or method is not "first"):
            signals = np.divide(signals, num_angles)
        # 3-D signal (x,y,k) for each symmetry
        signals = signals.transpose(navigation_axes=(0, 1, 2))
        signals.set_signal_type("symmetry")
        signals.symmetries = symmetries
        signals.axes_manager.navigation_axes[2].scale = 1
        signals.axes_manager.navigation_axes[2].name = "Symmetry"
        signals.axes_manager.navigation_axes[2].offset = 0
        return signals

    def get_blurred_library(self,
                            min_cluster_size=0.5,
                            max_cluster_size=5.0,
                            sigma_ratio=1.6,
                            k_sigma=2,
                            phi_sigma=2):
        gaussian_symmetry_stem = []
        if isinstance(k_sigma, float):
            k_sigma = k_sigma / self.axes_manager.signal_axes["k"].scale
        if isinstance(phi_sigma, float):
            k_sigma = phi_sigma / self.axes_manager.signal_axes["phi"].scale
        if isinstance(min_cluster_size, float):
            min_cluster_size = min_cluster_size / self.axes_manager["x"].scale
        if isinstance(max_cluster_size, float):
            max_cluster_size = max_cluster_size / self.axes_manager["x"].scale

        # k such that min_sigma*(sigma_ratio**k) > max_sigma
        k = int(np.mean(np.log(max_cluster_size / min_cluster_size) / np.log(sigma_ratio) + 1))

        # a geometric progression of standard deviations for gaussian kernels
        sigma_list = np.array([[min_cluster_size * (sigma_ratio ** i),
                                min_cluster_size * (sigma_ratio ** i),
                                phi_sigma,
                                k_sigma]
                               for i in range(k + 1)])

        for s in sigma_list:
            filtered = self.gaussian_filter(sigma=s, inplace=False)
            gaussian_symmetry_stem.append(filtered)

        print(sigma_list)
        dog_images = [(gaussian_symmetry_stem[i] - gaussian_symmetry_stem[i + 1]) * np.mean(sigma_list[i])
                      for i in range(k)]
        image_cube = stack(dog_images, axis=None)
        image_cube.axes_manager.navigation_axes[-1].name ="Sigma"
        image_cube.sigma = sigma_list[:, 0]
        return image_cube



class LazyCorrelation2D(LazySignal, Correlation2D):

    _lazy = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
