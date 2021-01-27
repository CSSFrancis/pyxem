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
from hyperspy.api import stack
import numpy as np
from skimage.feature import blob_dog
from skimage.draw import circle
from scipy.ndimage import gaussian_filter as sci_gaussian_filter


from pyxem.utils.correlation_utils import _correlation, _power


class PolarDiffraction2D(Signal2D):
    _signal_type = "polar_diffraction"

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

    def get_angular_correlation(
        self, mask=None, normalize=True, inplace=False, **kwargs
    ):
        r"""Returns the angular auto-correlation function in the form of a Signal2D class.

        The angular correlation measures the angular symmetry by computing the self or auto
        correlation. The equation being calculated is
        $ C(\phi,k,n)= \frac{ <I(\theta,k,n)*I(\theta+\phi,k,n)>_\theta-<I(\theta,k,n)>^2}{<I(\theta,k,n)>^2}$

        Parameters
        ----------
        mask: Numpy array or Signal2D
            A bool mask of values to ignore of shape equal to the signal shape.  If the mask
            is a BaseSignal than it is iterated with the polar signal
        normalize: bool
            Normalize the radial correlation by the average value at some radius.
        kwargs: dict
            Any additional options for the hyperspy.BaseSignal.map() function
        inplace: bool
            From hyperspy.signal.map(). inplace=True means the signal is
            overwritten.

        Returns
        -------
        correlation: Signal2D
            The radial correlation for the signal2D

        Examples
        --------
        Basic example, no mask applied and normalization applied.
        >polar.get_angular_correlation()
        Angular correlation with a static matst for

        """
        if normalize:
            normalize_axes = 1
        else:
            normalize_axes=None
        correlation = self.map(
            _correlation,
            axis=1,
            mask=mask,
            normalize_axes=normalize_axes,
            inplace=inplace,
            **kwargs
        )
        if inplace:
            self.set_signal_type("correlation")
            correlation_axis = self.axes_manager.signal_axes[0]
        else:
            correlation.set_signal_type("correlation")
            correlation_axis = correlation.axes_manager.signal_axes[0]
        correlation_axis.name = "Angular Correlation, $/phi$"
        return correlation

    def get_angular_power(self, mask=None, normalize=True, inplace=False, **kwargs):
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
        power = self.map(
            _power, axis=1, mask=mask, normalize=normalize, inplace=inplace, **kwargs
        )
        if inplace:
            self.set_signal_type("power")
            fourier_axis = self.axes_manager.signal_axes[0]
        else:
            power.set_signal_type("power")
            fourier_axis = power.axes_manager.signal_axes[0]
        fourier_axis.name = "Fourier Coefficient"
        fourier_axis.units = "a.u"
        fourier_axis.offset = 0.5
        fourier_axis.scale = 1
        return power

    def get_common_signal(self,
                          method="multiply",
                          local_threshold=True,
                          **kwargs):
        """This function takes all of some group of signals and looks for common
        features among the signals.

        Parameters
        -----------
        method: ['multiply', 'sum']
            A method used to determine the common signal
        local_threshold: bool
            Apply a local threshold to better visualize the "Eigen Pattern"
        kwargs: dict
            Any additional arguments to pass to `skimage.filters.threshold_local`
        """
        if method is "multiply":
            self.unfold(unfold_navigation=True, unfold_signal=False)
            multiplied = np.prod(self.data, axis=0)
            blobs = blob_dog(multiplied, **kwargs)
            shape = np.shape(multiplied)
            mask = np.ones(shape=shape, dtype=bool)
            for x, y,sigma in blobs:
                rr, cc = circle(x, y, sigma*1.414, shape=shape)
                mask[rr, cc] = 0
            mean = self.mean().data
            mean[mask] = 0
            signal = self._deepcopy_with_new_data(data=mean)
            signal.axes_manager.remove(0)
            return signal
        elif method is "sum":
            return self.sum()
        else:
            print("Method " + method + " is not one of 'sum' ,'multiply', 'correlation' please "
                                       "use one of these methods")
            return

    def get_symmetry_stem_library(self,
                                  theta_size,
                                  k_size,
                                  min_cluster_size=1,
                                  max_cluster_size=10,
                                  sigma_ratio=1.6,
                                  mask=None,
                                  threshold=2.0,
                                  exclude_border=False,
                                  ):
        gaussian_symmetry_stem = []
        if isinstance(theta_size,float):
            theta_size =theta_size/self.axes_manager.signal_axes[1].scale
        if isinstance(k_size, float):
            k_size = k_size / self.axes_manager.signal_axes[0].scale
        if isinstance(min_cluster_size, float):
            min_cluster_size = min_cluster_size / self.axes_manager.navigation_axes[0].scale
        if isinstance(max_cluster_size, float):
            max_cluster_size = min_cluster_size / self.axes_manager.navigation_axes[0].scale

        # k such that min_sigma*(sigma_ratio**k) > max_sigma
        k = int(np.mean(np.log(max_cluster_size / min_cluster_size) / np.log(sigma_ratio) + 1))

        # a geometric progression of standard deviations for gaussian kernels
        sigma_list = np.array([[min_cluster_size * (sigma_ratio ** i),
                                min_cluster_size * (sigma_ratio ** i),
                                theta_size,
                                k_size]
                               for i in range(k + 1)])



        for s in sigma_list:
            filtered = self.gaussian_filter(sigma=s, inplace=False)
            filtered.get_angular_correlation(mask=mask, inplace=True)
            filtered = filtered.get_symmetry_stem()
            filtered.metadata.General["title"] = str(s) + " Sigma Sym STEM"
            gaussian_symmetry_stem.append(filtered)
            print(s)

        dog_images = [(gaussian_symmetry_stem[i] - gaussian_symmetry_stem[i + 1])
                      * np.mean(sigma_list[i]) for i in range(k)]

        image_cube = stack(dog_images, axis=None)
        image_cube = image_cube.split(axis=0)
        for i,sym in zip(image_cube,[1,2,4,6,8,10]):
            i.sigma = sigma_list
            i.symmetries = sym
        return image_cube

    def gaussian_filter(self,
                        sigma,
                        inplace=False):
        if inplace:
            self.data = sci_gaussian_filter(self.data,sigma)
        else:
            return self._deepcopy_with_new_data(data=sci_gaussian_filter(self.data,sigma))


    def speckle_filter(self,
                       sigmas,
                       mode='reflect',
                       cval=0,
                       inplace=True
                       ):
        from scipy.ndimage import gaussian_filter, sobel
        derivatives = [sobel(self.data, axis=i, mode=mode, cval=cval)
                       for i in range(2)]
        print(len(derivatives))
        derivatives = derivatives[0]*derivatives[1]
        A_elems = gaussian_filter(derivatives,
                                  sigmas,
                                  mode=mode,
                                  cval=cval)
        if inplace:
            self.data =A_elems
        else:
            return self._deepcopy_with_new_data(data=A_elems)


class LazyPolarDiffraction2D(LazySignal, PolarDiffraction2D):
    pass