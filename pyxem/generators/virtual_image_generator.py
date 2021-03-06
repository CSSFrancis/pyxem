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

"""VDF generator and associated tools."""
import numpy as np

import hyperspy.api as hs

from pyxem.utils.virtual_images_utils import normalize_virtual_images


NORMALISE_DOCSTRING = """normalize : boolean
            If True each VDF image is normalized so that the maximum intensity
            in each VDF is 1.
        """


class VirtualImageGenerator:
    """Generates virtual images for a specified signal and set of aperture.

    Attributes
    ----------
    signal : Diffraction2D or subclass
        The signal of electron diffraction patterns to be indexed.

    """

    def __init__(self, signal, *args, **kwargs):
        self.signal = signal

    def get_concentric_virtual_images(self, k_min, k_max, k_steps,
                                      normalize=False):
        """Obtain the intensity scattered at each navigation position in an
        Diffraction2D Signal by summation over a series of concentric
        in annuli between a specified inner and outer radius in a number of
        steps.

        Parameters
        ----------
        k_min : float
            Minimum radius of the annular integration window in units of
            reciprocal space.

        k_max : float
            Maximum radius of the annular integration window in units of
            reciprocal space.

        k_steps : int
            Number of steps within the annular integration window
        %s

        Returns
        -------
        virtual_images : VDFImage
            VDFImage object containing virtual images for all steps
            within the annulus.
        """
        k_step = (k_max - k_min) / k_steps
        k0s = np.linspace(k_min, k_max - k_step, k_steps)
        k1s = np.linspace(k_min + k_step, k_max, k_steps)

        ks = np.array((k0s, k1s)).T

        roi_args_list = [(0, 0, r[1], r[0]) for r in ks]

        new_axis_dict = {'name': 'Annular bins',
                         'scale': k_step,
                         'units': self.signal.axes_manager[-1].units,
                         'offset': k_min}

        return self._get_virtual_images(roi_args_list, normalize,
                                        new_axis_dict=new_axis_dict)

    get_concentric_virtual_images.__doc__ %= (NORMALISE_DOCSTRING)

    def _get_virtual_images(self, roi_args_list, normalize, new_axis_dict):
        """Obtain the intensity scattered at each navigation position in an
        Diffraction2D Signal by summation over the roi defined by the
        ``roi_args_list`` parameter.

        Parameters
        ----------
        roi_args_list : list of `hyperspy.roi.CircleROI` arguments
            Arguments required to initialise a CircleROI
        %s

        Returns
        -------
        virtual_images : VDFImage
            VDFImage object containing the virtual images
        """
        vdfs = [
            self.signal.get_integrated_intensity(hs.roi.CircleROI(*roi_args))
            for roi_args in roi_args_list
            ]

        vdfim = hs.stack(vdfs, new_axis_name=new_axis_dict['name'],
                         show_progressbar=False)

        vdfim.set_signal_type("virtual_dark_field")

        # Set new axis properties
        new_axis = vdfim.axes_manager[new_axis_dict['name']]
        for k, v in new_axis_dict.items():
            setattr(new_axis, k, v)

        if normalize:
            vdfim.map(normalize_virtual_images)

        return vdfim

    _get_virtual_images.__doc__ %= (NORMALISE_DOCSTRING)


class VirtualDarkFieldGenerator(VirtualImageGenerator):
    """Generates VDF images for a specified signal and set of aperture
    positions.

    Attributes
    ----------
    signal : Diffraction2D of subclass
        The signal of diffraction patterns to be indexed.
    vectors: DiffractionVectors
        The vector positions, in calibrated units, at which to position
        integration windows for virtual dark field formation.

    """

    def __init__(self, signal, vectors, *args, **kwargs):
        super().__init__(signal)
        if len(vectors.axes_manager.signal_axes) == 0:
            unique_vectors = vectors.get_unique_vectors(*args, **kwargs)
        else:
            unique_vectors = vectors

        self.vectors = unique_vectors

    def get_virtual_dark_field_images(self, radius, normalize=False):
        """Obtain the intensity scattered to each diffraction vector at each
        navigation position in an Diffraction2D Signal by summation in a
        circular window of specified radius.

        Parameters
        ----------
        radius : float
            Radius of the integration window - in units of the reciprocal
            space.
        %s

        Returns
        -------
        vdfs : VDFImage
            VDFImage object containing virtual dark field images for all unique
            vectors.
        """
        roi_args_list = [(v[0], v[1], radius, 0) for v in self.vectors.data]
        new_axis_dict = {'name': 'Vector index'}
        vdfim = self._get_virtual_images(roi_args_list, normalize,
                                         new_axis_dict=new_axis_dict)

        # Assign vectors used to generate images to vdfim attribute.
        vdfim.vectors = self.vectors

        return vdfim

    get_virtual_dark_field_images.__doc__ %= (NORMALISE_DOCSTRING)
