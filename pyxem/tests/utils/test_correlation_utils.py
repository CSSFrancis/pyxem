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

import pytest
import numpy as np

from pyxem.utils.correlation_utils import _correlation,wrap_set_float, get_interpolation_matrix


class TestCorrelations:
    @pytest.fixture
    def ones_array(self):
        return np.ones((10, 20))

    @pytest.fixture
    def ones_zero(self):
        ones = np.ones((10, 20))
        ones[0:20:2, :] = 0
        return ones

    @pytest.fixture
    def random_array(self):
        rand_array = np.random.rand(10, 20)
        return rand_array

    @pytest.fixture
    def ones_hundred(self):
        ones = np.ones((10, 20))
        ones[0:20:2, :] = 100
        return ones

    def test_correlation_ones(self, ones_array):
        c = _correlation(ones_array, normalize_axes=1)
        np.testing.assert_array_equal(c, np.zeros((10, 20)))

    def test_correlation_random(self, random_array):
        c = _correlation(random_array,normalize_axes=0)
        c2 = _correlation(random_array*10, normalize_axes=0)
        ra_size = random_array.shape
        ra_size = (ra_size[0], ra_size[1]*2)
        random_array_long = np.empty(ra_size, dtype=random_array.dtype)
        random_array_long[:, 0::2] = random_array
        random_array_long[:, 1::2] = random_array
        c3 = _correlation(random_array_long, normalize_axes=0)
        assert (np.max(c) < 1)
        assert (np.max(c) > -1)
        np.testing.assert_array_almost_equal(c, c2)
        np.testing.assert_array_almost_equal(c, c3[:, 0::2])

    def test_correlation_axes(self, random_array):
        c = _correlation(random_array, axis=0, normalize_axes=0)
        c2 = _correlation(random_array.transpose(), axis=1, normalize_axes=1)
        np.testing.assert_array_almost_equal(c, c2.transpose())

    def test_correlations_axis(self, ones_zero):
        c = _correlation(ones_zero, axis=0, normalize_axes=0)
        result = np.ones((10, 20))
        result[1::2, :] = -1
        np.testing.assert_array_equal(c, result)
        # Show intensity doesn't matter
        c = _correlation(np.multiply(ones_zero, 3), axis=0, normalize_axes=0)
        np.testing.assert_array_equal(c, result)
        # Along the axis where everything is the same... This should be equal to zero
        c = _correlation(ones_zero, axis=1, normalize_axes=1)
        result = np.zeros((10, 20))
        np.testing.assert_array_almost_equal(c, result)

    def test_correlations_normalization(self, ones_hundred):
        c = _correlation(ones_hundred, axis=0, normalize_axes=0)
        result = np.zeros((10, 20))
        result[1::2, :] = -0.96078816
        result[0::2, :] = 0.96078816
        np.testing.assert_array_almost_equal(c, result)
        c = _correlation(ones_hundred, axis=1, normalize_axes=1)
        result = np.zeros((10, 20))
        np.testing.assert_array_almost_equal(c, result)

    def test_correlations_mask(self, ones_hundred):
        m = np.zeros((10, 20))
        m[2:4, :] = 1
        c = _correlation(ones_hundred, axis=0, normalize_axes=0, mask=m)
        print(c)
        result = np.zeros((10, 20))
        result[1::2, :] = -0.96078816
        result[0::2, :] = 0.96078816
        np.testing.assert_array_almost_equal(c, result)

    def test_correlations_mask2(self, ones_array):
        m = np.zeros((10, 20))
        m[1, 1] = 1
        m[2, 1:3] = 1
        m[3, 1:4] = 1
        m[4, 1:5] = 1
        m[5, 1:5] = 1
        m[6, 1:5] = 1
        m[7, 1:5] = 1
        c = _correlation(ones_array*7, axis=1, normalize_axes=1, mask=m)
        c2 = _correlation(ones_array*7, axis=0,normalize_axes=0,  mask=m)
        print(c2)
        result = np.zeros((10, 20))
        np.testing.assert_array_almost_equal(c, result)
        np.testing.assert_array_almost_equal(c2, result)

    def test_correlations_wrapping(
        self, ones_hundred
    ):  # Need to do extra checks to assure this is correct
        m = np.zeros((10, 20))
        m[2:4, :] = 1
        c = _correlation(ones_hundred, axis=0, normalize_axes=0, wrap=False)
        print(c)
        result = np.zeros((10, 20))
        result[0::2, :] = 2.26087665
        result[1::2, :] = -0.93478899
        np.testing.assert_array_almost_equal(c, result)

    def test_wrap_set(self):
        z = np.zeros(10)
        wrap_set_float(z, bottom=3.5, top=6.7, value=10)
        answer = [0., 0., 0., 5., 10., 10., 10., 7., 0., 0.]
        np.testing.assert_array_almost_equal(z,answer)

    def test_wrap_set(self):
        z = np.zeros(10)
        wrap_set_float(z, bottom=3.5, top=6.7, value=10)
        answer = [0., 0., 0., 5., 10., 10., 10., 7., 0., 0.]
        np.testing.assert_array_almost_equal(z, answer)

    def test_get_interp_matrix(self):
        angles = [0, .25, .5, .75, 1]
        mat = get_interpolation_matrix(angles=angles, angular_range=0, num_points=18)
        print(mat)
        z = np.zeros(90)
        z[:2] = 1
        z[-1] = 1
        z[44:47] = 1
        z[22:24] = 1
        z[67:69] = 1
        z[21] = .5
        z[24] = .5
        z[66] = .5
        z[69] = .5
        np.testing.assert_array_almost_equal(mat, z)
