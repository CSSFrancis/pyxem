# -*- coding: utf-8 -*-
# Copyright 2016-2021 The pyXem developers
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
from pyxem.decomposition.vector_decomposition import VectorDecomposition2D
from pyxem.decomposition.diffraction_vector import DiffractionVector
from pyxem.signals import Diffraction2D
from skimage.draw import disk
import hyperspy.api as hs


class TestVectorDecomposition:
    @pytest.fixture
    def three_section(self):
        x = np.random.random((100, 50, 20, 20))
        x[0:20, :, 5:7, 5:7] = x[0:20, :, 5:7, 5:7] + 10
        x[20:60, :, 1:3, 14:16] = x[20:60, :, 1:3, 14:16] + 10
        x[60:100, :, 6:8, 10:12] = x[60:100, :, 6:8, 10:12] + 10
        d = Diffraction2D(x)
        return d

    @pytest.fixture
    def three_section(self):
        data = np.random.random((25, 25, 20, 80))
        # 4 fold symmetry
        rr1, cc1 = disk((10, 10), 3)
        rr2, cc2 = disk((10, 30), 3)
        rr3, cc3 = disk((10, 50), 3)
        rr4, cc4 = disk((10, 70), 3)
        rr2fold, cc2fold = np.append(rr1, rr2), np.append(cc1, cc2)
        rr4fold, cc4fold = np.append(rr1, [rr2, rr3, rr4]), np.append(cc1, [cc2, cc3, cc4])
        data[10:13, 10:13, rr2fold, cc2fold] = 100
        data[4:7, 14:17, rr4fold, cc4fold] = 100
        data[5:8, 20:23, rr2fold, cc2fold] = 100
        data[21:24, 13:16, rr4fold, cc4fold] = 100
        data = Diffraction2D(data)
        return data

    def test_decomposition(self, three_section):
        filtered = three_section.filter()
        peaks = filtered.find_peaks(threshold_rel=0.7)
        print(peaks)
        peaks.get_extents(three_section)
        print(peaks)
        print(len(peaks.labels))

    def test_decomp(self, three_section):
        filtered = three_section.filter()
        peaks = filtered.find_peaks(threshold_rel=0.7)
        new_peaks = peaks.get_extents(three_section, threshold=0.5)
        print("new", new_peaks[0:2].vectors)



class Test_Refinement:
    @pytest.fixture
    def circles(self):
        rr1, cc1 = disk((10, 10), 3)
        rr2, cc2 = disk((3, 47), 6, shape=(100, 50))
        x = np.zeros((1, 1, 20, 20))
        y = np.zeros((100, 50,1,1))
        x[:, :, rr1, cc1] = 10
        y[rr2, cc2, :, :] = 10
        c = np.multiply(x,y)
        d = Diffraction2D(c)
        return d

    def test_extent(self, circles):
        v = DiffractionVector([[4, 46, 12, 11]])
        print(v)
        e = v.get_extents(data=circles)
        print("extent", e.extents)
        ref = e.refine_positions(data=circles.data)
        np.testing.assert_equal(ref.vectors, [[3, 45, 10, 10]])

    def test_save(self, circles, tmp_path):
        v = DiffractionVector([[4, 46, 12, 11]])
        print(v)
        e = v.get_extents(data=circles)
        print("extent", e.extents)
        ref = e.refine_positions(data=circles.data)
        fname = tmp_path / "test.zspy"
        ref.axes_manager.signal_axes[0].index=0
        ref.axes_manager.signal_axes[1].index =0
        print(ref.axes_manager._axes)
        for a in ref.axes_manager.signal_axes:
            print(a)
            a.convert_to_vector_axis()
        #ref.axes_manager.signal_indices_in_array = None
        ref.slices=None
        ref.save(fname)
        new_ref = hs.load(fname)
