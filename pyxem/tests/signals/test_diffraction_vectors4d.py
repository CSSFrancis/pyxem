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
import dask.array
import matplotlib.pyplot as plt
import numpy as np
import pytest
from skimage.draw import disk
from pyxem.signals.polar_diffraction2d import PolarDiffraction2D


class TestFindPeaks:
    @pytest.fixture
    def dataset(self):
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
        data = PolarDiffraction2D(data)
        data.axes_manager[0].name = "x"
        data.axes_manager[1].name = "y"
        data.axes_manager[2].name = "k"
        data.axes_manager[3].name = "$\theta$"

        data.axes_manager[0].scale = .25
        data.axes_manager[1].scale = .25
        data.axes_manager[2].scale = .5
        data.axes_manager[3].scale = (np.pi*2/80)

        data.axes_manager[0].unit = "nm"
        data.axes_manager[1].unit = "nm"
        data.axes_manager[2].unit = "nm^-1"
        data.axes_manager[3].unit = "Rad"
        data = data.filter(sigma=(2,2,2,2))
        return data

    def test_get_peaks(self, dataset):
        peaks = dataset.find_peaks_nd()
        return peaks

    def test_get_peaks_lazy(self,
                            dataset):
        dataset = dataset.as_lazy()
        dataset.rechunk(nav_chunks=(5, 5))
        peaks = dataset.find_peaks_nd()
        return peaks

    def test_get_extents(self,
                         dataset):
        peaks = dataset.find_peaks_nd()
        peaks.get_extents(dataset.data)

    def test_get_extents_lazy(self,
                              dataset):
        dataset = dataset.as_lazy()
        dataset.rechunk(nav_chunks=(5, 5))
        peaks = dataset.find_peaks_nd()
        peaks.get_extents(dataset.data)

    def test_refine(self,
                    dataset):
        peaks = dataset.find_peaks_nd()
        peaks.get_extents(dataset.data)
        peaks.refine_positions(dataset.data)

    def test_refine_lazy(self,
                         dataset):
        dataset = dataset.as_lazy()
        dataset.rechunk(nav_chunks=(5, 5))
        peaks = dataset.find_peaks_nd()
        peaks.get_extents(dataset.data)
        ref = peaks.refine_positions(dataset.data)
        print(peaks.data)
        print(ref.data)

    def test_combine_vectors(self,
                             dataset):
        dataset = dataset.as_lazy()
        dataset.rechunk(nav_chunks=(5, 5))
        peaks = dataset.find_peaks_nd()
        peaks.get_extents(dataset.data)
        ref = peaks.refine_positions(dataset.data)
        combo = ref.combine_vectors2(distance=1, duplicate_distance=2)
        print(len(combo.extents))
        print(len(combo.data))
        print(combo)

    def test_get_mean(self,
                     clusters):
        mean = clusters.clusters[0].get_mean(clusters.signal)
