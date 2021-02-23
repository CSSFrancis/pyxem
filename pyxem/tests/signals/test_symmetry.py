# -*- coding: utf-8 -*-
# Copyright 2017-2019 The pyXem developers
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
import dask.array as da
import matplotlib.pyplot as plt

from pyxem.signals.correlation2d import Correlation2D
from pyxem.signals.symmetry1d import Symmetry1D


class TestSymmetry1D:
    @pytest.fixture
    def sym_pattern(self):
        sym_data = np.random.random(size=(21, 20, 6, 19))
        sym_data[1:3, 1:3, 1, 4:9] = 10  # 2 fold
        sym_data[6:10, 3:8, 2, 4:9] = 10  # 4 fold
        sym_data[13:18, 1:6, 3, 7:14] = 10  # 6 fold
        pd = Symmetry1D(data=sym_data)
        pd.axes_manager.signal_axes[0].scale = 2
        pd.axes_manager.signal_axes[0].name = "k"
        pd.axes_manager.navigation_axes[0].name = "Symmetry"
        pd.axes_manager.navigation_axes[1].name = "x"
        pd.axes_manager.navigation_axes[2].name = "y"
        pd.symmetries = [1, 2, 4, 6, 8, 10]
        return pd

    def test_blurring(self, sym_pattern):
        blurred = sym_pattern.get_blurred_library()
        blurred.find_peaks(threshold_abs=1)
        print(blurred.clusters)
        assert(blurred.clusters[1][0].cx == 2)
        assert (blurred.clusters[1][0].cy == 2)
        assert (blurred.clusters[2][0].cx == 5)
        assert (blurred.clusters[2][0].cy == 8)
        assert (blurred.clusters[3][0].cx == 3)
        assert (blurred.clusters[3][0].cy == 15)



    @pytest.mark.parametrize("method", ["log", "dog", "doh","adsk"])
    def test_cluster(self, sym_pattern, method):
        sym_pattern.get_clusters(method=method)

    def test_plot(self,
                  sym_pattern):
        sym_pattern.plot_all(k_range=[0, 38])
        plt.show()