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

from pyxem.signals.correlation2d import Correlation2D, LazyCorrelation2D
from pyxem.signals.symmetry1d import Symmetry1D
from pyxem.signals.power2d import Power2D



class TestComputeAndAsLazy2D:
    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyCorrelation2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()

        assert s.__class__ == Correlation2D
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15), chunks=(1, 1, 10, 15))
        s = LazyCorrelation2D(dask_array)
        s.compute()
        assert s.__class__ == Correlation2D
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = Correlation2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyCorrelation2D
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = Correlation2D(data)
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyCorrelation2D
        assert data.shape == s_lazy.data.shape


class TestGetPower:
    @pytest.fixture
    def flat_pattern(self):
        pd = Correlation2D(data=np.ones(shape=(2, 2, 5, 5)))
        pd.axes_manager.signal_axes[0].scale = 0.5
        pd.axes_manager.signal_axes[0].name = "theta"
        pd.axes_manager.signal_axes[1].scale = 2
        pd.axes_manager.signal_axes[1].name = "k"
        return pd

    def test_power_signal(self, flat_pattern):
        power = flat_pattern.get_angular_power()
        assert isinstance(power, Power2D)

    def test_power_signal_inplace(self, flat_pattern):
        power = flat_pattern.get_angular_power(inplace=True)
        assert isinstance(flat_pattern, Power2D)
        assert power is None

    def test_summed_power_signal(self, flat_pattern):
        power = flat_pattern.get_summed_angular_power(inplace=False)
        assert isinstance(power, Power2D)

    def test_summed_power_signal_inplace(self, flat_pattern):
        power = flat_pattern.get_summed_angular_power(inplace=True)
        assert isinstance(flat_pattern, Power2D)
        assert power is None

    def test_symmetry_stem(self, flat_pattern):
        p = flat_pattern.get_symmetry_stem()
        assert isinstance(p,SymmetrySTEM)

class TestSymmetrySTEM:
    @pytest.fixture
    def sym_pattern(self):
        sym_data = np.random.random(size=(21, 20, 19, 90))
        sym_data[1:3, 1:3, 15:19, 45] = 10  # 2 fold
        sym_data[1:3, 1:3, 10:15, 22:23] = 10  # 4 fold
        sym_data[1:3, 1:3, 10:15, 67:68] = 10  # 4 fold
        pd = Correlation2D(data=sym_data)
        pd.axes_manager.signal_axes[0].scale = (np.pi / 45)
        pd.axes_manager.signal_axes[0].name = "phi"
        pd.axes_manager.signal_axes[1].scale = 2
        pd.axes_manager.signal_axes[1].name = "k"
        return pd

    def test_find_clusters(self, sym_pattern):
        sym = sym_pattern.get_symmetry_stem()
        assert sym.axes_manager.navigation_shape[1:] == sym_pattern.axes_manager.navigation_shape
        assert isinstance(sym, Symmetry1D)
        np.testing.assert_array_equal(sym.symmetries, [1, 2, 4, 6, 8, 10])

class TestSymmetry1D:
    @pytest.fixture
    def sym_pattern(self):
        sym_data = np.random.random(size=(21, 20, 6,  19))
        sym_data[1, 1:3, 1:3, 15:19] = 10  # 2 fold
        sym_data[2, 1:3, 10:15] = 10  # 4 fold
        sym_data[3, 1:3, 10:15] = 10  # 6 fold
        pd = Symmetry1D(data=sym_data)
        pd.axes_manager.signal_axes[0].scale = 2
        pd.axes_manager.signal_axes[0].name = "k"
        pd.symmetries = [1, 2, 4, 6, 8, 10]
        return pd

    @pytest.mark.parametrize("method", ["log", "dog", "doh","adsk"]         )
    def test_cluster(self, sym_pattern, method):
        sym_pattern.get_clusters(method=method)

    def test_plot(self,
                  sym_pattern):
        sym_pattern.plot_all(k_range=[0, 38])
        plt.show()


class TestDecomposition:
    def test_decomposition_is_performed(self, diffraction_pattern):
        s = Correlation2D(diffraction_pattern)
        s.decomposition()
        assert s.learning_results is not None

    def test_decomposition_class_assignment(self, diffraction_pattern):
        s = Correlation2D(diffraction_pattern)
        s.decomposition()
        assert isinstance(s, Correlation2D)
