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

from hyperspy.signals import Signal2D

from pyxem.signals.polar_diffraction2d import PolarDiffraction2D, LazyPolarDiffraction2D
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D
from pyxem.signals.correlation2d import Correlation2D
from pyxem.signals.power2d import Power2D
from pyxem.signals.symmetry1d import Symmetry1D

from hyperspy.decorators import lazifyTestClass

@lazifyTestClass
class TestComputeAndAsLazy2D:
    @pytest.fixture
    def flat_pattern(self):
        pd = PolarDiffraction2D(data=np.ones(shape=(4, 5, 15, 10)))
        pd.axes_manager.signal_axes[0].scale = 0.5
        pd.axes_manager.signal_axes[0].name = "theta"
        pd.axes_manager.signal_axes[1].scale = 2
        pd.axes_manager.signal_axes[1].name = "k"
        return pd
    @pytest.fixture
    def cluster_pattern(self):
        data = np.ones((20, 20, 11, 60))
        data[2:5, 2:5, 5, ::30] = 100  # 2 fold
        data[10:13, 10:13:, 7, ::15] = 100  # 4 fold
        data[10:13, 3:6:, 9, ::10] = 100  # 6 fold
        data[2:5, 15:18:, 2, ::6] = 100  # 10 fold
        pd = PolarDiffraction2D(data)
        pd.axes_manager.signal_axes[0].scale = 0.5
        pd.axes_manager.signal_axes[0].name = "theta"
        pd.axes_manager.signal_axes[1].scale = 2
        pd.axes_manager.signal_axes[1].name = "k"
        return pd

    def test_2d_data_compute(self):
        dask_array = da.random.random((100, 150), chunks=(50, 50))
        s = LazyPolarDiffraction2D(dask_array)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s.compute()
        assert s.__class__ == PolarDiffraction2D
        assert not hasattr(s.data, "compute")
        assert s.axes_manager[0].scale == scale0
        assert s.axes_manager[1].scale == scale1
        assert s.metadata.Test == metadata_string
        assert dask_array.shape == s.data.shape

    def test_4d_data_compute(self):
        dask_array = da.random.random((4, 4, 10, 15), chunks=(1, 1, 10, 15))
        s = LazyPolarDiffraction2D(dask_array)
        s.compute()
        assert s.__class__ == PolarDiffraction2D
        assert dask_array.shape == s.data.shape

    def test_2d_data_as_lazy(self):
        data = np.random.random((100, 150))
        s = PolarDiffraction2D(data)
        scale0, scale1, metadata_string = 0.5, 1.5, "test"
        s.axes_manager[0].scale = scale0
        s.axes_manager[1].scale = scale1
        s.metadata.Test = metadata_string
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyPolarDiffraction2D
        assert hasattr(s_lazy.data, "compute")
        assert s_lazy.axes_manager[0].scale == scale0
        assert s_lazy.axes_manager[1].scale == scale1
        assert s_lazy.metadata.Test == metadata_string
        assert data.shape == s_lazy.data.shape

    def test_4d_data_as_lazy(self):
        data = np.random.random((4, 10, 15))
        s = PolarDiffraction2D(data)
        s_lazy = s.as_lazy()
        assert s_lazy.__class__ == LazyPolarDiffraction2D
        assert data.shape == s_lazy.data.shape

    def test_get_common_signal(self):
        data = np.ones((10, 11, 15))
        s = PolarDiffraction2D(data)
        common = s.get_common_signal()
        print(common.data)
        np.testing.assert_array_equal(np.shape(common), (11, 15))

    def test_speckle_filter(self):
        data = np.ones((10,10, 11, 15))
        s = PolarDiffraction2D(data)
        common = s.speckle_filter(sigmas=[2,3,1,1])

    def test_gaussian_filter(self,
                             flat_pattern):
        filtered = flat_pattern.gaussian_filter(sigma=2,
                                                inplace=False)
        np.testing.assert_array_almost_equal(filtered, np.ones(shape=(4, 5, 15, 10)))
        if flat_pattern._lazy:
            assert isinstance(flat_pattern.data, da.array)

    def test_get_symmetry_stem_library(self,
                                       cluster_pattern):
        common = cluster_pattern.get_symmetry_stem_library(theta_size=1,
                                                           k_size=1,
                                                           min_cluster_size=.5,
                                                           max_cluster_size=2,
                                                           sigma_ratio=1.2)
        assert isinstance(common, Symmetry1D)
        if cluster_pattern._lazy:
            assert isinstance(common.data, da.array)
        peaks = common.find_peaks(threshold_rel=.1)
        ans = [[[7, 11], [6, 3]],
               [[1, 1], [4, 4], [7, 7], [6, 4]],
               [[4, 4]],
               [[7, 7]],
               [],
               [[6, 6]]]
        centers = [[[p.cx, p.cy] for p in s] for s in peaks]
        assert np.testing.assert_array_almost_equal(ans, centers)

    def test_plot_clusters(self,cluster_pattern):
        common = cluster_pattern.get_symmetry_stem_library(theta_size=1,
                                                           k_size=1,
                                                           min_cluster_size=.5,
                                                           max_cluster_size=2,
                                                           sigma_ratio=1.2)
        assert isinstance(common, Symmetry1D)
        if cluster_pattern._lazy:
            assert isinstance(common.data, da.array)
        peaks = common.find_peaks(threshold_rel=.1)
        common.plot_clusters()

    def test_get_space_scale(self, cluster_pattern):
        space_sclae = cluster_pattern.get_space_scale_representation()
        print(space_sclae.axes_manager)



class TestCorrelations:
    @pytest.fixture
    def flat_pattern(self):
        pd = PolarDiffraction2D(data=np.ones(shape=(2, 2, 5, 5)))
        pd.axes_manager.signal_axes[0].scale = 0.5
        pd.axes_manager.signal_axes[0].name = "theta"
        pd.axes_manager.signal_axes[1].scale = 2
        pd.axes_manager.signal_axes[1].name = "k"
        return pd

    @pytest.fixture
    def ones_hundred(self):
        ones = np.ones((2, 3, 10, 20))
        ones[:, :, 0:20:2, :] = 100
        pd = PolarDiffraction2D(data=ones)
        pd.axes_manager.signal_axes[0].scale = 0.5
        pd.axes_manager.signal_axes[0].name = "theta"
        pd.axes_manager.signal_axes[1].scale = 2
        pd.axes_manager.signal_axes[1].name = "k"
        return pd

    def test_correlation_signal(self, flat_pattern):
        ac = flat_pattern.get_angular_correlation()
        assert isinstance(ac, Correlation2D)

    def test_axes_transfer(self, flat_pattern):
        ac = flat_pattern.get_angular_correlation()
        assert (
            ac.axes_manager.signal_axes[0].scale
            == flat_pattern.axes_manager.signal_axes[0].scale
        )
        assert (
            ac.axes_manager.signal_axes[1].scale
            == flat_pattern.axes_manager.signal_axes[1].scale
        )
        assert (
            ac.axes_manager.signal_axes[1].name
            == flat_pattern.axes_manager.signal_axes[1].name
        )

    def test_masking_correlation(self, flat_pattern):
        mask = np.zeros(shape=(5, 5))
        mask[1:2, 2:4] = 1
        twoflat = flat_pattern*2
        ap = twoflat.get_angular_correlation(mask=mask, normalize =False)
        np.testing.assert_array_almost_equal(ap.data, np.ones((2,2,5,4)))
        assert isinstance(ap, Correlation2D)

    def test_correlation_inplace(self, flat_pattern):
        ac = flat_pattern.get_angular_correlation(inplace=True)
        assert ac is None
        assert isinstance(flat_pattern, Correlation2D)

    @pytest.mark.parametrize(
        "mask", [None, np.zeros(shape=(5, 5)), Signal2D(np.zeros(shape=(2, 2, 5, 5)))]
    )
    def test_masking_angular_power(self, flat_pattern, mask):
        ap = flat_pattern.get_angular_power(mask=mask)
        print(ap)
        assert isinstance(ap, Power2D)

    def test_masking_angular_power_ones(self, ones_hundred):
        mask = np.zeros((10, 20))
        mask[2:4, :] = 1
        ap = ones_hundred.get_angular_correlation(mask=mask)
        print(ap)

    def test_axes_transfer_power(self, flat_pattern):
        ac = flat_pattern.get_angular_power()
        assert ac.axes_manager.signal_axes[0].scale == 1
        assert (
            ac.axes_manager.signal_axes[1].scale
            == flat_pattern.axes_manager.signal_axes[1].scale
        )
        assert (
            ac.axes_manager.signal_axes[1].name
            == flat_pattern.axes_manager.signal_axes[1].name
        )

    def test_power_inplace(self, flat_pattern):
        ac = flat_pattern.get_angular_power(inplace=True)
        assert ac is None
        assert isinstance(flat_pattern, Power2D)


class TestSymmetrySTEMProcess:
    @pytest.fixture
    def four_fold_cluster(self):
        data = np.random.random((10, 10, 100, 100))
        data[:, :, 48:52, 23:27] = 100
        data[:, :, 23:27, 48:52] = 100
        data[:, :, 73:77, 48:52] = 100
        data[:, :, 48:52, 73:77] = 100
        d = ElectronDiffraction2D(data)
        d.axes_manager.signal_axes[0].scale = 0.1
        d.axes_manager.signal_axes[0].name = "kx"
        d.axes_manager.signal_axes[1].scale = 0.1
        d.axes_manager.signal_axes[1].name = "ky"
        d.unit = "k_nm^-1"
        d.beam_energy=200
        d.set_ai()
        return d

    def test_to_polar(self, four_fold_cluster):
        four_fold_cluster.get_azimuthal_integral2d(50)


class TestDecomposition:
    def test_decomposition_is_performed(self, diffraction_pattern):
        s = PolarDiffraction2D(diffraction_pattern)
        s.decomposition()
        assert s.learning_results is not None

    def test_decomposition_class_assignment(self, diffraction_pattern):
        s = PolarDiffraction2D(diffraction_pattern)
        s.decomposition()
        assert isinstance(s, PolarDiffraction2D)
