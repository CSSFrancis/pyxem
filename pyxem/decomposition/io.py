import h5py
import logging
from packaging.version import Version
from pathlib import Path
import warnings

import dask.array as da

from hyperspy.io_plugins._hierarchical import (
    # hyperspy.io_plugins.hspy.get_signal_chunks is in the hyperspy public API
    HierarchicalWriter, HierarchicalReader, version, get_signal_chunks
    )


_logger = logging.getLogger(__name__)


# Plugin characteristics
# ----------------------
format_name = 'DSPY'
description = \
    'The default file format for HyperSpy based on the HDF5 standard'
full_support = True
# Recognised file extension
file_extensions = ['dspy',]
default_extension = 0
# Writing capabilities
writes = True
version = version
# ----------------------
class DecompositionWriter(HierarchicalWriter):
    def __init__(self, file, signal, expg, **kwds):
        super().__init__(file, signal, expg, **kwds)
        self.Dataset = h5py.Dataset
        self.Group = h5py.Group
        #self.unicode_kwds = {"dtype": h5py.special_dtype(vlen=str)}
        #self.ragged_kwds = {"dtype": h5py.special_dtype(vlen=signal.data[0].dtype)}

    @staticmethod
    def _store_data(data, dset, group, key, chunks):
        if isinstance(data, da.Array):
            if data.chunks != dset.chunks:
                data = data.rechunk(dset.chunks)
            da.store(data, dset)
        elif data.flags.c_contiguous:
            dset.write_direct(data)
        else:
            dset[:] = data




def save(file, data, labels, extents, axes,):
    f = h5py.File(file,d)