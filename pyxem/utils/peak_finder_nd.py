from hyperspy.ui_registry import add_gui_method
import traits.api as t
from hyperspy.exceptions import SignalDimensionError
import numpy as np
from hyperspy.drawing.marker import

@add_gui_method(toolkey="pyxem.Diffraction2D.find_peaks_nd_interactive")
class PeaksFinder2D(t.HasTraits):
    sigma0 = t.Float(1)
    sigma1 = t.Float(1)
    sigma2 = t.Float(1)
    sigma3 = t.Float(1)
    sigma4 = t.Float(1)
    recalculate = t.Button()
    #filter = t.Button()
    threshold = t.Float()

    show_navigation_sliders = t.Bool(False)

    def __init__(self, signal, peaks=None, **kwargs):
        if signal.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(
                signal.axes.signal_dimension, 2)
        self.original_signal = signal
        self.signal = signal
        self.peaks = peaks
        if self.signal._plot is None or not self.signal._plot.is_active:
            self.signal.plot()
        if self.signal.axes_manager.navigation_size > 0:
            self.show_navigation_sliders = True
            #self.signal.axes_manager.events.indices_changed.connect(
            #    self._update_peak_finding, [])
            #self.signal._plot.signal_plot.events.closed.connect(self.disconnect, [])
        # Set initial parameters:
        # As a convenience, if the template argument is provided, we keep it
        # even if the method is different, to be able to use it later.

    def filter(self):

        sigmas = [self.__getattribute__("sigma"+str(i))
                  for i in range(self.signal.data.ndim)]
        print(sigmas)
        self.signal.data = -self.original_signal.filter(sigma=sigmas).data
        nav_marker = h


        self.signal.update_plot()
        #self.signal. .update()
        print("filtering Data...")