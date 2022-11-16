from hyperspy.ui_registry import add_gui_method
import traits.api as t
from hyperspy.exceptions import SignalDimensionError
import numpy as np
import hyperspy.api as hs

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
        nav_corner = self.signal.axes_manager.navigation_extent[::2]
        sig_corner = self.signal.axes_manager.signal_extent[::2]

        kernel_size = np.multiply(sigmas, 2*np.sqrt(2))
        nav_marker = hs.plot.markers.ellipse(x=nav_corner[0]+kernel_size[0],
                                             y=nav_corner[1]++kernel_size[1],
                                             width=kernel_size[0],
                                             height=kernel_size[1],
                                             fill=True, alpha=0.7)

        sig_marker = hs.plot.markers.ellipse(x=sig_corner[0]+kernel_size[2],
                                             y=sig_corner[1]+kernel_size[3],
                                             width=kernel_size[2],
                                             height=kernel_size[3],
                                             fill=True, alpha=0.7)
        self.signal._plot.signal_plot.remove_markers(render_figure=True)
        self.signal._plot.navigator_plot.remove_markers(render_figure=True)
        self.signal.add_marker(sig_marker)
        self.signal.add_marker(nav_marker, plot_on_signal=False)

        self.signal.update_plot()
        print("filtering Data...")

    def find_peaks(self):
        print(self.threshold)
        #self.signal.find_peaks_nd()