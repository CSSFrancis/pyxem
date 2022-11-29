from hyperspy.ui_registry import add_gui_method
import traits.api as t
from hyperspy.exceptions import SignalDimensionError
import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt


@add_gui_method(toolkey="pyxem.Diffraction2D.find_peaks_nd_interactive")
class PeaksFinder2D(t.HasTraits):
    sigma0 = t.Float(1)
    sigma1 = t.Float(1)
    sigma2 = t.Float(1)
    sigma3 = t.Float(1)
    sigma4 = t.Float(1)
    threshold = t.Float()

    show_navigation_sliders = t.Bool(False)

    def __init__(self, signal, peaks=None, **kwargs):
        if signal.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(
                signal.axes.signal_dimension, 2)
        self.original_signal = signal.deepcopy()
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
        self.fig, self.axs = plt.subplots(1, 3, figsize=(9, 3))
        self.axs[0].set_title("Peak Positions(Nav Axis)")
        self.axs[1].set_title("Peak Positions (Sig Axis)")
        self.axs[2].set_title("Peak Intensities")

        self.axs[0].set_ylim(self.signal.axes_manager.navigation_extent[0:2])
        self.axs[0].set_xlim(self.signal.axes_manager.navigation_extent[2:4])
        self.axs[1].set_ylim(self.signal.axes_manager.signal_extent[0:2])
        self.axs[1].set_xlim(self.signal.axes_manager.signal_extent[2:4])

    def filter(self):

        sigmas = [self.__getattribute__("sigma"+str(i))
                  for i in range(self.signal.data.ndim)]
        print("filtering Data...")
        self.signal.data = -self.original_signal.filter(sigma=sigmas).data
        nav_corner = self.signal.axes_manager.navigation_extent[::2]
        sig_corner = self.signal.axes_manager.signal_extent[::2]

        scales = [a.scale for a in self.signal.axes_manager._axes]
        kernel_size = np.multiply(np.multiply(sigmas, 2*np.sqrt(2)),
                                  scales)

        nav_marker = hs.plot.markers.Ellipse(x=nav_corner[0]+kernel_size[0],
                                             y=nav_corner[1]+kernel_size[1],
                                             width=kernel_size[0],
                                             height=kernel_size[1],
                                             fill=True,
                                             alpha=0.7)


        sig_marker = hs.plot.markers.Ellipse(x=sig_corner[0]+kernel_size[2],
                                             y=sig_corner[1]+kernel_size[3],
                                             width=kernel_size[2],
                                             height=kernel_size[3],
                                             fill=True, alpha=0.7)
        self.signal._plot.signal_plot.remove_markers(render_figure=True)
        self.signal._plot.navigator_plot.remove_markers(render_figure=True)
        self.signal.add_marker(sig_marker)
        self.signal.add_marker(nav_marker, plot_on_signal=False)
        self.signal.update_plot()

    def find_peaks(self):
        self.peaks = self.signal.find_peaks_nd(threshold_abs=self.threshold, get_intensity=True)
        if self.peaks is not None:
            print(len(self.peaks.data), " peaks found")
            nav_scales = [a.scale for a in self.signal.axes_manager.navigation_axes]
            nav_offsets = [a.offset for a in self.signal.axes_manager.navigation_axes]
            sig_scales = [a.scale for a in self.signal.axes_manager.signal_axes]
            sig_offsets = [a.offset for a in self.signal.axes_manager.signal_axes]
            navigation_pos = np.add(np.multiply(self.peaks.data[:, :2],
                                                nav_scales),
                                    nav_offsets)
            signal_pos = np.add(np.multiply(self.peaks.data[:, 2:4],
                                            sig_scales),
                                sig_offsets)
            self.axs[0].clear()
            self.axs[1].clear()
            self.axs[2].clear()
            self.axs[0].set_title("Peak Positions(Nav Axis)")
            self.axs[1].set_title("Peak Positions (Sig Axis)")
            self.axs[2].set_title("Peak Intensities")
            self.axs[0].scatter(navigation_pos[:, 1], navigation_pos[:, 0])
            self.axs[1].scatter(signal_pos[:, 1], signal_pos[:, 0])
            self.axs[0].set_xlim(self.signal.axes_manager.navigation_extent[0:2])
            self.axs[0].set_ylim(self.signal.axes_manager.navigation_extent[2:4][::-1])
            self.axs[1].set_xlim(self.signal.axes_manager.signal_extent[0:2])
            self.axs[1].set_ylim(self.signal.axes_manager.signal_extent[2:4][::-1])
            self.axs[2].hist(self.peaks.data[:,4])





