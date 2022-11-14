import traits.api as t
from hyperspy.exceptions import SignalDimensionError
from hyperspy.signal import BaseSignal

from hyperspy.ui_registry import add_gui_method


@add_gui_method(toolkey="pyxem.DiffractionSignal2D.find_peaks_nd_interactive")
class PeaksFinderND(t.HasTraits):
    filter = t.Enum(
        'log',
        default='log')
    # for log
    dim = t.Int
    kernel_sigma = t.Array([])
    current_sigma = t.Array([])
    is_sigma_equal = t.Bool(True)
    threshold = t.Float(1.0)

    current_nav_pos = t.Array([])
    filtered_data = t.Instance(BaseSignal)
    peaks = t.Instance(BaseSignal)

    # for peak finding
    filter_data = t.Button()
    compute_peaks = t.Button()
    compute_intensities = t.Button()
    compute_distribution = t.Button()
    compute_sizes = t.Button()

    def __init__(self, signal):
        if signal.axes_manager.signal_dimension != 2:
            raise SignalDimensionError(
                signal.axes.signal_dimension, 2)
        self.dim = signal.data.ndim
        self.kernel_sigma = [1, ] * self.dim
        self.current_sigma = [0, ] * self.dim
        self.current_nav_pos = [int(ax.size/2) for ax in signal.axes_manager.navigation_axes]
        self.filtered_data = signal

    def filter_data(self, ):
        self.filtered_data = -self.signal.filter(sigma=self.kernel_sigma)

    def compute_peaks(self):
        self.peaks = self.filtered_data.find_peaks_nd()

    def _parse_paramaters_initial_values(self, **kwargs):
        # Get the attribute to argument mapping for the current method
        arg_mapping = self._attribute_argument_mapping_dict[
            self._normalise_method_name(self.method)]
        for attr, arg in arg_mapping.items():
            if arg in kwargs.keys():
                setattr(self, attr, kwargs[arg])

    def _update_peak_finding(self, method=None):
        if method is None:
            method = self.method
        self._find_peaks_current_index(method=method)
        self._plot_markers()

    def _method_changed(self, old, new):
        if new == 'Template matching' and self.xc_template is None:
            raise RuntimeError('The "template" argument is required.')
        self._update_peak_finding(method=new)

    def _parameter_changed(self, old, new):
        self._update_peak_finding()

    def _set_parameters_observer(self):
        for parameters_mapping in self._attribute_argument_mapping_dict.values():
            for parameter in list(parameters_mapping.keys()):
                self.on_trait_change(self._parameter_changed, parameter)

    def _get_parameters(self, method):
        # Get the attribute to argument mapping for the given method
        arg_mapping = self._attribute_argument_mapping_dict[method]
        # return argument and values as kwargs
        return {arg: getattr(self, attr) for attr, arg in arg_mapping.items()}

    def _normalise_method_name(self, method):
        return method.lower().replace(' ', '_')

    def _find_peaks_current_index(self, method):
        method = self._normalise_method_name(method)
        self.peaks.data = self.signal.find_peaks(method, current_index=True,
                                                 interactive=False,
                                                 **self._get_parameters(method))

    def _plot_markers(self):
        if self.signal._plot is not None and self.signal._plot.is_active:
            self.signal._plot.signal_plot.remove_markers(render_figure=True)
        peaks_markers = self._peaks_to_marker()
        self.signal.add_marker(peaks_markers, render_figure=True)

    def _peaks_to_marker(self, markersize=20, add_numbers=True,
                         color='red'):
        # make marker_list for current index
        from hyperspy.drawing._markers.point import Point

        x_axis = self.signal.axes_manager.signal_axes[0]
        y_axis = self.signal.axes_manager.signal_axes[1]

        if np.isnan(self.peaks.data).all():
            marker_list = []
        else:
            marker_list = [Point(x=x_axis.index2value(int(round(x))),
                                 y=y_axis.index2value(int(round(y))),
                                 color=color,
                                 size=markersize)
                for x, y in zip(self.peaks.data[:, 1], self.peaks.data[:, 0])]

        return marker_list

    def compute_navigation(self):
        method = self._normalise_method_name(self.method)
        with self.signal.axes_manager.events.indices_changed.suppress():
            self.peaks.data = self.signal.find_peaks(
                method, interactive=False, current_index=False,
                **self._get_parameters(method))

    def close(self):
        # remove markers
        if self.signal._plot is not None and self.signal._plot.is_active:
            self.signal._plot.signal_plot.remove_markers(render_figure=True)
        self.disconnect()

    def disconnect(self):
        # disconnect event
        am = self.signal.axes_manager
        if self._update_peak_finding in am.events.indices_changed.connected:
            am.events.indices_changed.disconnect(self._update_peak_finding)

    def set_random_navigation_position(self):
        index = np.random.randint(0, self.signal.axes_manager._max_index)
        self.signal.axes_manager.indices = np.unravel_index(index,
            tuple(self.signal.axes_manager._navigation_shape_in_array))[::-1]