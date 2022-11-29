import ipywidgets
from hyperspy_gui_ipywidgets.utils import add_display_arg
from link_traits import link


@add_display_arg
def peak_finder_nd(obj):
    """ Take some signal (obj) and create a widget for finding peaks.
    """
    box_layout = ipywidgets.Layout( width="100px")
    num_dim = obj.signal.data.ndim
    sigmas = [ipywidgets.FloatText(value=1,layout=box_layout)
                              for i in range(num_dim)]
    for i, s in enumerate(sigmas):
        link((obj, "sigma"+str(i)), (s, "value"))
    sigma_box = ipywidgets.HBox([ipywidgets.Label("Filtering Sigmas: "), ] + sigmas)
    calculate = ipywidgets.Button(description='Find Peaks')
    threshold_peaks = ipywidgets.Button(description='Threshold Peaks')
    threshold = ipywidgets.FloatSlider(discription="Threshold")
    link((obj, "threshold"), (threshold, "value"))

    filter = ipywidgets.Button(description='Filter Data',)
    buttons = ipywidgets.HBox([filter, calculate])

    box = ipywidgets.VBox([sigma_box, threshold, buttons])
    wdict = {}

    def on_filter_click(b):
        obj.filter()

    def on_calculate_click(b):
        obj.find_peaks()

    filter.on_click(on_filter_click)
    calculate.on_click(on_calculate_click)
    return {"widget": box, "wdict": wdict}

