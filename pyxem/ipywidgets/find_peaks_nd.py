import ipywidgets
import traits.api as t

from hyperspy_gui_ipywidgets.utils import (labelme, enum2dropdown,
        add_display_arg, set_title_container)
from link_traits import link
from hyperspy_gui_ipywidgets.axes import get_ipy_navigation_sliders


@add_display_arg
def find_peaksnd_ipy(obj, **kwargs):
    wdict = {}
    # Define widgets
    sigma_values = ipywidgets.Box([labelme("Sigma " + str(i),ipywidgets.FloatText(min=1, max=20, value=1))
                                   for i in range(obj.dim)])
    threshold = ipywidgets.FloatSlider(min=0, max=6, value=3)
    rel_threshold = ipywidgets.Checkbox(False)

    compute = ipywidgets.Button()
    filter = ipywidgets.Button

    wdict["sigma_values"] = sigma_values
    wdict["threshold"] = threshold
    wdict["rel_threshold"] = rel_threshold
    wdict["compute"] = compute
    wdict["filter"] = filter


    # Connect
    link((obj, "current_sigma"), (sigma_values, "value"))
    link((obj, "threshold"), (threshold, "value"))

    close = ipywidgets.Button(
        description="Close",
        tooltip="Close widget and close figure.")
    compute = ipywidgets.Button(
        description="Compute over navigation axes.",
        tooltip="Find the peaks by iterating over the navigation axes.")

    widgets_list = []

    #if obj.show_navigation_sliders:
    #    nav_widget = get_ipy_navigation_sliders(
    #            obj.signal.axes_manager.navigation_axes,
    #            in_accordion=True,
    #            random_position_button=True)
    #    widgets_list.append(nav_widget['widget'])
    #    wdict.update(nav_widget['wdict'])

    box = ipywidgets.VBox(widgets_list)



    def on_compute_clicked(b):
        obj.compute_navigation()
        obj.signal._plot.close()
        obj.close()
        box.close()
    compute.on_click(on_compute_clicked)

    def on_close_clicked(b):
        obj.signal._plot.close()
        obj.close()
        box.close()
    close.on_click(on_close_clicked)
    return {
        "widget": box,
        "wdict": wdict,
    }