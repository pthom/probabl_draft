from .scatter_data import ScatterData
from .scatter_presenter import ScatterPresenter
from .scatter_with_gui import register_widget_fiatlight_gui

register_widget_fiatlight_gui()

__all__ = ["ScatterData", "ScatterPresenter"]