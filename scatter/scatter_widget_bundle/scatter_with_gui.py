from fiatlight.fiat_types import FiatAttributes, JsonDict
from fiatlight.fiat_core.any_data_with_gui import AnyDataWithGui
from imgui_bundle import imgui
from .scatter_data import ScatterData
from .scatter_presenter import ScatterPresenter


class ScatterWithGui(AnyDataWithGui[ScatterData]):
    _presenter: ScatterPresenter

    def __init__(self) -> None:
        super().__init__(ScatterData)
        self._presenter = ScatterPresenter()

        # present_str: short string form (when presenting in collapsed form)
        self.callbacks.present_str = lambda value: value.info()

        # present: full form (when presenting in expanded form). Only needed if we want to present it as a function output)
        # self.callbacks.present = self.present  #

        # default_value_provider: function that provides a default value. Should be provided, if not using the default constructor
        self.callbacks.default_value_provider = ScatterData.make_default

        # edit: function that edits the value. Should be provided, if the value is editable
        # (this is where we do the graphical user edition of the scatter data)
        self.callbacks.edit = self.edit

        # on_change needs to be set, since the presenter has a cache
        # (this is where it will update its cache)
        self.callbacks.on_change = self.on_change

        # save/load_gui_options_to_json:
        # here, we save and load the gui options to/from a json dictionary
        # (this is where we save the presenter internal state: size of the image, etc.)
        self.callbacks.save_gui_options_to_json = self._presenter.save_gui_options_to_json
        self.callbacks.load_gui_options_from_json = self._presenter.load_gui_options_from_json

        # clipboard_copy_str: function that copies the value as a string to the clipboard
        self.callbacks.clipboard_copy_str = lambda value: value.data_as_pandas().to_csv()

    # def present(self, value: ScatterData) -> None:
    #     imgui.text(f"present {value.info()}")

    def edit(self, _value: ScatterData) -> tuple[bool, ScatterData]:
        # _value is not used, it is cached in the presenter
        changed = self._presenter.gui()
        return changed, self._presenter.scatter

    def on_change(self, value: ScatterData) -> None:
        self._presenter.scatter = value
        self._presenter.invalidate_cache()


def register_widget_fiatlight_gui() -> None:
    from fiatlight.fiat_togui.gui_registry import register_type

    register_type(ScatterData, ScatterWithGui)
