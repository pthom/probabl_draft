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

        self.callbacks.present_str = self.present_str  # needed when presenting in collapsed form
        # self.callbacks.present = self.present  # only needed if we want to present it as a function output
        self.callbacks.default_value_provider = self.default_value_provider  # needed since we do not use the default constructor
        self.callbacks.edit = self.edit  # needed for edition

        # on_change needs to be set, since the presenter has a cache
        # (this is where it will update its cache)
        self.callbacks.on_change = self.on_change

    def present_str(self, value: ScatterData) -> str:
        return value.info()

    # def present(self, value: ScatterData) -> None:
    #     imgui.text(f"present {value.info()}")

    def edit(self, _value: ScatterData) -> tuple[bool, ScatterData]:
        # _value is not used, it is cached in on_change
        changed = self._presenter.gui()
        return changed, self._presenter.scatter

    @staticmethod
    def default_value_provider() -> ScatterData:
        return ScatterData.make_default()

    def on_change(self, value: ScatterData) -> None:
        self._presenter.scatter = value
        self._presenter._need_plot_image_update = True


def register_gui() -> None:
    from fiatlight.fiat_togui.gui_registry import register_type

    register_type(ScatterData, ScatterWithGui)


# Register the GUI at startup
register_gui()
