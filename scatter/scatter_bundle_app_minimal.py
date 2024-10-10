from imgui_bundle import immapp
from scatter_widget_bundle import ScatterData, ScatterPresenter

scatter = ScatterData.make_default()
scatter_present = ScatterPresenter(scatter)

immapp.run(scatter_present.gui)
