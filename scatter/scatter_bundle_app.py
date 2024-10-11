import matplotlib
matplotlib.use('Agg')

from imgui_bundle import immapp, imgui_fig, hello_imgui, imgui
from scatter_widget_bundle import ScatterData, ScatterPresenter
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.inspection import DecisionBoundaryDisplay  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
import pandas as pd
import numpy as np


def plot_boundary(df: pd.DataFrame) -> Figure | None:
    if len(df) and (df['color'].nunique() > 1):
        X = df[['x', 'y']].values
        y = df['color']
        fig, ax = plt.subplots()
        classifier = DecisionTreeClassifier().fit(X, y)
        disp = DecisionBoundaryDisplay.from_estimator(
            classifier, X,
            response_method="predict_proba" if len(np.unique(df['color'])) == 2 else "predict",
            xlabel="x", ylabel="y",
            eps=0.1,
            ax=ax
        )
        disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
        ax.set_title(f"{classifier.__class__.__name__}")
        return fig
    else:
        return None


class App:
    scatter_data: ScatterData
    scatter_presenter: ScatterPresenter
    figure: Figure | None = None

    def __init__(self):
        self.scatter_data = ScatterData.make_default()
        self.scatter_presenter = ScatterPresenter(self.scatter_data)
        self.figure = None

    def gui(self):
        changed = self.scatter_presenter.gui()
        if changed:
            self.figure = plot_boundary(self.scatter_data.data_as_pandas())
        if self.figure:
            imgui_fig.fig("Plot", self.figure, refresh_image=changed)

        imgui.text(f"FPS: {hello_imgui.frame_rate()}")


if __name__ == "__main__":
    APP = App()
    immapp.run(APP.gui, window_size=(800, 1000))
