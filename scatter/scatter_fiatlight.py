import matplotlib
matplotlib.use('Agg')

from imgui_bundle import immapp, imgui_fig
from scatter_widget_bundle import ScatterData, ScatterPresenter
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from enum import Enum


class DecisionStrategy(Enum):
    logistic_regression = LogisticRegression
    decision_tree = DecisionTreeClassifier


def plot_boundary(df: pd.DataFrame, strategy: DecisionStrategy, eps: float=1.0) -> Figure | None:
    if len(df) and (df['color'].nunique() > 1):
        X = df[['x', 'y']].values
        y = df['color']
        fig, ax = plt.subplots()
        if strategy == DecisionStrategy.logistic_regression:
            classifier = LogisticRegression().fit(X, y)
        else:
            classifier = DecisionTreeClassifier().fit(X, y)
        disp = DecisionBoundaryDisplay.from_estimator(
            classifier, X,
            response_method="predict_proba" if len(np.unique(df['color'])) == 2 else "predict",
            xlabel="x", ylabel="y",
            #alpha=0.5,
            eps=eps,
            ax=ax
        )
        disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
        ax.set_title(f"{classifier.__class__.__name__}")
        return fig
    else:
        return None


import fiatlight as fl
from scatter_widget_bundle.scatter_with_gui import ScatterWithGui

# Version 0: only the scatter widget
# def scatter_source(scatter_data: ScatterData) -> ScatterData:
#     return scatter_data
#
# fl.run(scatter_source)

# Version 1: only one function
# def scatter_to_figure(scatter_data: ScatterData, strategy: DecisionStrategy) -> Figure:
#     return plot_boundary(scatter_data.data_as_pandas(), strategy)
# fl.run(scatter_to_figure)


# Version 2: compose two functions
def scatter_source(scatter_data: ScatterData) -> ScatterData:
    return scatter_data

@fl.with_fiat_attributes(eps__range=(0.1, 10.0))
def scatter_to_figure(
        scatter_data: ScatterData,
        strategy: DecisionStrategy = DecisionStrategy.logistic_regression,
        eps: float=1.0) -> Figure:
    return plot_boundary(scatter_data.data_as_pandas(), strategy, eps)

# fl.run([scatter_source, scatter_to_figure])

# Version 3: a function graph
def scatter_to_df(scatter_data: ScatterData) -> pd.DataFrame:
    return scatter_data.data_as_pandas()

graph = fl.FunctionsGraph()
graph.add_function(scatter_source)
graph.add_function(scatter_to_figure)
graph.add_function(scatter_to_df)
graph.add_link(scatter_source, scatter_to_df)
graph.add_link(scatter_source, scatter_to_figure)
fl.run(graph)

