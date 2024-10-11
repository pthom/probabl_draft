# Example of a standalone application
# ===================================
# Important notes:
#   - Copy this cell into a standalone python file to create an app
#   - This cell is long? This is for demonstration purposes only: it is an aggregation of the previous cells, with some added documentation
#     In a real world application, you would add modules to place your functions and utilities (such as DecisionStrategy and plot_boundary below)
#   - The string below is the docstring of the application, it will be displayed in a separate node, as a user documentation.

"""Interactive distribution partitioning
========================================
In this application, we display an interactive visualization. You can draw a dataset, and see partition boundaries,
using different strategies.

Quick user instructions
-----------------------
- Use the mouse wheel to zoom-in / zoom-out
- Right click and drag to move (pan) the view
- Drag nodes to move them
- Click on the (i) button on each node to hide its documentation
- Click on the "minimize" button on top of this cell to minimize it

"""

# Part 1: imports
# ---------------
import matplotlib ; matplotlib.use("Agg")  # setup step needed to integrate matplotlib in Fiatlight
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

import numpy as np
from enum import Enum
import pandas as pd
import time

# Specific imports for fiatlight
import fiatlight as fl
from scatter_widget_bundle import ScatterData


# Part 2: define the functions we want to use in the application
# --------------------------------------------------------------
# i. An enum used by plot_boundary to choose between logistic regression and decision tree
#    (Fiatlight will automatically convert this to radio buttons in the UI)
class DecisionStrategy(Enum):
    """This is a simple enum to choose between logistic regression and decision tree
    Fiatlight will automatically convert this to radio buttons in the UI
    """
    logistic_regression = LogisticRegression
    decision_tree = DecisionTreeClassifier


# ii. Below, we define a function that will plot the decision boundary of a classifier on a 2D dataset
#    It is decorated with `@fl.with_fiat_attributes`, where we specify the UI options
@fl.with_fiat_attributes(
    label = "Plot decision boundaries",  # label of the node in the UI
    strategy__label = "Choose strategy",  # label of the strategy argument in the UI
    strategy__tooltip = "you may choose between logistic and decision tree",  # tooltip for the strategy argument
    eps__label = "Epsilon value",  # label of the eps argument in the UI
    eps__tooltip = "Epsilon value used to draw the boundary",  # tooltip for the eps argument
    eps__range=(0.01, 10)  # range of the eps argument in the UI
)
def plot_boundary(
        df: pd.DataFrame,
        strategy: DecisionStrategy = DecisionStrategy.logistic_regression,
        eps: float = 1.0) -> Figure | None:
    """This function will plot the decision boundary of a classifier on a 2D dataset
    * df is a DataFrame with columns 'x', 'y', 'color'
    * strategy is a DecisionStrategy enum (choose between logistic regression and decision tree)
    * eps is the step size in the meshgrid

    It is decorated with `@fl.with_fiat_attributes(eps__range = (0.01, 10))` which means that the
    eps argument will be exposed in the UI as a slider with a range from 0.01 to 10.
    """
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
            # alpha=0.5,
            eps=eps,
            ax=ax
        )
        disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
        ax.set_title(f"{classifier.__class__.__name__}")
        return fig
    else:
        return None


@fl.with_fiat_attributes(label = "Draw data distribution")
def scatter_source(data: ScatterData) -> ScatterData:
    """Draw the distribution of data below, using different classes.

    """
    return data


@fl.with_fiat_attributes(label="View as DataFrame")
def scatter_to_df(data: ScatterData) -> pd.DataFrame:
    """Expand this node output to see the dataframe.
    To expand it, click on the eye to the left of the output region.
    """
    return data.data_as_pandas()


# Part 3: add a standalone GUI node
# ---------------------------------
# In this contrived example, we show an imaginary "timer" for a course of a given duration
time_start = time.time()


# This function will be shown in the GUI
def show_time_left() -> None:
    """This course should be completed in 1 hour."""
    from imgui_bundle import imgui_md, imgui_color_text_edit as editor
    total_duration = 60 * 60  # Total duration in seconds (1 hour)
    elapsed_time = time.time() - time_start
    time_left = max(0., total_duration - elapsed_time)
    minutes, seconds = divmod(int(time_left), 60)
    imgui_md.render(f"# Time left: {minutes:02d}:{seconds:02d}")


# Part 4: demo of input validation
# --------------------------------
# This part is just a demo of input validation. We create a node that asks the user to enter a prime number.
# (it has no relation to the rest of the application)
def prime_validator(x: int) -> int:
    """This validator checks that the value is a prime number, and warns the user if it is not."""
    if x < 2:
        raise ValueError("Please enter a number greater than 1")
    for i in range(2, int(x ** 0.5) + 1):
        if x % i == 0:
            raise ValueError("Please enter a prime number")
    return x

@fl.with_fiat_attributes(
    label="Enter a prime number",
    n__validator = prime_validator
)
def enter_prime_number(n: int) -> None:
    """Enter a prime number.
    This node is just a demo of input validation.
    """
    pass


# Part 4: create the graph and run the application
# ------------------------------------------------
graph = fl.FunctionsGraph()  #
graph.add_function_composition([scatter_source, scatter_to_df, plot_boundary])  # Add a functions composition to the graph
graph.add_markdown_node(__doc__)  # Add a markdown node with the docstring of the application
graph.add_gui_node(show_time_left)  # Add a GUI node to show the time left
graph.add_function(enter_prime_number)  # Add a function node to enter a prime number
fl.run(graph, app_name="example_app")   # Run the application with the graph we created
