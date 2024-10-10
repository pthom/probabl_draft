"""A widget that enables to draw a 2D dataset of points inside different classes.

cf https://www.youtube.com/watch?v=STPv0jSAQEk&list=PLSIzlWDI17bS025ph6R0W_3RKM0qJ3qoO&index=4
"Drawing a Dataset from inside Jupyter"
And the scatter ipywidget here: https://github.com/koaning/drawdata, by @koaning (vincent d warmerdam)
"""
from pydantic import BaseModel
import pandas as pd

Point2d = tuple[float, float]
Bounding = tuple[Point2d, Point2d]
Color = tuple[int, int, int]


def color_to_hex_string(color: Color) -> str:
    """Convert a color tuple to a hex string."""
    return f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"


class ScatterCluster(BaseModel):
    """A cluster of points in a scatter plot. It has a name, a color, and a list of points."""
    name: str
    color: Color
    points: list[Point2d] = []  # OK to have mutable default values in pydantic

    def info(self) -> str:
        return f"{self.name}: ({len(self.points)})"


class ScatterData(BaseModel):
    """Scatter plot data
    It has a list of classes, and a bounding box (min, max).
    """
    classes: list[ScatterCluster] = []
    bounding: Bounding = ((0, 0), (1, 1))

    def info(self) -> str:
        classes_info = ", ".join([c.info() for c in self.classes])
        r = f"[{classes_info}], bounding box: {self.bounding}"
        return r

    def data_as_pandas(self) -> pd.DataFrame:
        """Return the scatter data as a pandas DataFrame."""
        data = []
        for cluster in self.classes:
            for point in cluster.points:
                data.append({"x": point[0], "y": point[1], "class": cluster.name, "color": color_to_hex_string(cluster.color)})
        return pd.DataFrame(data)

    @staticmethod
    def make_default() -> "ScatterData":
        # Provide 4 classes, with colors (light blue, light orange, light green, light red)
        # a random number of points between 10 and 100
        light_blue = (173, 216, 230)
        light_orange = (255, 165, 0)
        light_green = (144, 238, 144)
        light_red = (255, 192, 203)
        classes = [
            ScatterCluster(name="a", color=light_blue),
            ScatterCluster(name="b", color=light_orange),
            ScatterCluster(name="c", color=light_green),
            ScatterCluster(name="d", color=light_red),
        ]
        r = ScatterData(classes=classes)
        return r
