from imgui_bundle import imgui, hello_imgui, ImVec4, imgui_ctx, immvision, ImVec2
from fiatlight.fiat_kits.fiat_image import ImageRgb
from pydantic import BaseModel
from .scatter_data import ScatterData, Point2d, Color
from .coordinate_transformer import CoordinateTransformer


class ScatterGuiOptions(BaseModel):
    image_size_em: Point2d = (20, 20)
    random_brush_size: float = 0.1  # as a ratio of the scatter bounds
    selected_class_idx: int = 0


def color_to_imvec4(color: Color) -> ImVec4:
    r, g, b, a = color[0], color[1], color[2], 255
    return ImVec4(r / 255, g / 255, b / 255, a / 255)

def imvec4_to_color(color: ImVec4) -> Color:
    return int(color.x * 255), int(color.y * 255), int(color.z * 255)


def color_edit(label: str, color: Color) -> tuple[bool, Color]:
    def color_to_list(color: Color) -> list[float]:
        return [c / 255 for c in color]
    def list_to_color(color_list: list[float]) -> Color:
        return tuple(int(c * 255) for c in color_list)

    color_list = color_to_list(color)
    imgui.set_next_item_width(hello_imgui.em_size(6))
    changed, color_list = imgui.color_picker3(label, color_list)
    if changed:
        color = list_to_color(color_list)
    return changed, color

def color_edit_(label: str, color: Color) -> tuple[bool, Color]:
    def color_to_list(color: Color) -> list[float]:
        return [c / 255 for c in color]
    def list_to_color(color_list: list[float]) -> Color:
        return tuple(int(c * 255) for c in color_list)

    color_list = color_to_list(color)
    imgui.set_next_item_width(hello_imgui.em_size(4))
    changed, color_list = imgui.color_edit3(label, color_list)
    if changed:
        print("color_list", color_list)
        color = list_to_color(color_list)
    return changed, color


class ScatterPresenter:
    # Serializable data
    scatter: ScatterData | None
    gui_options: ScatterGuiOptions
    # Caches
    _cache_valid: bool = False
    _plot_image: ImageRgb  # a cache of the scatter plot as an image
    _transformer: CoordinateTransformer | None = None  # Coordinate transformer instance

    def __init__(self, scatter: ScatterData | None = None):
        self.scatter = scatter
        self.gui_options = ScatterGuiOptions()

    def invalidate_cache(self) -> None:
        self._cache_valid = False

    def _update_cache(self) -> None:
        if self._cache_valid:
            return
        if self.scatter is None:  # no data yet
            return
        self._cache_valid = True

        # fill self._transformer
        em_size = imgui.get_font_size()
        self._transformer = CoordinateTransformer(
            scatter_bounding=self.scatter.bounding,
            image_size_em=self.gui_options.image_size_em,
            em_size=em_size
        )
        # fill self._plot_image
        self._compute_plot_image()

    def _compute_plot_image(self) -> None:
        """Convert the scatter plot to an image."""
        from PIL import Image, ImageDraw
        import numpy as np

        em_pixel_size = imgui.get_font_size()
        width_px = int(self.gui_options.image_size_em[0] * em_pixel_size)
        height_px = int(self.gui_options.image_size_em[1] * em_pixel_size)
        # Create a blank white image
        image = Image.new("RGB", (width_px, height_px), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Draw the dots
        dot_size_em = 0.4
        dot_size_px = em_pixel_size * dot_size_em

        for cluster in self.scatter.classes:
            color = cluster.color  # Assuming ColorRgb is a tuple of ints
            cluster_points_pixel = self._transformer.to_pixels(cluster.points)
            for point_pixel in cluster_points_pixel:
                x, y = point_pixel

                # Define the bounding box for the brush
                upper_left = (x - dot_size_px / 2, y - dot_size_px / 2)
                lower_right = (x + dot_size_px / 2, y + dot_size_px / 2)

                # Draw the point as a filled ellipse (circle)
                draw.ellipse([upper_left, lower_right], fill=color, outline=None)

        self._plot_image = np.array(image)  # type: ignore

    def _add_random_point_around(self, point_pixel: Point2d) -> None:
        """Add a point to the scatter plot, given in pixel coordinates."""
        import random
        import math

        if not self._transformer or not self.scatter:
            return  # Early exit if transformer or scatter data is not initialized

        point_bounds = self._transformer.to_bounds(point_pixel)

        # Randomize the point position inside the brush size
        def make_random_deviation() -> Point2d:
            brush_ratio = self.gui_options.random_brush_size
            bound_width = self.scatter.bounding[1][0] - self.scatter.bounding[0][0]
            brush_radius = bound_width * brush_ratio
            angle = 2 * math.pi * random.random()
            r = brush_radius * random.random()
            random_deviation = (r * math.cos(angle), r * math.sin(angle))
            return random_deviation

        random_deviation = make_random_deviation()
        point_with_deviation = (
            point_bounds[0] + random_deviation[0],
            point_bounds[1] + random_deviation[1]
        )
        scatter_class = self.scatter.classes[self.gui_options.selected_class_idx]
        scatter_class.points.append(point_with_deviation)

    # ========================================
    # GUI
    # ========================================
    def _gui_bounds(self) -> bool:
        def edit_one_value(label: str, value: float) -> float:
            imgui.set_next_item_width(hello_imgui.em_size(10))
            changed_one_value, value = imgui.slider_float(label, value, -1000, 1000, "%.3f",
                                                          imgui.SliderFlags_.logarithmic.value)
            nonlocal changed
            changed = changed or changed_one_value
            return value

        changed = False
        x_min = self.scatter.bounding[0][0]
        y_min = self.scatter.bounding[0][1]
        x_max = self.scatter.bounding[1][0]
        y_max = self.scatter.bounding[1][1]

        x_min = edit_one_value("Min x", x_min)
        imgui.same_line()
        y_min = edit_one_value("Min y", y_min)

        x_max = edit_one_value("Max x", x_max)
        imgui.same_line()
        y_max = edit_one_value("Max y", y_max)

        if changed:
            self.scatter.bounding = ((x_min, y_min), (x_max, y_max))

        return changed

    def _gui_options(self) -> None:
        """This draws the options on top of the scatter plot."""
        for i, scatter_class in enumerate(self.scatter.classes):
            is_selected = self.gui_options.selected_class_idx == i
            with imgui_ctx.push_style_color(imgui.Col_.text.value, color_to_imvec4(scatter_class.color)):
                if imgui.radio_button(scatter_class.name, is_selected):
                    self.gui_options.selected_class_idx = i
                imgui.same_line()
                imgui.text(f"({len(scatter_class.points)})")
            imgui.same_line()
        imgui.new_line()

        if imgui.collapsing_header("Edit classes and bounds"):
            imgui.separator_text("Bounds")
            changed_bounds = self._gui_bounds()
            if changed_bounds:
                self.invalidate_cache()

            imgui.separator_text("Classes")
            for i, scatter_class in enumerate(self.scatter.classes):
                imgui.push_id(str(i))  # Ensure unique ids within the loop (labels are ids for imgui)

                imgui.set_next_item_width(100)
                _, scatter_class.name = imgui.input_text("Name", scatter_class.name)
                imgui.same_line()

                changed_color, scatter_class.color = color_edit("Color", scatter_class.color)
                if changed_color:
                    self.invalidate_cache()
                imgui.same_line()

                if imgui.small_button("Clear"):
                    scatter_class.points = []
                    self.invalidate_cache()
                imgui.same_line()

                if imgui.small_button("Delete"):
                    del self.scatter.classes[i]
                    self.gui_options.selected_class_idx = max(0, self.gui_options.selected_class_idx - 1)
                    self.invalidate_cache()
                imgui.pop_id()

        # Brush size
        image_width_pixels = hello_imgui.em_size(self.gui_options.image_size_em[0])
        imgui.set_next_item_width(image_width_pixels)
        _, self.gui_options.random_brush_size = imgui.slider_float(
            "Brush size",
            self.gui_options.random_brush_size,
            0.01,
            0.5
        )

    def _gui_plot(self, needs_texture_refresh: bool) -> bool:
        changed = False

        # Display the plot with immvision.image_display
        # "##" hides the label to use the image's position for mouse events
        mouse_position = immvision.image_display(
            "##Scatter plot",
            self._plot_image,
            refresh_image=needs_texture_refresh
        )
        # Handle event
        if imgui.is_item_hovered():
            if not self._transformer:
                return changed  # Early exit if transformer is not initialized

            # Draw circle around the mouse position on hover
            image_position = imgui.get_item_rect_min()
            brush_ratio = self.gui_options.random_brush_size
            circle_radius = self.gui_options.image_size_em[0] * self._transformer.em_size * brush_ratio
            circle_center = ImVec2(
                mouse_position[0] + image_position.x,
                mouse_position[1] + image_position.y
            )
            circle_color = imgui.IM_COL32(0, 0, 255, 60)
            imgui.get_window_draw_list().add_circle_filled(circle_center, circle_radius, circle_color)
            # Add a point on click
            if imgui.is_mouse_down(0):
                self._add_random_point_around(mouse_position)
                self.invalidate_cache()
                changed = True

            # Draw invisible button to capture mouse events
            imgui.set_cursor_screen_pos(image_position)
            imgui.invisible_button("##Scatter plot", imgui.get_item_rect_size())

            imgui.text("Hey")

        return changed

    def gui(self) -> bool:
        needs_texture_refresh = not self._cache_valid
        self._update_cache()
        if self.scatter is None:
            imgui.text("No scatter data")
            return False

        changed = False
        self._gui_options()
        changed = self._gui_plot(needs_texture_refresh)
        return changed
