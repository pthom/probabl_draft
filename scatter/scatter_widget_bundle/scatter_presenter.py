from imgui_bundle import imgui, hello_imgui, ImVec4, imgui_ctx, immvision, ImVec2, icons_fontawesome
from fiatlight.fiat_kits.fiat_image import ImageRgb
from fiatlight.fiat_types import JsonDict
from pydantic import BaseModel
from .scatter_data import ScatterData, ScatterCluster, Point2d, Color
from .coordinate_transformer import CoordinateTransformer


class ScatterGuiOptions(BaseModel):
    image_size_em: Point2d = (20, 20)
    random_brush_size: float = 0.1  # as a ratio of the scatter bounds
    brush_intensity: int = 1
    selected_class_idx: int = 0


def color_to_imvec4(color: Color) -> ImVec4:
    r, g, b, a = color[0], color[1], color[2], 255
    return ImVec4(r / 255, g / 255, b / 255, a / 255)


def color_edit(label: str, color: Color) -> tuple[bool, Color]:
    def color_to_list(color: Color) -> list[float]:
        return [c / 255 for c in color]
    def list_to_color(color_list: list[float]) -> Color:
        return tuple(int(c * 255) for c in color_list)  # type: ignore

    changed = False
    color_list = color_to_list(color)
    imgui.set_next_item_width(hello_imgui.em_size(10))
    changed, color_list = imgui.color_edit3(label, color_list)
    if changed:
        color = list_to_color(color_list)

    return changed, color


class ScatterPresenter:
    # Serializable data
    scatter: ScatterData
    gui_options: ScatterGuiOptions
    # Caches
    _cache_valid: bool = False
    _plot_image: ImageRgb  # a cache of the scatter plot as an image
    _transformer: CoordinateTransformer  # Coordinate transformer instance
    # undo/redo
    _undo_stack: list[ScatterData] = []
    _redo_stack: list[ScatterData] = []

    def __init__(self, scatter: ScatterData | None = None):
        if scatter is None:
            scatter = ScatterData.make_default()
        self.scatter = scatter
        self.gui_options = ScatterGuiOptions()

    def invalidate_cache(self) -> None:
        self._cache_valid = False

    def _store_undo(self) -> None:
        import copy
        self._undo_stack.append(copy.deepcopy(self.scatter))
        self._redo_stack = []

    def _undo(self) -> None:
        if len(self._undo_stack) > 0:
            self._redo_stack.append(self.scatter)
            self.scatter = self._undo_stack.pop()
            self.invalidate_cache()

    def _redo(self) -> None:
        if len(self._redo_stack) > 0:
            self._undo_stack.append(self.scatter)
            self.scatter = self._redo_stack.pop()
            self.invalidate_cache()

    def _can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def _can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def _update_cache(self) -> None:
        if not (0 <= self.gui_options.selected_class_idx < len(self.scatter.classes)):
            self.gui_options.selected_class_idx = 0
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
        dot_size_em = 0.35
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
        changed = False

        def edit_one_value(label: str, value: float) -> float:
            imgui.set_next_item_width(hello_imgui.em_size(10))
            changed_one_value, value = imgui.slider_float(label, value, -1000, 1000, "%.3f",
                                                          imgui.SliderFlags_.logarithmic.value)
            nonlocal changed
            changed = changed or changed_one_value
            return value

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

    def _gui_classes(self) -> bool:
        changed = False
        for i, scatter_class in enumerate(self.scatter.classes):
            imgui.push_id(str(i))  # Ensure unique ids within the loop (labels are ids for imgui)

            imgui.set_next_item_width(100)
            _, scatter_class.name = imgui.input_text("Name", scatter_class.name)
            imgui.same_line()

            changed_color, scatter_class.color = color_edit("Color", scatter_class.color)
            if changed_color:
                changed = True
            imgui.same_line()

            if imgui.small_button("Clear"):
                scatter_class.points = []
                changed = True
            imgui.same_line()

            if imgui.small_button("Delete"):
                del self.scatter.classes[i]
                self.gui_options.selected_class_idx = max(0, self.gui_options.selected_class_idx - 1)
                changed = True
            imgui.pop_id()

        if imgui.button("Add class"):
            new_class = ScatterCluster(name="new", color=(0, 0, 255))
            self.scatter.classes.append(new_class)
            changed = True

        return changed

    def _gui_options(self) -> None:
        """This draws the options on top of the scatter plot."""
        for i, scatter_class in enumerate(self.scatter.classes):
            is_selected = self.gui_options.selected_class_idx == i
            with imgui_ctx.push_style_color(imgui.Col_.text.value, color_to_imvec4(scatter_class.color)):
                # imgui.text(icons_fontawesome.ICON_FA_CIRCLE)
                # imgui.same_line()
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
            changed_classes = self._gui_classes()
            if changed_classes:
                self.invalidate_cache()

        # Brush options
        imgui.text("Brush")
        imgui.same_line()
        imgui.set_next_item_width(hello_imgui.em_size(6))
        _, self.gui_options.random_brush_size = imgui.slider_float(
            "size",
            self.gui_options.random_brush_size,
            0.01,
            0.5
        )
        imgui.same_line()
        imgui.set_next_item_width(hello_imgui.em_size(6))
        _, self.gui_options.brush_intensity = imgui.slider_int(
            "intensity",
            self.gui_options.brush_intensity,
            1,
            10
        )

        # Undo/redo
        imgui.begin_disabled(not self._can_undo())
        if imgui.button(icons_fontawesome.ICON_FA_UNDO):
            self._undo()
        imgui.end_disabled()
        imgui.same_line()
        imgui.begin_disabled(not self._can_redo())
        if imgui.button(icons_fontawesome.ICON_FA_REDO):
            self._redo()
        imgui.end_disabled()

    def _gui_plot(self, needs_texture_refresh: bool) -> bool:
        changed = False

        # Display the plot with immvision.image_display
        # "##" hides the label to use the image's position for mouse events
        em_pixel_size = imgui.get_font_size()

        def image_size_as_vec2():
            width_px = int(self.gui_options.image_size_em[0] * em_pixel_size)
            height_px = int(self.gui_options.image_size_em[1] * em_pixel_size)
            return ImVec2(width_px, height_px)

        # Display the image and make it resizable
        image_display_size = image_size_as_vec2()  # will be changed if the user resizes the widget
        image_display_size_backup = image_size_as_vec2()
        mouse_position = immvision.image_display_resizable(
            "##Scatter plot",
            self._plot_image,
            size=image_display_size,
            refresh_image=needs_texture_refresh
        )
        if image_display_size.x != image_display_size_backup.x or image_display_size.y != image_display_size_backup.y:
            self.gui_options.image_size_em = (
                image_display_size.x / em_pixel_size,
                image_display_size.y / em_pixel_size
            )
            self.invalidate_cache()

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
            if len(self.scatter.classes) > 0:
                if imgui.is_mouse_clicked(0):
                    self._store_undo()  # first store an undo state at the start of the operation
                if imgui.is_mouse_down(0):
                    for i in range(self.gui_options.brush_intensity):
                        self._add_random_point_around(mouse_position)
                    self.invalidate_cache()
                    changed = True

            # Draw invisible button to capture mouse events on the image
            imgui.set_cursor_screen_pos(image_position)
            imgui.invisible_button("##Scatter plot", imgui.get_item_rect_size())

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

    def save_gui_options_to_json(self) -> JsonDict:
        return self.gui_options.model_dump(mode="json")

    def load_gui_options_from_json(self, json_dict: JsonDict) -> None:
        self.gui_options = ScatterGuiOptions(**json_dict)
        self.invalidate_cache()
