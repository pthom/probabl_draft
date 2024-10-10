from imgui_bundle import imgui, hello_imgui, ImVec4, imgui_ctx, immvision, ImVec2
from fiatlight.fiat_kits.fiat_image import ImageRgb
from pydantic import BaseModel
import numpy as np
from numpy.typing import NDArray

from .scatter_data import ScatterData, Point2d, Color, Bounding


class ScatterGuiOptions(BaseModel):
    image_size_em: Point2d = (20, 20)
    random_brush_size: float = 0.1  # as a ratio of the scatter bounds
    selected_class_idx: int = 0


def color_to_imvec4(color: Color) -> ImVec4:
    r, g, b, a = color[0], color[1], color[2], 255
    return ImVec4(r / 255, g / 255, b / 255, a / 255)


AffineTransform = NDArray[np.float64]  # affine transformation matrix (2x3)


class ScatterPresenter:
    # Serializable data
    scatter: ScatterData | None
    gui_options: ScatterGuiOptions
    # Cache
    _plot_image: ImageRgb  # a cache of the scatter plot as an image
    _need_plot_image_update: bool = True

    def __init__(self, scatter: ScatterData | None = None):
        self.scatter = scatter
        self.gui_options = ScatterGuiOptions()

    def _compute_plot_image(self) -> None:
        """Convert the scatter plot to an image."""
        # Create an image with the scatter bounds
        # Draw the scatter points inside the image
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
            cluster_points_pixel = self._to_pixels(cluster.points)
            for point_pixel in cluster_points_pixel:
            # for point in cluster.points:
                # Convert bounds coordinates to pixel coordinates

                # point_pixel = self._to_pixel(point)
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
        point_bounds = self._to_bounds(point_pixel)

        # randomize the point position inside the brush size
        def make_random_deviation() -> Point2d:
            brush_ratio = self.gui_options.random_brush_size
            bound_width = self.scatter.bounding[1][0] - self.scatter.bounding[0][0]
            brush_radius = bound_width * brush_ratio
            angle = 2 * math.pi * random.random()
            r = brush_radius * random.random()
            random_deviation = (r * math.cos(angle), r * math.sin(angle))
            return random_deviation

        random_deviation = make_random_deviation()
        point_with_deviation = point_bounds[0] + random_deviation[0], point_bounds[1] + random_deviation[1]
        scatter_class = self.scatter.classes[self.gui_options.selected_class_idx]
        scatter_class.points.append(point_with_deviation)

    # ========================================
    # Cumbersome utilities to change coords,
    # because I wanted the scatter coords to differ from the image coords
    # ========================================
    @staticmethod
    def _compute_transform_matrix(src: Bounding, dst: Bounding) -> AffineTransform:
        """Compute a 2x3 affine transformation matrix for mapping between bounds."""
        src_min = np.array(src[0])
        src_max = np.array(src[1])
        dst_min = np.array(dst[0])
        dst_max = np.array(dst[1])

        scale = (dst_max - dst_min) / (src_max - src_min)

        # Create affine transformation matrix (2x3)
        M = np.array([
            [scale[0], 0, dst_min[0] - src_min[0] * scale[0]],
            [0, scale[1], dst_min[1] - src_min[1] * scale[1]]
        ])
        return M

    def _transform_bounds_to_pixel(self) -> AffineTransform:
        em_size = imgui.get_font_size()
        image_width = self.gui_options.image_size_em[0] * em_size
        image_height = self.gui_options.image_size_em[1] * em_size
        image_bounding = (0, image_height), (image_width, 0)
        return self._compute_transform_matrix(self.scatter.bounding, image_bounding)

    def _transform_pixel_to_bounds(self) -> AffineTransform:
        em_size = imgui.get_font_size()
        image_width = self.gui_options.image_size_em[0] * em_size
        image_height = self.gui_options.image_size_em[1] * em_size
        image_bounding = (0, image_height), (image_width, 0)
        return self._compute_transform_matrix(image_bounding, self.scatter.bounding)

    def _to_pixel(self, point: Point2d) -> Point2d:
        # Lots of room for optimization here: we could transform all points at once
        transform_bounds_to_pixel = self._transform_bounds_to_pixel()
        ones = np.ones(1)
        point_homogeneous = np.hstack([point, ones])
        point_pixel = np.dot(transform_bounds_to_pixel, point_homogeneous)
        return tuple(point_pixel)  # type: ignore

    def _to_pixels(self, points: list[Point2d]) -> list[Point2d]:
        #  Transforms a list of 2D points using the affine transformation matrix.
        if not points:
            return []

        transform: AffineTransform = self._transform_bounds_to_pixel()
        points_array = np.array(points, dtype=np.float64)  # Shape: (N, 2)

        # Concatenate a column of ones for homogeneous coordinates (N, 1)
        ones = np.ones((points_array.shape[0], 1), dtype=np.float64)
        points_homogeneous = np.hstack([points_array, ones])  # Shape: (N, 3)

        # Apply the affine transformation: (N, 3) dot (3, 2) = (N, 2)
        transformed_points = points_homogeneous @ transform.T  # Shape: (N, 2)

        # Convert the transformed NumPy array back to a list of tuples
        return [tuple(pt) for pt in transformed_points]

    def _to_bounds(self, point_pixel: Point2d) -> Point2d:
        transform_pixels_to_bounds = self._transform_pixel_to_bounds()
        ones = np.ones(1)
        point_homogeneous = np.hstack([point_pixel, ones])
        point_bounds = np.dot(transform_pixels_to_bounds, point_homogeneous)
        return tuple(point_bounds)  # type: ignore

    # ========================================
    # GUI
    # ========================================
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

        if imgui.collapsing_header("Edit classes"):
            for i, scatter_class in enumerate(self.scatter.classes):
                imgui.push_id(str(i))
                with imgui_ctx.begin_horizontal("edit class"):
                    # _, scatter_class.color = imgui.color_edit3("Color", scatter_class.color)
                    imgui.set_next_item_width(100)
                    _, scatter_class.name = imgui.input_text("Name", scatter_class.name)
                    if imgui.small_button("Clear"):
                        scatter_class.points = []
                        self._need_plot_image_update = True
                    if imgui.small_button("Delete"):
                        del self.scatter.classes[i]
                        self.gui_options.selected_class_idx = max(0, self.gui_options.selected_class_idx - 1)
                        self._need_plot_image_update = True
                    imgui.pop_id()

        # Brush size
        image_width_pixels = hello_imgui.em_size(self.gui_options.image_size_em[0])
        imgui.set_next_item_width(image_width_pixels)
        _, self.gui_options.random_brush_size = imgui.slider_float("Brush size", self.gui_options.random_brush_size, 0.01, 0.5)

    def _gui_plot(self, needs_texture_refresh: bool) -> bool:
        changed = False

        # Display the plot with immvision.image_display
        #     Below, "##" means "hide the label". This is important, because we want
        #     imgui.get_item_rect_min() to return the position of the image
        mouse_position = immvision.image_display("##Scatter plot", self._plot_image, refresh_image=needs_texture_refresh)
        # Handle event
        if imgui.is_item_hovered():
            # Draw circle around the mouse position on hover
            image_position = imgui.get_item_rect_min()
            brush_ratio = self.gui_options.random_brush_size
            circle_radius = hello_imgui.em_size(self.gui_options.image_size_em[0]) * brush_ratio
            circle_center = ImVec2(mouse_position[0] + image_position.x, mouse_position[1] + image_position.y)
            circle_color = imgui.IM_COL32(0, 0, 255, 60)
            imgui.get_window_draw_list().add_circle_filled(circle_center, circle_radius, circle_color)
            # Add a point on click
            if imgui.is_mouse_down(0):
                self._add_random_point_around(mouse_position)
                self._need_plot_image_update = True
                changed = True

            # Draw invisible button to capture mouse events
            # (otherwise they may be handled by other widgets, even if the image is hovered)
            imgui.set_cursor_screen_pos(image_position)
            imgui.invisible_button("##Scatter plot", imgui.get_item_rect_size())

            imgui.text("Hey")

        return changed

    def gui(self) -> bool:
        if self.scatter is None:
            imgui.text("No scatter data")
            return False
        needs_texture_refresh = self._need_plot_image_update
        if self._need_plot_image_update:
            self._compute_plot_image()
            self._need_plot_image_update = False

        changed = False
        self._gui_options()
        changed = self._gui_plot(needs_texture_refresh)
        return changed
