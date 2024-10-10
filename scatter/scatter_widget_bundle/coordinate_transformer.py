import numpy as np
from numpy.typing import NDArray
from .scatter_data import Bounding, Point2d


AffineTransform = NDArray[np.float64]  # Affine transformation matrix (2x3)


class CoordinateTransformer:
    """A transformer for mapping points between scatter bounds and pixel coordinates."""
    def __init__(self, scatter_bounding: Bounding, image_size_em: Point2d, em_size: float):
        """
        Initializes the transformer with scatter bounds, image size in em units, and em size in pixels.
        """
        self.scatter_bounding = scatter_bounding
        self.image_size_em = image_size_em
        self.em_size = em_size
        self.image_bounding = self._compute_image_bounding()
        self.transform_bounds_to_pixel = self._compute_transform_matrix(
            self.scatter_bounding,
            self.image_bounding
        )
        self.transform_pixel_to_bounds = self._compute_transform_matrix(
            self.image_bounding,
            self.scatter_bounding
        )

    def _compute_image_bounding(self) -> Bounding:
        """
        Computes the bounding box of the image in pixel coordinates.
        """
        image_width = self.image_size_em[0] * self.em_size
        image_height = self.image_size_em[1] * self.em_size
        return ((0, image_height), (image_width, 0))

    @staticmethod
    def _compute_transform_matrix(src: Bounding, dst: Bounding) -> AffineTransform:
        """
        Computes a 2x3 affine transformation matrix for mapping between source and destination bounds.
        """
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

    def to_pixel(self, point: Point2d) -> Point2d:
        """
        Transforms a point from scatter bounds to pixel coordinates.
        """
        point_homogeneous = np.array([point[0], point[1], 1.0])
        point_pixel = self.transform_bounds_to_pixel @ point_homogeneous
        return tuple(point_pixel)

    def to_pixels(self, points: list[Point2d]) -> list[Point2d]:
        """
        Transforms a list of points from scatter bounds to pixel coordinates.
        """
        if not points:
            return []
        points_array = np.array(points, dtype=np.float64)  # Shape: (N, 2)
        ones = np.ones((points_array.shape[0], 1), dtype=np.float64)
        points_homogeneous = np.hstack([points_array, ones])  # Shape: (N, 3)
        transformed_points = points_homogeneous @ self.transform_bounds_to_pixel.T  # Shape: (N, 2)
        return [tuple(pt) for pt in transformed_points]

    def to_bounds(self, point_pixel: Point2d) -> Point2d:
        """
        Transforms a point from pixel coordinates to scatter bounds.
        """
        point_homogeneous = np.array([point_pixel[0], point_pixel[1], 1.0])
        point_bounds = self.transform_pixel_to_bounds @ point_homogeneous
        return tuple(point_bounds)
