import cv2
import numpy as np

class ViewTransformer:
    def __init__(self):
        # Standard tactical pitch layout dimensions (in meters)
        court_width = 68
        court_length = 23.32

        # 4 key points on the screen (source coordinates in pixels)
        self.pixel_vertices = np.array([
            [110, 1035], 
            [265, 275], 
            [910, 260], 
            [1640, 1035]
        ], dtype=np.float32)

        # Target coordinates in meters
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ], dtype=np.float32)

        # Calculate Perspective Transform Matrix
        self.perpective_transform = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = int(point[0]), int(point[1])
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perpective_transform)
        return transformed_point.reshape(-1, 2)[0]