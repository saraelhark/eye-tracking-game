"""This module contains the EyeTrackingGame class"""

import time
import cv2
import numpy as np
import config as cfg
from utils.visualization import draw_gaze_point, show_timer
from utils.video import video_loop
from utils.coordinate_transform import transform_coordinates, calculate_gaze_point, calculate_gaze_point_displacements
from utils.filters import apply_moving_average_filter, KalmanFilter
from utils.gaze_detection import detect_gazes


class EyeTrackingGame:
    def __init__(self, cap, transformation_matrix):
        self.cap = cap
        self.transformation_matrix = transformation_matrix
        self.gaze_history = []
        self.window_size = cfg.GAZE_HISTORY_WINDOW_SIZE
        self.kalman_filter = KalmanFilter([cfg.WIDTH_OF_PLAYGROUND // 2, cfg.HEIGHT_OF_PLAYGROUND // 2])
        self.is_tracking = False
        self.n_shape_points = self.create_n_shape()
        self.timer_start = None
        self.timer = 0
        self.tracked_points = []

    def create_n_shape(self):
        # Define the coordinates of the "N" shape
        n_shape_points = [
            (100, cfg.HEIGHT_OF_PLAYGROUND - 100),
            (100, 100),
            (cfg.WIDTH_OF_PLAYGROUND - 100, cfg.HEIGHT_OF_PLAYGROUND - 100),
            (cfg.WIDTH_OF_PLAYGROUND - 100, 100)
        ]
        return n_shape_points
    
    def check_gaze_point(self, gaze_x, gaze_y):
        # Check if the gaze point overlaps with the "N" shape
        gaze_point = (gaze_x, gaze_y)
        is_inside = cv2.pointPolygonTest(np.array(self.n_shape_points, dtype=np.int32), gaze_point, False) >= 0

        if is_inside and gaze_point not in self.tracked_points:
            # Find the closest point on the "N" shape
            closest_point = min(self.n_shape_points, key=lambda p: np.linalg.norm(np.array(p) - np.array(gaze_point)))
            self.tracked_points.append(closest_point)

    def draw_tracked_points(self, img):
        for point in set(self.tracked_points):
            img = cv2.circle(img, point, cfg.POLYLINE_THICKNESS // 2, (0,255,0), -1)
        return img

    def detect_draw_gaze(self, frame):
        gaze_data_list = detect_gazes(frame)

        # Fill the frame with a white background
        frame = np.ones_like(frame) * 255

        # Draw the "N" shape
        frame = cv2.polylines(frame, [np.array(self.n_shape_points, dtype=np.int32)], False, (0, 0, 255), cfg.POLYLINE_THICKNESS)

        if not gaze_data_list:
            return frame, False

        gaze = gaze_data_list[0]

        # Calculate gaze point
        dx, dy = calculate_gaze_point_displacements(gaze)
        gaze_x, gaze_y = calculate_gaze_point(dx, dy, cfg.WIDTH_OF_PLAYGROUND, cfg.HEIGHT_OF_PLAYGROUND)

        # Transform coordinates
        gaze_x, gaze_y = transform_coordinates(gaze_x, gaze_y, self.transformation_matrix, cfg.WIDTH_OF_PLAYGROUND, cfg.HEIGHT_OF_PLAYGROUND)

        # Apply Kalman filter
        #filtered_point = self.kalman_filter.update(np.array([gaze_x, gaze_y]))
        #filtered_x, filtered_y = map(int, filtered_point)

        # apply weighted average filter
        filtered_x, filtered_y = apply_moving_average_filter(self.gaze_history, (gaze_x, gaze_y), self.window_size)

        # Draw the gaze point on the frame
        frame = draw_gaze_point(frame, (filtered_x, filtered_y))

        self.check_gaze_point(filtered_x, filtered_y)
        self.draw_tracked_points(frame)

        if cv2.waitKey(1) & 0xFF == ord(" "):
            self.is_tracking = True
            self.timer_start = time.time()

        if self.is_tracking:
            self.timer = (time.time() - self.timer_start) * 1000  # Convert to milliseconds
            show_timer(frame, f"{self.timer / 1000:.1f} s")

        return frame, False

    def run(self):
        text = "Press the spacebar to start tracking the shape"
        video_loop(self.cap, self.detect_draw_gaze, display_name="Gaze Tracking", extra_text=text)
