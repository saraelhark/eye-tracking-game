"""This module contains the EyeTrackingGame class"""

import time
import cv2
import numpy as np
import random
import config as cfg
from utils.visualization import draw_gaze_point, show_timer, draw_target
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
        self.target_positions = self.generate_target_positions()
        self.targets_remaining = cfg.NUMBER_OF_TARGETS
        self.timer_start = None
        self.timer = 0
        self.best_score = 0

    def generate_target_positions(self):
        target_positions = []
        for _ in range(3):
            x = random.randint(100, cfg.WIDTH_OF_PLAYGROUND - 100)
            y = random.randint(100, cfg.HEIGHT_OF_PLAYGROUND - 100)
            target_positions.append((x, y))
        return target_positions

    def check_gaze_point(self, gaze_x, gaze_y):
        for i, target_pos in enumerate(self.target_positions):
            if np.sqrt((gaze_x - target_pos[0])**2 + (gaze_y - target_pos[1])**2) < 50:
                self.target_positions.pop(i)
                self.targets_remaining -= 1
                
        return False

    def detect_draw_gaze(self, frame):
        gaze_data_list = detect_gazes(frame)

        # Fill the frame with a white background
        frame = np.ones_like(frame) * 255

        # Draw the targets
        for target_pos in self.target_positions:
            frame = draw_target(frame, target_pos)

        if not gaze_data_list:
            return frame, False

        gaze = gaze_data_list[0]

        # Calculate gaze point
        dx, dy = calculate_gaze_point_displacements(gaze)
        gaze_x, gaze_y = calculate_gaze_point(dx, dy, cfg.WIDTH_OF_PLAYGROUND, cfg.HEIGHT_OF_PLAYGROUND)

        # Transform coordinates
        gaze_x, gaze_y = transform_coordinates(gaze_x, gaze_y, self.transformation_matrix, cfg.WIDTH_OF_PLAYGROUND, cfg.HEIGHT_OF_PLAYGROUND)

        # Apply Kalman filter
        filtered_point = self.kalman_filter.update(np.array([gaze_x, gaze_y]))
        filtered_x, filtered_y = map(int, filtered_point)

        # Draw the gaze point on the frame
        frame = draw_gaze_point(frame, (filtered_x, filtered_y))

        if cv2.waitKey(1) & 0xFF == ord(" "):
            self.is_tracking = True
            self.timer_start = time.time()

        if self.is_tracking:
            self.timer = (time.time() - self.timer_start) * 1000  # Convert to milliseconds
            show_timer(frame, f"{self.timer / 1000:.1f} s")
            if self.check_gaze_point(filtered_x, filtered_y):
                self.is_tracking = False
                self.timer_start = None
                self.timer = 0
        else:
            if self.best_score > 0:
                show_timer(frame, f"Best: {self.best_score / 1000:.1f} s")

        if self.targets_remaining == 0:
            if self.best_score == 0 or self.timer < self.best_score:
                self.best_score = self.timer
            self.is_tracking = False
            self.timer_start = None
            self.timer = 0
            self.target_positions = self.generate_target_positions()
            self.targets_remaining = cfg.NUMBER_OF_TARGETS

        return frame, self.targets_remaining == 0

    def run(self):
        text = "Press the spacebar to start the game"
        video_loop(self.cap, self.detect_draw_gaze, display_name="Eye Tracking Game - Targets", extra_text=text)
        