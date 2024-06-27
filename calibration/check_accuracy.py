""" This module is used to check the accuracy of the gaze detection system. """

import time
import cv2
import numpy as np
import config as cfg
import logging
from utils.coordinate_transform import transform_coordinates, calculate_gaze_point_displacements, calculate_gaze_point
from utils.filters import KalmanFilter
from utils.gaze_detection import detect_gazes
from utils.visualization import draw_face_square, draw_calibration_point, draw_gaze_point
from utils.video import video_loop

logging.basicConfig(level=logging.INFO)

class CheckGazeAccuracyForTarget:
    """
    A class that checks the gaze accuracy for a target point.

    Args:
        cap (object): The video capture object.
        transformation_matrix (numpy.ndarray): The transformation matrix for coordinate transformation.
        target_point (tuple): The coordinates of the target point.

    Attributes:
        cap (object): The video capture object.
        transformation_matrix (numpy.ndarray): The transformation matrix for coordinate transformation.
        target_point (tuple): The coordinates of the target point.
        gaze_points (list): A list to store the gaze points.
        target_start_time (float): The start time of the target.
        target_duration (float): The duration for which the target should be held.
        started (bool): A flag indicating if the target has started.
        gaze_history (list): A list to store the gaze history.
        kalman_filter (KalmanFilter): An instance of the KalmanFilter class.

    """

    def __init__(self, cap, transformation_matrix, target_point):
        self.cap = cap
        self.transformation_matrix = transformation_matrix
        self.target_point = target_point
        self.gaze_points = []
        self.target_start_time = None
        self.target_duration = cfg.ACCURACY_TARGET_DURATION
        self.started = False

        self.gaze_history = []
        self.kalman_filter = KalmanFilter([cfg.WIDTH_OF_PLAYGROUND // 2, cfg.HEIGHT_OF_PLAYGROUND // 2])

    def frame_processing_func(self, frame):
        """
        Process each frame of the video.

        Args:
            frame (numpy.ndarray): The frame of the video.

        Returns:
            tuple: A tuple containing the processed frame and a flag indicating if the target duration has elapsed.

        """
        gazes = detect_gazes(frame)
        if len(gazes) > 0:
            gaze = gazes[0]
            frame = draw_face_square(frame, gaze)
            image_width, image_height = frame.shape[:2]
            dx, dy = calculate_gaze_point_displacements(gaze)
            gaze_x, gaze_y = calculate_gaze_point(dx, dy, image_width, image_height)
            gaze_x, gaze_y = transform_coordinates(gaze_x, gaze_y, self.transformation_matrix, image_width, image_height)

            target_x, target_y = self.target_point
            draw_calibration_point(frame, (target_x, target_y))

            # add kalman filter
            filtered_point = self.kalman_filter.update(np.array([gaze_x, gaze_y]))
            gaze_x, gaze_y = map(int, filtered_point)

            # Draw gaze point
            frame = draw_gaze_point(frame, (gaze_x, gaze_y))

            if self.started:
                self.gaze_points.append((gaze_x, gaze_y))

                if self.target_start_time is None:
                    self.target_start_time = time.time()
                elif time.time() - self.target_start_time >= self.target_duration:
                    self.started = False
                    return frame, True

        if cv2.waitKey(1) & 0xFF == ord(" "):
            self.started = True
            self.gaze_points = []

        return frame, False

    def run(self):
        """
        Run the gaze accuracy check.

        Returns:
            float: The accuracy for the target.

        """
        text = f"Look at the target point and press the spacebar to start. Hold for {self.target_duration} seconds."
        video_loop(self.cap, self.frame_processing_func, "Gaze Accuracy Check", text, destroy_windows=False)

        accuracy = self.calculate_accuracy()
        logging.info(f"Accuracy for this target: {accuracy:.2f}%")
        return accuracy

    def calculate_accuracy(self):
        """
        Calculate the accuracy for the target.

        Returns:
            float: The accuracy for the target.

        """
        if not self.gaze_points:
            return 0.0

        target_x, target_y = self.target_point
        total_distance = 0
        for gaze_x, gaze_y in self.gaze_points:
            distance = np.sqrt((gaze_x - target_x) ** 2 + (gaze_y - target_y) ** 2)
            total_distance += distance

        avg_distance = total_distance / len(self.gaze_points)
        max_distance = np.sqrt(cfg.WIDTH_OF_PLAYGROUND ** 2 + cfg.HEIGHT_OF_PLAYGROUND ** 2)
        accuracy = (1 - avg_distance / max_distance) * 100
        return accuracy


class CheckGazeAccuracy:
    """
    Class to check the gaze accuracy for a set of target points.

    Args:
        cap (object): The video capture object.
        transformation_matrix (numpy.ndarray): The transformation matrix for gaze calibration.
        target_points (list): List of target points to check gaze accuracy.

    Attributes:
        cap (object): The video capture object.
        transformation_matrix (numpy.ndarray): The transformation matrix for gaze calibration.
        target_points (list): List of target points to check gaze accuracy.
        overall_accuracy (float): The overall gaze detection accuracy.

    Methods:
        run(): Runs the gaze accuracy check for each target point.
    """

    def __init__(self, cap, transformation_matrix, target_points):
        self.cap = cap
        self.transformation_matrix = transformation_matrix
        self.target_points = target_points
        self.overall_accuracy = 0.0

    def run(self):
        """
        Runs the gaze accuracy check for each target point.

        Returns:
            float: The overall gaze detection accuracy.
        """
        for target_point in self.target_points:
            checker = CheckGazeAccuracyForTarget(self.cap, self.transformation_matrix, target_point)
            accuracy = checker.run()
            self.overall_accuracy += accuracy

        cv2.destroyAllWindows()
        self.overall_accuracy /= len(self.target_points)
        logging.info(f"Overall gaze detection accuracy: {self.overall_accuracy:.2f}%")
        return self.overall_accuracy
