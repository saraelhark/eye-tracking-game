"""This module contains classes for calibrating the gaze mapping."""

import cv2
import numpy as np
import config as cfg
from utils.gaze_detection import detect_gazes
from utils.coordinate_transform import calculate_gaze_point_displacements, calculate_gaze_point
from utils.visualization import draw_face_square, draw_calibration_point
from utils.video import video_loop


class CalibrateCorner:
    """Class for calibrating a specific corner of the gaze mapping."""

    def __init__(self, cap, corner_x, corner_y, corner_name):
        """
        Initialize the CalibrateCorner object.

        Args:
            cap: The video capture object.
            corner_x: The x-coordinate of the corner.
            corner_y: The y-coordinate of the corner.
            corner_name: The name of the corner.
        """
        self.cap = cap
        self.corner_x = corner_x
        self.corner_y = corner_y
        self.corner_name = corner_name
        self.gaze_points = []

    def frame_processing_func(self, frame):
        """
        Process each frame of the video capture.

        Args:
            frame: The current frame of the video capture.

        Returns:
            A tuple containing the processed frame and a boolean indicating if the calibration is complete.
        """
        gazes = detect_gazes(frame)
        if len(gazes) > 0:
            gaze = gazes[0]
            draw_face_square(frame, gaze)
            draw_calibration_point(frame, (self.corner_x, self.corner_y))
            if cv2.waitKey(1) & 0xFF == ord(" "):
                dx, dy = calculate_gaze_point_displacements(gaze)
                gaze_x, gaze_y = calculate_gaze_point(dx, dy, cfg.WIDTH_OF_PLAYGROUND, cfg.HEIGHT_OF_PLAYGROUND)
                self.gaze_points.append((gaze_x, gaze_y))
                print(f"Calibration point {len(self.gaze_points)} captured.")
            if len(self.gaze_points) == cfg.CALIBRATION_POINTS:
                return frame, True
        return frame, len(self.gaze_points) >= cfg.CALIBRATION_POINTS

    def calibrate(self):
        """
        Perform the calibration for the corner.

        Returns:
            The mean gaze point coordinates for the corner.
        """
        text = f"Look at the {self.corner_name} corner of the playground and press the spacebar."
        print(text)
        video_loop(self.cap, self.frame_processing_func, display_name="Gaze Calibration", extra_text=text, destroy_windows=False)
        if self.gaze_points:
            return np.mean(self.gaze_points, axis=0)

        return (0.0, 0.0)


class CalibrateGazeMapping:
    """Class for calibrating the gaze mapping."""

    def __init__(self, cap):
        """
        Initialize the CalibrateGazeMapping object.

        Args:
            cap: The video capture object.
        """
        self.cap = cap
        self.corners = [
            (0, 0, "top-left"),
            (cfg.WIDTH_OF_PLAYGROUND, 0, "top-right"),
            (0, cfg.HEIGHT_OF_PLAYGROUND, "bottom-left"),
            (cfg.WIDTH_OF_PLAYGROUND, cfg.HEIGHT_OF_PLAYGROUND, "bottom-right"),
            (cfg.WIDTH_OF_PLAYGROUND // 2, cfg.HEIGHT_OF_PLAYGROUND // 2, "middle")
        ]

    def perform_calibration(self):
        """
        Perform the calibration for all corners.

        Returns:
            The transformation matrix for the gaze mapping.
        """
        src_points = []
        for corner in self.corners:
            x, y = CalibrateCorner(self.cap, *corner).calibrate()
            src_points.append([x, y])

        dst_points = np.float32([[0, 0],
                                [cfg.WIDTH_OF_PLAYGROUND, 0],
                                [0, cfg.HEIGHT_OF_PLAYGROUND],
                                [cfg.WIDTH_OF_PLAYGROUND, cfg.HEIGHT_OF_PLAYGROUND],
                                [cfg.WIDTH_OF_PLAYGROUND // 2, cfg.HEIGHT_OF_PLAYGROUND // 2]])

        # Convert to numpy arrays with float32 data type
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        transformation_matrix, _ = cv2.findHomography(src_points, dst_points)
        cv2.destroyAllWindows()
        return transformation_matrix
