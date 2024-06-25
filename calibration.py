import cv2
import numpy as np
import time
from config import *
from gaze_detection import detect_gazes
from coordinate_transform import calculate_gaze_point_displacements, calculate_gaze_point
from visualization import draw_face_square, draw_ideal_square, draw_calibration_point
from video import video_loop


class AlignFace:
    def __init__(self, cap):
        self.cap = cap
        self.start_time = None
        self.face_aligned = False

    def check_face_in_ideal_square(self, gaze):
        face = gaze["face"]
        x_min = int(WIDTH_OF_PLAYGROUND / 2 - HEIGHT_OF_HUMAN_FACE / 2)
        x_max = int(WIDTH_OF_PLAYGROUND / 2 + HEIGHT_OF_HUMAN_FACE / 2)
        y_min = int(HEIGHT_OF_PLAYGROUND / 2 - HEIGHT_OF_HUMAN_FACE / 2)
        y_max = int(HEIGHT_OF_PLAYGROUND / 2 + HEIGHT_OF_HUMAN_FACE / 2)
        
        return (x_min < face["x"] - face["width"] / 2 < x_max and
                x_min < face["x"] + face["width"] / 2 < x_max and
                y_min < face["y"] - face["height"] / 2 < y_max and
                y_min < face["y"] + face["height"] / 2 < y_max)

    def frame_processing_func(self, frame):
        gazes = detect_gazes(frame)
        if len(gazes) > 0:
            gaze = gazes[0]
            draw_face_square(frame, gaze)
            draw_ideal_square(frame)
            if self.check_face_in_ideal_square(gaze):
                if self.start_time is None:
                    self.start_time = time.time()
                elif time.time() - self.start_time >= FACE_ALIGNMENT_TIME:
                    self.face_aligned = True
                    return frame, self.face_aligned
            else:
                self.start_time = None
                self.face_aligned = False

        return frame, self.face_aligned

    def run(self):
        print("Please align your face in the green square in the middle of the playground for 5 seconds.")
        video_loop(self.cap, self.frame_processing_func, "Align Face in Ideal Square")
        cv2.destroyAllWindows()


class CalibrateCorner:
    def __init__(self, cap, corner_x, corner_y, corner_name):
        self.cap = cap
        self.corner_x = corner_x
        self.corner_y = corner_y
        self.corner_name = corner_name
        self.gaze_points = []

    def frame_processing_func(self, frame):
        gazes = detect_gazes(frame)
        if len(gazes) > 0:
            gaze = gazes[0]
            draw_face_square(frame, gaze)
            draw_calibration_point(frame, (self.corner_x, self.corner_y))
            if cv2.waitKey(1) & 0xFF == ord(" "):
                dx, dy = calculate_gaze_point_displacements(gaze)
                gaze_x, gaze_y = calculate_gaze_point(dx, dy, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)
                self.gaze_points.append((gaze_x, gaze_y))
                print(f"Calibration point {len(self.gaze_points)} captured.")
            if len(self.gaze_points) == CALIBRATION_POINTS:
                return frame, True
        return frame, len(self.gaze_points) >= CALIBRATION_POINTS

    def calibrate(self):
        print(f"Look at the {self.corner_name} corner of the playground and press the spacebar.")
        video_loop(self.cap, self.frame_processing_func, "Gaze Calibration", destroy_windows=False)
        if self.gaze_points:
            return np.mean(self.gaze_points, axis=0)
        else:
            return (0.0, 0.0)


class CalibrateGazeMapping:
    def __init__(self, cap):
        self.cap = cap
        self.corners = [
            (0, 0, "top-left"),
            (WIDTH_OF_PLAYGROUND, 0, "top-right"),
            (0, HEIGHT_OF_PLAYGROUND, "bottom-left"),
            (WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND, "bottom-right"),
            (WIDTH_OF_PLAYGROUND // 2, HEIGHT_OF_PLAYGROUND // 2, "middle")
        ]

    def perform_calibration(self):
        src_points = []
        for corner in self.corners:
            x, y = CalibrateCorner(self.cap, *corner).calibrate()
            src_points.append([x, y])

        dst_points = np.float32([[0, 0],
                                [WIDTH_OF_PLAYGROUND, 0],
                                [0, HEIGHT_OF_PLAYGROUND],
                                [WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND],
                                [WIDTH_OF_PLAYGROUND // 2, HEIGHT_OF_PLAYGROUND // 2]])

        # Convert to numpy arrays with float32 data type
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        transformation_matrix, _ = cv2.findHomography(src_points, dst_points)
        cv2.destroyAllWindows()
        return transformation_matrix

