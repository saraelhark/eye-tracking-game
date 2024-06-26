import cv2
import numpy as np
import time
from config import *
from gaze_detection import detect_gazes, draw_gaze_point
from coordinate_transform import *
from visualization import *
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
        text = "Please align your face in the green square for 5 seconds."
        print(text)
        video_loop(self.cap, self.frame_processing_func, display_name="Align Face in Ideal Square", extra_text=text)


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
        text = f"Look at the {self.corner_name} corner of the playground and press the spacebar."
        print(text)
        video_loop(self.cap, self.frame_processing_func, display_name="Gaze Calibration", extra_text=text, destroy_windows=False)
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


class CheckGazeAccuracyForTarget:
    def __init__(self, cap, transformation_matrix, target_point):
        self.cap = cap
        self.transformation_matrix = transformation_matrix
        self.target_point = target_point
        self.gaze_points = []
        self.target_start_time = None
        self.target_duration = 5  # Seconds
        self.started = False

        self.gaze_history = []
        self.kalman_filter = KalmanFilter([WIDTH_OF_PLAYGROUND // 2, HEIGHT_OF_PLAYGROUND // 2])

    def frame_processing_func(self, frame):
        gazes = detect_gazes(frame)
        if len(gazes) > 0:
            gaze = gazes[0]
            frame = draw_face_square(frame, gaze)

            dx, dy = calculate_gaze_point_displacements(gaze)
            gaze_x, gaze_y = calculate_gaze_point(dx, dy, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)
            gaze_x, gaze_y = transform_coordinates(gaze_x, gaze_y, self.transformation_matrix, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)

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
        text = f"Look at the target point and press the spacebar to start. Hold for {self.target_duration} seconds."
        print(text)
        video_loop(self.cap, self.frame_processing_func, "Gaze Accuracy Check", text, destroy_windows=False)

        accuracy = self.calculate_accuracy()
        print(f"Accuracy for this target: {accuracy:.2f}%")
        return accuracy

    def calculate_accuracy(self):
        if not self.gaze_points:
            return 0.0

        target_x, target_y = self.target_point
        total_distance = 0
        for gaze_x, gaze_y in self.gaze_points:
            distance = np.sqrt((gaze_x - target_x) ** 2 + (gaze_y - target_y) ** 2)
            total_distance += distance

        avg_distance = total_distance / len(self.gaze_points)
        max_distance = np.sqrt(WIDTH_OF_PLAYGROUND ** 2 + HEIGHT_OF_PLAYGROUND ** 2)
        accuracy = (1 - avg_distance / max_distance) * 100
        return accuracy


class CheckGazeAccuracy:
    def __init__(self, cap, transformation_matrix, target_points):
        self.cap = cap
        self.transformation_matrix = transformation_matrix
        self.target_points = target_points
        self.overall_accuracy = 0.0

    def run(self):
        for target_point in self.target_points:
            checker = CheckGazeAccuracyForTarget(self.cap, self.transformation_matrix, target_point)
            accuracy = checker.run()
            self.overall_accuracy += accuracy

        cv2.destroyAllWindows()
        self.overall_accuracy /= len(self.target_points)
        print(f"Overall gaze detection accuracy: {self.overall_accuracy:.2f}%")
        return self.overall_accuracy
