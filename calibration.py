import cv2
import numpy as np
import time
from config import *
from gaze_detection import detect_gazes
from coordinate_transform import calculate_gaze_point_displacements, calculate_gaze_point
from visualization import draw_face_square, draw_ideal_square, draw_calibration_point
from video import video_loop

def check_face_in_ideal_square(gaze):
    face = gaze["face"]
    x_min = int(WIDTH_OF_PLAYGROUND / 2 - HEIGHT_OF_HUMAN_FACE / 2)
    x_max = int(WIDTH_OF_PLAYGROUND / 2 + HEIGHT_OF_HUMAN_FACE / 2)
    y_min = int(HEIGHT_OF_PLAYGROUND / 2 - HEIGHT_OF_HUMAN_FACE / 2)
    y_max = int(HEIGHT_OF_PLAYGROUND / 2 + HEIGHT_OF_HUMAN_FACE / 2)
    
    return (x_min < face["x"] - face["width"] / 2 < x_max and
            x_min < face["x"] + face["width"] / 2 < x_max and
            y_min < face["y"] - face["height"] / 2 < y_max and
            y_min < face["y"] + face["height"] / 2 < y_max)

def align_face_in_ideal_square(cap):
    print("Please align your face in the green square in the middle of the playground for 5 seconds.")
    start_time = None

    def frame_processing_func(frame, start_time):
        gazes = detect_gazes(frame)
        if len(gazes) > 0:
            gaze = gazes[0]
            draw_face_square(frame, gaze)
            draw_ideal_square(frame)
            if check_face_in_ideal_square(gaze):
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= FACE_ALIGNMENT_TIME:
                    return frame, start_time
            else:
                start_time = None
        return frame, start_time

    video_loop(cap, lambda frame: frame_processing_func(frame, start_time), "Align Face in Ideal Square")


def calibrate_corner(cap, corner_x, corner_y, corner_name):
    print(f"Look at the {corner_name} corner of the playground and press the spacebar.")
    gaze_points = []

    def frame_processing_func(frame):
        gazes = detect_gazes(frame)
        if len(gazes) > 0:
            gaze = gazes[0]
            draw_face_square(frame, gaze)
            draw_calibration_point(frame, (corner_x, corner_y))
            if cv2.waitKey(1) & 0xFF == ord(" "):
                dx, dy = calculate_gaze_point_displacements(gaze)
                gaze_x, gaze_y = calculate_gaze_point(dx, dy, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)
                gaze_points.append((gaze_x, gaze_y))
                print(f"Calibration point {len(gaze_points)} captured.")
        return frame

    video_loop(cap, frame_processing_func, "Gaze Calibration", False)
    return np.mean(gaze_points, axis=0)


def calibrate_gaze_mapping(cap):
    corners = [
        (0, 0, "top-left"),
        (WIDTH_OF_PLAYGROUND, 0, "top-right"),
        (0, HEIGHT_OF_PLAYGROUND, "bottom-left"),
        (WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND, "bottom-right"),
        (WIDTH_OF_PLAYGROUND // 2, HEIGHT_OF_PLAYGROUND // 2, "middle")
    ]
    
    src_points = []
    for corner in corners:
        x, y = calibrate_corner(cap, *corner)
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
    return transformation_matrix

def perform_calibration(cap):
    transformation_matrix = calibrate_gaze_mapping(cap)
    cv2.destroyAllWindows()
    return transformation_matrix
