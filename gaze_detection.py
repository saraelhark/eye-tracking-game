import base64
import cv2
import time
import numpy as np
import requests
from config import *
from coordinate_transform import *
from visualization import draw_face_square, draw_gaze_point, show_timer
from video import video_loop

def detect_gazes(frame: np.ndarray):
    """
    Detect gazes in the given frame using the Roboflow API.

    Args:
    frame (numpy.ndarray): The input frame to detect gazes in.

    Returns:
    list: A list of detected gazes, where each gaze is a dictionary containing gaze information.
    """
    # Encode the frame as a JPEG image
    _, img_encode = cv2.imencode(".jpg", frame)
    
    # Convert the encoded image to base64
    img_base64 = base64.b64encode(img_encode).decode("utf-8")

    # Prepare the request payload
    payload = {
        "api_key": API_KEY,
        "image": {"type": "base64", "value": img_base64},
    }

    # Send the request to the Roboflow API
    response = requests.post(GAZE_DETECTION_URL, json=payload)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the predictions from the response
        predictions = response.json()[0]["predictions"]
        return predictions
    else:
        print(f"Error in gaze detection: {response.status_code} - {response.text}")
        return []


class EyesTrackingPositions:
    def __init__(self, cap, transformation_matrix):
        self.cap = cap
        self.transformation_matrix = transformation_matrix
        self.gaze_history = []
        self.window_size = GAZE_HISTORY_WINDOW_SIZE
        self.kalman_filter = KalmanFilter([WIDTH_OF_PLAYGROUND // 2, HEIGHT_OF_PLAYGROUND // 2])
        self.is_tracking = False
        self.n_shape_points = self.create_n_shape()
        self.timer_start = None
        self.timer = 0
        self.tracked_points = []

    def create_n_shape(self):
        # Define the coordinates of the "N" shape
        n_shape_points = [
            (100, HEIGHT_OF_PLAYGROUND - 100),
            (100, 100),
            (WIDTH_OF_PLAYGROUND - 100, HEIGHT_OF_PLAYGROUND - 100),
            (WIDTH_OF_PLAYGROUND - 100, 100)
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
            img = cv2.circle(img, point, POLYLINE_THICKNESS // 2, (0,255,0), -1)
        return img

    def detect_draw_gaze(self, frame):
        gaze_data_list = detect_gazes(frame)

        # Fill the frame with a white background
        frame = np.ones_like(frame) * 255

        # Draw the "N" shape
        frame = cv2.polylines(frame, [np.array(self.n_shape_points, dtype=np.int32)], False, (0, 0, 255), POLYLINE_THICKNESS)

        if not gaze_data_list:
            return frame, False

        gaze = gaze_data_list[0]

        # Calculate gaze point
        dx, dy = calculate_gaze_point_displacements(gaze)
        gaze_x, gaze_y = calculate_gaze_point(dx, dy, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)

        # Transform coordinates
        gaze_x, gaze_y = transform_coordinates(gaze_x, gaze_y, self.transformation_matrix, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)

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
