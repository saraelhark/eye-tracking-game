import base64
import cv2
import numpy as np
import requests
from config import *
from coordinate_transform import *
from visualization import draw_face_square, draw_gaze_point
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

    def detect_draw_gaze(self, frame):
        gaze_data_list = detect_gazes(frame)

        if not gaze_data_list:
            return frame, False

        gaze = gaze_data_list[0]  # Assuming we're only interested in the first detected gaze

        # Draw face square
        frame = draw_face_square(frame, gaze)

        # Calculate gaze point
        dx, dy = calculate_gaze_point_displacements(gaze)
        gaze_x, gaze_y = calculate_gaze_point(dx, dy, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)

        # Transform coordinates
        gaze_x, gaze_y = transform_coordinates(gaze_x, gaze_y, self.transformation_matrix, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)

        # Apply Kalman filter
        filtered_point = self.kalman_filter.update(np.array([gaze_x, gaze_y]))

        filtered_x, filtered_y = map(int, filtered_point)

        filtered_x = max(0, min(filtered_x, WIDTH_OF_PLAYGROUND - 1))
        filtered_y = max(0, min(filtered_y, HEIGHT_OF_PLAYGROUND - 1))

        print(f"Raw gaze point: ({gaze_x}, {gaze_y})")
        print(f"Filtered gaze point: ({filtered_x}, {filtered_y})")

        # Draw gaze point
        frame = draw_gaze_point(frame, (filtered_x, filtered_y))

        return frame, False

    def run(self):
        video_loop(self.cap, self.detect_draw_gaze, "Gaze Tracking")

