import base64
import cv2
import numpy as np
import requests
from config import *
from coordinate_transform import *
from visualization import draw_face_square, draw_gaze_point, display_frame, flip_frame

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

def get_gaze_data(gaze):
    """
    Extract relevant gaze data from the gaze prediction.

    Args:
    gaze (dict): The gaze prediction dictionary.

    Returns:
    dict: A dictionary containing relevant gaze data (yaw, pitch, face).
    """
    return {
        "yaw": gaze.get("yaw", 0),
        "pitch": gaze.get("pitch", 0),
        "face": gaze.get("face", {})
    }

def process_frame(frame):
    """
    Process a frame to detect gazes and extract gaze data.

    Args:
    frame (numpy.ndarray): The input frame to process.

    Returns:
    list: A list of processed gaze data dictionaries.
    """
    gazes = detect_gazes(frame)
    return [get_gaze_data(gaze) for gaze in gazes]


def eyes_tracking_positions(cap, transformation_matrix):
    # Initialize gaze history for smoothing
    gaze_history = []
    window_size = GAZE_HISTORY_WINDOW_SIZE

    # Initialize Kalman filter with the center of the image
    initial_state = [WIDTH_OF_PLAYGROUND // 2, HEIGHT_OF_PLAYGROUND // 2]
    kalman_filter = KalmanFilter(initial_state)

    while True:
        _, frame = cap.read()
        frame = flip_frame(frame)

        gaze_data_list = process_frame(frame)

        if not gaze_data_list:
            continue

        gaze = gaze_data_list[0]  # Assuming we're only interested in the first detected gaze

        # Draw face square
        frame = draw_face_square(frame, gaze)

        # Calculate gaze point
        dx, dy = calculate_gaze_point_displacements(gaze)
        gaze_x, gaze_y = calculate_gaze_point(dx, dy, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)

        # Transform coordinates
        gaze_x, gaze_y = transform_coordinates(gaze_x, gaze_y, transformation_matrix, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)

        # Apply moving average filter
        #filtered_x, filtered_y = apply_moving_average_filter(gaze_history, (gaze_x, gaze_y), window_size)

        #filtered_x, filtered_y = apply_median_filter(gaze_history, (gaze_x, gaze_y), window_size)

        #filtered_x, filtered_y = adaptive_weighted_moving_average(gaze_history, (gaze_x, gaze_y), window_size)

        filtered_point = kalman_filter.update(np.array([gaze_x, gaze_y]))

        filtered_x, filtered_y = map(int, filtered_point)

        filtered_x = max(0, min(filtered_x, WIDTH_OF_PLAYGROUND - 1))
        filtered_y = max(0, min(filtered_y, HEIGHT_OF_PLAYGROUND - 1))

        print(f"Raw gaze point: ({gaze_x}, {gaze_y})")
        print(f"Filtered gaze point: ({filtered_x}, {filtered_y})")

        # Draw gaze point
        frame = draw_gaze_point(frame, (filtered_x, filtered_y))

        # Display the frame
        display_frame("Gaze Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()