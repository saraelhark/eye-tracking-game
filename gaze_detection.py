import base64
import cv2
import numpy as np
import requests
from config import API_KEY, GAZE_DETECTION_URL

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

# You can add more helper functions here if needed

