import base64

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
import os

load_dotenv()

IMG_PATH = "image.jpg"
API_KEY = os.environ.get("API_KEY")
DISTANCE_TO_OBJECT = 300  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm
GAZE_DETECTION_URL = (
    "http://127.0.0.1:9001/gaze/gaze_detection?api_key=" + API_KEY
)

def detect_gazes(frame: np.ndarray):
    img_encode = cv2.imencode(".jpg", frame)[1]
    img_base64 = base64.b64encode(img_encode)
    resp = requests.post(
        GAZE_DETECTION_URL,
        json={
            "api_key": API_KEY,
            "image": {"type": "base64", "value": img_base64.decode("utf-8")},
        },
    )

    gazes = resp.json()[0]["predictions"]
    return gazes


def draw_gaze(img: np.ndarray, gaze: dict):
    # draw face bounding box
    face = gaze["face"]
    x_min = int(face["x"] - face["width"] / 2)
    x_max = int(face["x"] + face["width"] / 2)
    y_min = int(face["y"] - face["height"] / 2)
    y_max = int(face["y"] + face["height"] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    image_height, image_width = img.shape[:2]

    length_per_pixel = HEIGHT_OF_HUMAN_FACE / gaze["face"]["height"]

    dx = -DISTANCE_TO_OBJECT * np.tan(gaze['yaw']) / length_per_pixel
    # 100000000 is used to denote out of bounds
    dx = dx if not np.isnan(dx) else 100000000
    dy = -DISTANCE_TO_OBJECT * np.arccos(gaze['yaw']) * np.tan(gaze['pitch']) / length_per_pixel
    dy = dy if not np.isnan(dy) else 100000000

    # invert left-right the whole frame
    img = cv2.flip(img, 1)

    # Adjust the gaze_point_x coordinate for the flipped frame
    gaze_point_x = image_width - int(image_width / 2 + dx)
    gaze_point_y = int(image_height / 2 + dy)
    gaze_point = (gaze_point_x, gaze_point_y)

    cv2.circle(img, gaze_point, 25, (0, 0, 255), -1)

    return img


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # check if the webcam is opened correctly
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        _, frame = cap.read()

        gazes = detect_gazes(frame)

        if len(gazes) == 0:
            continue

        # draw face & gaze
        gaze = gazes[0]
        frame = draw_gaze(frame, gaze)

        cv2.imshow("gaze", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

