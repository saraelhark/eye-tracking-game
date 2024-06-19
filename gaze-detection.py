import base64
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
import os

load_dotenv()

IMG_PATH = "image.jpg"
API_KEY = os.environ.get("API_KEY")
DISTANCE_TO_OBJECT = 500  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm
GAZE_DETECTION_URL = (
    "http://127.0.0.1:9001/gaze/gaze_detection?api_key=" + API_KEY
)

# Playground size
WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND = 640, 480

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


def calculate_gaze_point(gaze_x_raw, gaze_y_raw, image_width, image_height):

    gaze_point_x = image_width / 2 + gaze_x_raw
    gaze_point_y = image_height / 2 + gaze_y_raw

    return gaze_point_x, gaze_point_y


def calculate_gaze_point_displacements(gaze):
    length_per_pixel = HEIGHT_OF_HUMAN_FACE / gaze["face"]["height"]

    dx = -DISTANCE_TO_OBJECT * np.tan(gaze['yaw']) / length_per_pixel
    dx = dx if not np.isnan(dx) else 100000000

    yaw_cos = np.clip(gaze['yaw'], -1, 1)
    dy = -DISTANCE_TO_OBJECT * np.arccos(yaw_cos) * np.tan(gaze['pitch']) / length_per_pixel
    dy = dy if not np.isnan(dy) else 100000000

    return dx, dy


def draw_face_square(img, gaze):
    # draw face bounding box
    face = gaze["face"]
    x_min = int(face["x"] - face["width"] / 2)
    x_max = int(face["x"] + face["width"] / 2)
    y_min = int(face["y"] - face["height"] / 2)
    y_max = int(face["y"] + face["height"] / 2) 
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    return img


def transform_coordinates(gaze_x_raw, gaze_y_raw, transformation_matrix, image_width, image_height):
    # Reshape the input array to the expected shape (1, 1, 2)
    input_array = np.array([[gaze_x_raw, gaze_y_raw]]).reshape(1, 1, 2)

    # Apply the transformation matrix to the raw gaze point coordinates
    adjusted_x, adjusted_y = cv2.perspectiveTransform(input_array, transformation_matrix)[0][0]

    # Ensure the adjusted coordinates are within the frame boundaries
    adjusted_x = max(0, min(adjusted_x, image_width - 1))
    adjusted_y = max(0, min(adjusted_y, image_height - 1))

    return int(adjusted_x), int(adjusted_y)


def draw_gaze_point(img, gaze, transformation_matrix):
    image_height, image_width = img.shape[:2]

    gaze_point_x, gaze_point_y = calculate_gaze_point_displacements(gaze)
    gaze_point_x, gaze_point_y = calculate_gaze_point(gaze_point_x, gaze_point_y, image_width, image_height)

    print("Raw", "gaze_point_x: ", gaze_point_x, " gaze_point_y: ", gaze_point_y)

    gaze_point_x, gaze_point_y = transform_coordinates(gaze_point_x, gaze_point_y, transformation_matrix, image_width, image_height)
    
    print("gaze_point_scaled_x: ", gaze_point_x, " gaze_point_scaled_y: ", gaze_point_y)
    
    gaze_point = (gaze_point_x, gaze_point_y)

    cv2.circle(img, gaze_point, 10, (0, 0, 255), -1)

    return img


def calibrate_corner(corner_x, corner_y, corner_name):
        image_height, image_width = HEIGHT_OF_PLAYGROUND, WIDTH_OF_PLAYGROUND
        print(f"Look at the {corner_name} corner of the playground and press the spacebar.")
        while True:
            _, frame = cap.read()
            frame = cv2.flip(frame, 1)
            gazes = detect_gazes(frame)
            if len(gazes) > 0:
                gaze = gazes[0]
                draw_face_square(frame, gaze)

                # add a red dot in the target corner of the playground
                cv2.circle(frame, (corner_x, corner_y), 20, (0, 0, 255), -1)

                cv2.imshow("gaze calib", frame)

                dx, dy = calculate_gaze_point_displacements(gaze)
                gaze_x, gaze_y = calculate_gaze_point(dx, dy, image_width, image_height)

                if cv2.waitKey(1) & 0xFF == ord(" "):
                    return gaze_x, gaze_y


def calibrate_gaze_mapping():
    image_height, image_width = HEIGHT_OF_PLAYGROUND, WIDTH_OF_PLAYGROUND

    top_left_x, top_left_y = calibrate_corner(0, 0, "top-left")
    top_right_x, top_right_y = calibrate_corner(image_width, 0, "top-right")
    bottom_left_x, bottom_left_y = calibrate_corner(0, image_height, "bottom-left")
    bottom_right_x, bottom_right_y = calibrate_corner(image_width, image_height, "bottom-right")
    
    # close the window
    cv2.destroyAllWindows()

    # Define the source and destination points
    src_points = np.float32([[top_left_x, top_left_y],
                             [top_right_x, top_right_y],
                             [bottom_left_x, bottom_left_y],
                             [bottom_right_x, bottom_right_y]])

    dst_points = np.float32([[0, 0],
                             [image_width, 0],
                             [0, image_height],
                             [image_width, image_height]])

    # Calculate the transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    return transformation_matrix


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # check if the webcam is opened correctly
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH_OF_PLAYGROUND)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT_OF_PLAYGROUND)

    # calibrate gaze mapping
    transformation_matrix = calibrate_gaze_mapping()

    while True:
        _, frame = cap.read()
        
        # invert left-right the whole frame
        frame = cv2.flip(frame, 1)

        gazes = detect_gazes(frame)

        if len(gazes) == 0:
            continue
        
        gaze = gazes[0]

        # draw face square and gaze point
        frame = draw_face_square(frame, gaze)

        frame = draw_gaze_point(frame, gaze, transformation_matrix)

        cv2.imshow("gaze", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

