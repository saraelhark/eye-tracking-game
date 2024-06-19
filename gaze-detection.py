import base64
import cv2
import numpy as np
import requests
from dotenv import load_dotenv
import os
import time

load_dotenv()

IMG_PATH = "image.jpg"
API_KEY = os.environ.get("API_KEY")
DISTANCE_TO_OBJECT = 400  # mm
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

# function to draw a green ideal square in the middle and check if the user face is in the frame
def draw_ideal_square(img):
    # draw ideal square
    x_min = int(WIDTH_OF_PLAYGROUND / 2 - HEIGHT_OF_HUMAN_FACE / 2)
    x_max = int(WIDTH_OF_PLAYGROUND / 2 + HEIGHT_OF_HUMAN_FACE / 2)
    y_min = int(HEIGHT_OF_PLAYGROUND / 2 - HEIGHT_OF_HUMAN_FACE / 2)
    y_max = int(HEIGHT_OF_PLAYGROUND / 2 + HEIGHT_OF_HUMAN_FACE / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

    return img

# function to check if the face is in the ideal square
def check_face_in_ideal_square(gaze):
    face = gaze["face"]
    x_min = int(WIDTH_OF_PLAYGROUND / 2 - HEIGHT_OF_HUMAN_FACE / 2)
    x_max = int(WIDTH_OF_PLAYGROUND / 2 + HEIGHT_OF_HUMAN_FACE / 2)
    y_min = int(HEIGHT_OF_PLAYGROUND / 2 - HEIGHT_OF_HUMAN_FACE / 2)
    y_max = int(HEIGHT_OF_PLAYGROUND / 2 + HEIGHT_OF_HUMAN_FACE / 2)

    if face["x"] - face["width"] / 2 > x_min and face["x"] + face["width"] / 2 < x_max and face["y"] - face["height"] / 2 > y_min and face["y"] + face["height"] / 2 < y_max:
        return True
    else:
        return False

# function to ask the use to align his face in the ideal square
def align_face_in_ideal_square():
    print("Please align your face in the green square in the middle of the playground for 5 seconds.")
    start_time = None
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gazes = detect_gazes(frame)
        if len(gazes) > 0:
            gaze = gazes[0]
            draw_face_square(frame, gaze)
            frame = draw_ideal_square(frame)
            cv2.imshow("gaze calib", frame)
            if check_face_in_ideal_square(gaze):
                if start_time is None:
                    start_time = time.time()
                elapsed_time = time.time() - start_time
                if elapsed_time >= 5:
                    break
            else:
                start_time = None

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

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

    gaze_point_x, gaze_point_y = transform_coordinates(gaze_point_x, gaze_point_y, transformation_matrix, image_width, image_height)

    # Initialize the moving average filter
    if not hasattr(draw_gaze_point, 'gaze_history'):
        draw_gaze_point.gaze_history = []
        draw_gaze_point.window_size = 5  # Adjust this value to control the smoothing effect

    # Add the current gaze point to the history
    draw_gaze_point.gaze_history.append((gaze_point_x, gaze_point_y))

    # Limit the history to the window size
    draw_gaze_point.gaze_history = draw_gaze_point.gaze_history[-draw_gaze_point.window_size:]

    # Calculate the moving average
    filtered_gaze_point_x = int(sum(x for x, _ in draw_gaze_point.gaze_history) / len(draw_gaze_point.gaze_history))
    filtered_gaze_point_y = int(sum(y for _, y in draw_gaze_point.gaze_history) / len(draw_gaze_point.gaze_history))

    print(f"Raw gaze point: ({gaze_point_x}, {gaze_point_y})")
    print(f"Filtered gaze point: ({filtered_gaze_point_x}, {filtered_gaze_point_y})")
    
    gaze_point = (filtered_gaze_point_x, filtered_gaze_point_y)

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

    top_left_x = []
    top_left_y = []
    for _ in range(4):
        x, y = calibrate_corner(0, 0, "top-left")
        top_left_x.append(x)
        top_left_y.append(y)
    top_left_x = int(sum(top_left_x) / len(top_left_x))
    top_left_y = int(sum(top_left_y) / len(top_left_y))

    top_right_x = []
    top_right_y = []
    for _ in range(4):
        x, y = calibrate_corner(image_width, 0, "top-right")
        top_right_x.append(x)
        top_right_y.append(y)
    top_right_x = int(sum(top_right_x) / len(top_right_x))
    top_right_y = int(sum(top_right_y) / len(top_right_y))

    bottom_left_x = []
    bottom_left_y = []
    for _ in range(4):
        x, y = calibrate_corner(0, image_height, "bottom-left")
        bottom_left_x.append(x)
        bottom_left_y.append(y)
    bottom_left_x = int(sum(bottom_left_x) / len(bottom_left_x))
    bottom_left_y = int(sum(bottom_left_y) / len(bottom_left_y))

    bottom_right_x = []
    bottom_right_y = []
    for _ in range(4):
        x, y = calibrate_corner(image_width, image_height, "bottom-right")
        bottom_right_x.append(x)
        bottom_right_y.append(y)
    bottom_right_x = int(sum(bottom_right_x) / len(bottom_right_x))
    bottom_right_y = int(sum(bottom_right_y) / len(bottom_right_y))
    
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

    # setup face distance
    align_face_in_ideal_square()

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

