""" This is a module for visualizing gaze detection results. """

import cv2
import config as cfg


def draw_face_square(img, gaze):
    """
    Draw a square around the detected face.
    
    Args:
    img (numpy.ndarray): The image to draw on.
    gaze (dict): The gaze data containing face information.
    
    Returns:
    numpy.ndarray: The image with the face square drawn.
    """
    face = gaze["face"]
    x_min = int(face["x"] - face["width"] / 2)
    x_max = int(face["x"] + face["width"] / 2)
    y_min = int(face["y"] - face["height"] / 2)
    y_max = int(face["y"] + face["height"] / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), cfg.FACE_SQUARE_COLOR, cfg.FACE_SQUARE_THICKNESS)
    return img


def draw_ideal_square(img):
    """
    Draw an ideal square in the middle of the image for face alignment.
    
    Args:
    img (numpy.ndarray): The image to draw on.
    
    Returns:
    numpy.ndarray: The image with the ideal square drawn.
    """
    image_height, image_width = img.shape[:2]

    x_min = int(image_width / 2 - cfg.HEIGHT_OF_HUMAN_FACE / 2)
    x_max = int(image_width / 2 + cfg.HEIGHT_OF_HUMAN_FACE / 2)
    y_min = int(image_height / 2 - cfg.HEIGHT_OF_HUMAN_FACE / 2)
    y_max = int(image_height / 2 + cfg.HEIGHT_OF_HUMAN_FACE / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), cfg.IDEAL_SQUARE_COLOR, cfg.IDEAL_SQUARE_THICKNESS)
    return img


def draw_gaze_point(img, gaze_point):
    """
    Draw the gaze point on the image.
    
    Args:
    img (numpy.ndarray): The image to draw on.
    gaze_point (tuple): The (x, y) coordinates of the gaze point.
    
    Returns:
    numpy.ndarray: The image with the gaze point drawn.
    """
    gaze_point_sat = gaze_point_saturation(img, gaze_point)
    cv2.circle(img, gaze_point_sat, cfg.GAZE_POINT_RADIUS, cfg.GAZE_POINT_COLOR, -1)
    return img


def gaze_point_saturation(img, gaze_point):
    """
    Draw the gaze point on the image.
    
    Args:
    img (numpy.ndarray): The image to draw on.
    gaze_point (tuple): The (x, y) coordinates of the gaze point.
    
    Returns:
    numpy.ndarray: The image with the gaze point drawn.
    """
    x, y = gaze_point
    x = max(0, min(x, img.shape[1] - 1))
    y = max(0, min(y, img.shape[0] - 1))
    gaze_point = (x, y)
    return gaze_point


def draw_calibration_point(img, point):
    """
    Draw a calibration point on the image.
    
    Args:
    img (numpy.ndarray): The image to draw on.
    point (tuple): The (x, y) coordinates of the calibration point.
    
    Returns:
    numpy.ndarray: The image with the calibration point drawn.
    """
    cv2.circle(img, point, cfg.CALIBRATION_POINT_RADIUS, cfg.CALIBRATION_POINT_COLOR, -1)
    return img


def add_text_overlay(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = 20
    text_color = (255, 0, 0)

    # Split the text into two rows if it's too long
    max_text_length = 40
    if len(text) > max_text_length:
        split_index = text.rfind(' ', 0, max_text_length)
        if split_index == -1:
            split_index = max_text_length
        row1 = text[:split_index]
        row2 = text[split_index+1:]

        # Draw the first row
        row1_size, _ = cv2.getTextSize(row1, font, font_scale, thickness)
        row1_x = (img.shape[1] - row1_size[0]) // 2
        row1_y = text_y
        cv2.putText(img, row1, (row1_x, row1_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        # Draw the second row
        row2_size, _ = cv2.getTextSize(row2, font, font_scale, thickness)
        row2_x = (img.shape[1] - row2_size[0]) // 2
        row2_y = row1_y + row1_size[1] + 10
        cv2.putText(img, row2, (row2_x, row2_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    else:
        cv2.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def show_timer(img, timer):
    image_height, image_width = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_x = image_width - 160
    text_y = image_height - 20
    text_color = (0, 0, 255)

    cv2.putText(img, timer, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_target(frame, target_position):
    """
    Draw a target on the frame at the given position.

    Args:
    frame (numpy.ndarray): The input frame to draw the target on.
    target_position (tuple): The (x, y) coordinates of the target position.

    Returns:
    numpy.ndarray: The frame with the target drawn on it.
    """
    x, y = target_position
    radius = 40
    color = (0, 0, 255)
    thickness = 2

    # Draw the target circle
    frame = cv2.circle(frame, (x, y), radius, color, thickness)

    return frame
