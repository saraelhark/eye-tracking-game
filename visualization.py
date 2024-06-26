import cv2
import numpy as np
from config import *

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
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), FACE_SQUARE_COLOR, FACE_SQUARE_THICKNESS)
    return img

def draw_ideal_square(img):
    """
    Draw an ideal square in the middle of the image for face alignment.
    
    Args:
    img (numpy.ndarray): The image to draw on.
    
    Returns:
    numpy.ndarray: The image with the ideal square drawn.
    """
    x_min = int(WIDTH_OF_PLAYGROUND / 2 - HEIGHT_OF_HUMAN_FACE / 2)
    x_max = int(WIDTH_OF_PLAYGROUND / 2 + HEIGHT_OF_HUMAN_FACE / 2)
    y_min = int(HEIGHT_OF_PLAYGROUND / 2 - HEIGHT_OF_HUMAN_FACE / 2)
    y_max = int(HEIGHT_OF_PLAYGROUND / 2 + HEIGHT_OF_HUMAN_FACE / 2)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), IDEAL_SQUARE_COLOR, IDEAL_SQUARE_THICKNESS)
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
    cv2.circle(img, gaze_point, GAZE_POINT_RADIUS, GAZE_POINT_COLOR, -1)
    return img

def draw_calibration_point(img, point):
    """
    Draw a calibration point on the image.
    
    Args:
    img (numpy.ndarray): The image to draw on.
    point (tuple): The (x, y) coordinates of the calibration point.
    
    Returns:
    numpy.ndarray: The image with the calibration point drawn.
    """
    cv2.circle(img, point, CALIBRATION_POINT_RADIUS, CALIBRATION_POINT_COLOR, -1)
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
