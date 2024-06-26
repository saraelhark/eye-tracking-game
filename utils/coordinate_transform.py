""" This module contains math functions to calculate gaze point coordinates. """

import numpy as np
import cv2
from config import DISTANCE_TO_OBJECT, HEIGHT_OF_HUMAN_FACE

def calculate_gaze_point(gaze_x_raw, gaze_y_raw, image_width, image_height):
    """
    Calculate the gaze point coordinates on the image.

    Args:
    gaze_x_raw (float): Raw x-coordinate of the gaze point.
    gaze_y_raw (float): Raw y-coordinate of the gaze point.
    image_width (int): Width of the image.
    image_height (int): Height of the image.

    Returns:
    tuple: (gaze_point_x, gaze_point_y) coordinates on the image.
    """
    gaze_point_x = image_width / 2 + gaze_x_raw
    gaze_point_y = image_height / 2 + gaze_y_raw
    return gaze_point_x, gaze_point_y

def calculate_gaze_point_displacements(gaze):
    """
    Calculate the gaze point displacements based on yaw and pitch.

    Args:
    gaze (dict): Gaze data containing yaw, pitch, and face information.

    Returns:
    tuple: (dx, dy) displacements of the gaze point.
    """
    length_per_pixel = HEIGHT_OF_HUMAN_FACE / gaze["face"]["height"]

    dx = -DISTANCE_TO_OBJECT * np.tan(gaze['yaw']) / length_per_pixel
    dx = dx if not np.isnan(dx) else 100000000

    yaw_cos = np.clip(gaze['yaw'], -1, 1)
    dy = -DISTANCE_TO_OBJECT * np.arccos(yaw_cos) * np.tan(gaze['pitch']) / length_per_pixel
    dy = dy if not np.isnan(dy) else 100000000

    return dx, dy

def transform_coordinates(gaze_x_raw, gaze_y_raw, transformation_matrix, image_width, image_height):
    """
    Transform raw gaze coordinates using the calibration transformation matrix.

    Args:
    gaze_x_raw (float): Raw x-coordinate of the gaze point.
    gaze_y_raw (float): Raw y-coordinate of the gaze point.
    transformation_matrix (numpy.ndarray): 3x3 transformation matrix from calibration.
    image_width (int): Width of the image.
    image_height (int): Height of the image.

    Returns:
    tuple: (adjusted_x, adjusted_y) transformed coordinates.
    """
    # Reshape the input array to the expected shape (1, 1, 2)
    input_array = np.array([[gaze_x_raw, gaze_y_raw]], dtype=np.float32).reshape(1, 1, 2)

    # Apply the transformation matrix to the raw gaze point coordinates
    adjusted_x, adjusted_y = cv2.perspectiveTransform(input_array, transformation_matrix)[0][0]

    # Ensure the adjusted coordinates are within the frame boundaries
    adjusted_x = max(0, min(adjusted_x, image_width - 1))
    adjusted_y = max(0, min(adjusted_y, image_height - 1))

    return int(adjusted_x), int(adjusted_y)
