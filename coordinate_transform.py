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

def apply_moving_average_filter(gaze_history, new_point, window_size):
    """
    Apply a moving average filter to smooth gaze point coordinates.

    Args:
    gaze_history (list): List of previous gaze points.
    new_point (tuple): New gaze point (x, y) to be added.
    window_size (int): Size of the moving average window.

    Returns:
    tuple: (filtered_x, filtered_y) smoothed gaze point coordinates.
    """
    gaze_history.append(new_point)
    if len(gaze_history) > window_size:
        gaze_history.pop(0)

    filtered_x = int(sum(x for x, _ in gaze_history) / len(gaze_history))
    filtered_y = int(sum(y for _, y in gaze_history) / len(gaze_history))

    return filtered_x, filtered_y

def apply_median_filter(gaze_history, new_point, window_size):
    """
    Apply a median filter to smooth gaze point coordinates.

    Args:
    gaze_history (list): List of previous gaze points.
    new_point (tuple): New gaze point (x, y) to be added.
    window_size (int): Size of the median filter window.

    Returns:
    tuple: (filtered_x, filtered_y) smoothed gaze point coordinates.
    """
    gaze_history.append(new_point)
    if len(gaze_history) > window_size:
        gaze_history.pop(0)

    x_values = [x for x, _ in gaze_history]
    y_values = [y for _, y in gaze_history]

    filtered_x = int(np.median(x_values))
    filtered_y = int(np.median(y_values))

    return filtered_x, filtered_y

def adaptive_weighted_moving_average(gaze_history, new_point, max_window_size=10):
    gaze_history.append(new_point)
    if len(gaze_history) > max_window_size:
        gaze_history.pop(0)
    
    # Calculate the speed of movement
    if len(gaze_history) > 1:
        speed = np.linalg.norm(np.array(gaze_history[-1]) - np.array(gaze_history[-2]))
    else:
        speed = 0
    
    # Adjust window size based on speed
    adaptive_window_size = max(2, int(max_window_size * (1 - speed / 100)))
    
    # Calculate weights (more recent points have higher weights)
    weights = np.linspace(0.5, 1.0, len(gaze_history))
    weights /= weights.sum()
    
    # Apply weighted average
    x_values = [x for x, _ in gaze_history]
    y_values = [y for _, y in gaze_history]
    
    filtered_x = int(np.average(x_values, weights=weights))
    filtered_y = int(np.average(y_values, weights=weights))
    
    return filtered_x, filtered_y
