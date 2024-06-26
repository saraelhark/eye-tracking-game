""" This module containe some filters to smooth gaze point coordinates. """
import numpy as np
import cv2


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

PROCESS_NOISE = 1e-3
MEASUREMENT_NOISE = 0.3

class KalmanFilter:
    def __init__(self, initial_state, process_noise=PROCESS_NOISE, measurement_noise=MEASUREMENT_NOISE):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 0.5, 0],
                                             [0, 1, 0, 0.5],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

        # Initialize state
        self.kf.statePost = np.array([[initial_state[0]],
                                      [initial_state[1]],
                                      [0],
                                      [0]], dtype=np.float32)

    def update(self, measurement):
        measurement = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        predicted = self.kf.predict()
        updated = self.kf.correct(measurement)
        # print(measurement[:2].flatten())
        # print(predicted[:2].flatten())
        # print(updated[:2].flatten())
        return updated[:2].flatten()
    