import cv2
import numpy as np

def video_loop(cap, frame_processing_func, display_name="Video Loop", destroy_windows=True):
    """
    A generic video loop function that can be used across different use cases.

    Args:
        cap (cv2.VideoCapture): The video capture object.
        frame_processing_func (callable): A function that takes a frame as input and returns the processed frame and a stop condition.
        display_name (str, optional): The name of the display window. Defaults to "Video Loop".
        destroy_windows (bool, optional): Whether to destroy the display windows at the end of the loop. Defaults to True.

    Returns:
        None
    """
    stop_condition = False
    while not stop_condition:
        ret, frame = cap.read()
        if not ret:
            break

        frame = flip_frame(frame)

        processed_frame, stop_condition = frame_processing_func(frame)

        display_frame(display_name, processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if destroy_windows:
        cv2.destroyAllWindows()


def display_frame(window_name, frame):

    cv2.imshow(window_name, frame)


def flip_frame(frame):
    """
    Flip the frame horizontally.
    """
    return cv2.flip(frame, 1)
