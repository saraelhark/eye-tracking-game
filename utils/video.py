""" This module contains utility functions for working with video streams. """

import time
import cv2
import logging
from utils.visualization import add_text_overlay

logging.basicConfig(level=logging.DEBUG)

def video_loop(cap, frame_processing_func, display_name="Video Loop", extra_text="", destroy_windows=True):
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
    fps_start_time = time.time()
    fps = 0
    frame_count = 0
    stop_condition = False
    while not stop_condition:
        ret, frame = cap.read()
        if not ret:
            break

        frame = flip_frame(frame)

        processed_frame, stop_condition = frame_processing_func(frame)

        display_frame(display_name, processed_frame, extra_text)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1
        if time.time() - fps_start_time >= 1:
            fps = frame_count / (time.time() - fps_start_time)
            frame_count = 0
            fps_start_time = time.time()
            logging.debug(f"FPS: {fps:.2f}")

    if destroy_windows:
        cv2.destroyAllWindows()


def display_frame(window_name, frame, text=""):
    """
    Show the frame with the text overlay (if any).
    """
    add_text_overlay(frame, text)
    cv2.imshow(window_name, frame)


def flip_frame(frame):
    """
    Flip the frame horizontally.
    """
    return cv2.flip(frame, 1)
