import cv2
import numpy as np

def video_loop(cap, frame_processing_func, display_name="Video Loop", destroy_windows=True):

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = flip_frame(frame)

        processed_frame = frame_processing_func(frame)

        display_frame(display_name, processed_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if destroy_windows:
        cv2.destroyAllWindows()

def display_frame(window_name, frame):

    if isinstance(frame, np.ndarray):
        cv2.imshow(window_name, frame)
    elif isinstance(frame, tuple):
        # If the frame is a tuple, assume it's (frame, start_time)
        cv2.imshow(window_name, frame[0])
    else:
        raise ValueError("Invalid frame type. Expected numpy.ndarray or tuple.")


def flip_frame(frame):
    """
    Flip the frame horizontally.
    """
    return cv2.flip(frame, 1)
