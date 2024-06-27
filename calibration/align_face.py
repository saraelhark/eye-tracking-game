"""This module is used to align the face in the frame a consistent way."""

import time
import config as cfg
from utils.gaze_detection import detect_gazes
from utils.visualization import draw_ideal_square, draw_face_square
from utils.video import video_loop


class AlignFace:
    """
    Class for aligning the face in the ideal square.

    Attributes:
        cap: The video capture object.
        start_time: The start time of the face alignment process.
        face_aligned: A boolean indicating whether the face is aligned or not.
    """

    def __init__(self, cap):
        self.cap = cap
        self.start_time = None
        self.face_aligned = False

    def check_face_in_ideal_square(self, gaze):
        """
        Check if the face is within the ideal square.

        Args:
            gaze: A dictionary containing gaze information.

        Returns:
            A boolean indicating whether the face is within the ideal square or not.
        """
        face = gaze["face"]
        x_min = int(cfg.WIDTH_OF_PLAYGROUND / 2 - cfg.HEIGHT_OF_HUMAN_FACE / 2)
        x_max = int(cfg.WIDTH_OF_PLAYGROUND / 2 + cfg.HEIGHT_OF_HUMAN_FACE / 2)
        y_min = int(cfg.HEIGHT_OF_PLAYGROUND / 2 - cfg.HEIGHT_OF_HUMAN_FACE / 2)
        y_max = int(cfg.HEIGHT_OF_PLAYGROUND / 2 + cfg.HEIGHT_OF_HUMAN_FACE / 2)

        return (x_min < face["x"] - face["width"] / 2 < x_max and
                x_min < face["x"] + face["width"] / 2 < x_max and
                y_min < face["y"] - face["height"] / 2 < y_max and
                y_min < face["y"] + face["height"] / 2 < y_max)

    def frame_processing_func(self, frame):
        """
        Process each frame of the video.

        Args:
            frame: The current frame of the video.

        Returns:
            A tuple containing the processed frame and the face alignment status.
        """
        gazes = detect_gazes(frame)
        if len(gazes) > 0:
            gaze = gazes[0]
            draw_face_square(frame, gaze)
            draw_ideal_square(frame)
            if self.check_face_in_ideal_square(gaze):
                if self.start_time is None:
                    self.start_time = time.time()
                elif time.time() - self.start_time >= cfg.FACE_ALIGNMENT_TIME:
                    self.face_aligned = True
                    return frame, self.face_aligned
            else:
                self.start_time = None
                self.face_aligned = False

        return frame, self.face_aligned

    def run(self):
        """
        Run the face alignment process.

        Prints a message and starts the video loop with the frame processing function.
        """
        text = "Please align your face in the green square for 5 seconds."
        video_loop(self.cap, self.frame_processing_func, display_name="Align Face in Ideal Square", extra_text=text)
