import cv2
import numpy as np
from config import WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND
from coordinate_transform import *
from calibration import CalibrateGazeMapping, AlignFace
from gaze_detection import eyes_tracking_positions

def main():
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # Set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH_OF_PLAYGROUND)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT_OF_PLAYGROUND)

    # check face position
    aligner = AlignFace(cap)
    aligner.run()

    # Calibrate gaze mapping
    calibrator = CalibrateGazeMapping(cap)
    transformation_matrix = calibrator.perform_calibration()

    eyes_tracking_positions(cap, transformation_matrix)


    

if __name__ == "__main__":
    main()
