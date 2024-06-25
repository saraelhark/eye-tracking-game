import cv2
import numpy as np
from config import WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND
from coordinate_transform import *
from calibration import CalibrateGazeMapping, AlignFace, CheckGazeAccuracy
from gaze_detection import EyesTrackingPositions

def main():
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # Set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH_OF_PLAYGROUND)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT_OF_PLAYGROUND)

    # step 1: check and align face position 
    aligner = AlignFace(cap)
    aligner.run()

    # step 2: calibrate gaze mapping with points on screen
    calibrator = CalibrateGazeMapping(cap)
    transformation_matrix = calibrator.perform_calibration()

    # step 3: check calibration accuracy
    accuracy_checker = CheckGazeAccuracy(cap, transformation_matrix)
    accuracy_checker.run()

    # step 4: detect and track eyes with filtering
    eyes_tracker = EyesTrackingPositions(cap, transformation_matrix)
    eyes_tracker.run()

    

if __name__ == "__main__":
    main()
