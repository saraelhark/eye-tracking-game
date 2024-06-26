import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.environ.get("API_KEY")
GAZE_DETECTION_URL = f"http://127.0.0.1:9001/gaze/gaze_detection?api_key={API_KEY}"

# Physical measurements
DISTANCE_TO_OBJECT = 500  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm

# Playground size
WIDTH_OF_PLAYGROUND = 640
HEIGHT_OF_PLAYGROUND = 480

# Calibration settings
CALIBRATION_POINTS = 3  # Number of times to calibrate each corner
FACE_ALIGNMENT_TIME = 5  # Seconds to align face in ideal square

# Gaze point filtering
GAZE_HISTORY_WINDOW_SIZE = 5  # Number of points to use for moving average

# Colors (in BGR format for OpenCV)
FACE_SQUARE_COLOR = (255, 0, 0)  # Blue
IDEAL_SQUARE_COLOR = (0, 255, 0)  # Green
GAZE_POINT_COLOR = (0, 255, 0)  # Green
CALIBRATION_POINT_COLOR = (0, 0, 255)  # Red

# Drawing parameters
FACE_SQUARE_THICKNESS = 3
IDEAL_SQUARE_THICKNESS = 3
GAZE_POINT_RADIUS = 15
CALIBRATION_POINT_RADIUS = 15

POLYLINE_THICKNESS = 45

# Webcam settings
WEBCAM_INDEX = 0  # Use 0 for the default webcam
