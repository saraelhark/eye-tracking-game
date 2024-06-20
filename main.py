import cv2
import numpy as np
from config import WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND, GAZE_HISTORY_WINDOW_SIZE
from gaze_detection import process_frame
from coordinate_transform import calculate_gaze_point_displacements, calculate_gaze_point, transform_coordinates, apply_moving_average_filter, apply_median_filter, adaptive_weighted_moving_average
from visualization import draw_face_square, draw_gaze_point, display_frame, flip_frame
from calibration import perform_calibration

def main():
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise Exception("Could not open video device")

    # Set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH_OF_PLAYGROUND)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT_OF_PLAYGROUND)

    # Calibrate gaze mapping
    transformation_matrix = perform_calibration(cap)

    # Initialize gaze history for smoothing
    gaze_history = []
    window_size = GAZE_HISTORY_WINDOW_SIZE

    while True:
        _, frame = cap.read()
        frame = flip_frame(frame)

        gaze_data_list = process_frame(frame)

        if not gaze_data_list:
            continue

        gaze = gaze_data_list[0]  # Assuming we're only interested in the first detected gaze

        # Draw face square
        frame = draw_face_square(frame, gaze)

        # Calculate gaze point
        dx, dy = calculate_gaze_point_displacements(gaze)
        gaze_x, gaze_y = calculate_gaze_point(dx, dy, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)

        # Transform coordinates
        gaze_x, gaze_y = transform_coordinates(gaze_x, gaze_y, transformation_matrix, WIDTH_OF_PLAYGROUND, HEIGHT_OF_PLAYGROUND)

        # Apply moving average filter
        #filtered_x, filtered_y = apply_moving_average_filter(gaze_history, (gaze_x, gaze_y), window_size)

        #filtered_x, filtered_y = apply_median_filter(gaze_history, (gaze_x, gaze_y), window_size)

        #filtered_x, filtered_y = adaptive_weighted_moving_average(gaze_history, (gaze_x, gaze_y), window_size)
    

        print(f"Raw gaze point: ({gaze_x}, {gaze_y})")
        print(f"Filtered gaze point: ({filtered_x}, {filtered_y})")

        # Draw gaze point
        frame = draw_gaze_point(frame, (filtered_x, filtered_y))

        # Display the frame
        display_frame("Gaze Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
