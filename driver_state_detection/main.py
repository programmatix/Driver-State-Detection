# Must run under an admin prompt with
# start "" /high python main.py
# ($Process = Start-Process "python" -ArgumentList "main.py" -PassThru).PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High

import time

import argparse
import cv2
import numpy as np
import os
import threading
# from influxdb_client_3 import InfluxDBClient3
from tensorflow.python.client import device_lib

from CameraCapture import capture_frames
from GlobalContext import GlobalContext
from ProcessFrames import process_frames
from SaveFrames import save_frames



#from hdrhistogram import HdrHistogram

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())

parser = argparse.ArgumentParser(description='Driver State Detection')

# selection the camera number, default is 0 (webcam)
parser.add_argument('-c', '--camera', type=int,
                    default=0, metavar='', help='Camera number, default is 0 (webcam)')
parser.add_argument('-f', '--flip', type=int)
parser.add_argument('--model', type=str)
parser.add_argument('-m', '--mode', type=int, default=0)


# Attention Scorer parameters (EAR, Gaze Score, Pose)
parser.add_argument('--smooth_factor', type=float, default=0.5,
                    metavar='', help='Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5')
# When average EAR was being used for blinks (pre Pixel):
# 0.20 avg too sensitive
# 0.10 avg not picking up
# 0.05 avg not picking up
# Elgato https://dynalist.io/d/W6zPj7VmtrR-Jmv2n1WHT-Pg#z=BuGt5_-qrFDYLm3b4GCQ-8xT
# 0.08 way too low, getting good closed-eye draws at 0.106
# Trying 0.2
parser.add_argument('--ear_thresh', type=float, default=0.23,
                    metavar='', help='Sets the EAR threshold for the Attention Scorer, default is 0.15')
parser.add_argument('--ear_time_thresh', type=float, default=2,
                    metavar='', help='Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds')
parser.add_argument('--gaze_thresh', type=float, default=0.015,
                    metavar='', help='Sets the Gaze Score threshold for the Attention Scorer, default is 0.2')
parser.add_argument('--gaze_time_thresh', type=float, default=2, metavar='',
                    help='Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2. seconds')
parser.add_argument('--pitch_thresh', type=float, default=20,
                    metavar='', help='Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees')
parser.add_argument('--yaw_thresh', type=float, default=20,
                    metavar='', help='Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees')
parser.add_argument('--roll_thresh', type=float, default=20,
                    metavar='', help='Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees')
parser.add_argument('--pose_time_thresh', type=float, default=2.5,
                    metavar='', help='Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds')

parser.add_argument('--input', type=str, default="")
parser.add_argument('-w', '--write_to_influx', type=bool, default=False, metavar='', )

# parse the arguments and store them in the args variable dictionary
args = parser.parse_args()

#model = load_model(args.model)


# Load environment variables from .env file

# Initialize InfluxDB client
# client = InfluxDBClient3(
#     host=INFLUXDB_URL,
#     token=INFLUXDB_TOKEN,
#     database=INFLUXDB_BUCKET
# )

# camera matrix obtained from the camera calibration script, using a 9x6 chessboard
camera_matrix = np.array(
    [[899.12150372, 0., 644.26261492],
     [0., 899.45280671, 372.28009436],
     [0, 0,  1]], dtype="double")

# distortion coefficients obtained from the camera calibration script, using a 9x6 chessboard
dist_coeffs = np.array(
    [[-0.03792548, 0.09233237, 0.00419088, 0.00317323, -0.15804257]], dtype="double")

def _get_landmarks(lms):
    surface = 0
    for lms0 in lms:
        landmarks = [np.array([point.x, point.y, point.z]) \
                     for point in lms0.landmark]

        landmarks = np.array(landmarks)

        landmarks[landmarks[:, 0] < 0., 0] = 0.
        landmarks[landmarks[:, 0] > 1., 0] = 1.
        landmarks[landmarks[:, 1] < 0., 1] = 0.
        landmarks[landmarks[:, 1] > 1., 1] = 1.

        dx = landmarks[:, 0].max() - landmarks[:, 0].min()
        dy = landmarks[:, 1].max() - landmarks[:, 1].min()
        new_surface = dx * dy
        if new_surface > surface:
            biggest_face = landmarks

    return biggest_face

def draw_ear_between_eyes(frame, landmarks, ear, ear_left, ear_right):
    global flip_eye_mode
    """
    Draws the EAR score between the eyes on the frame.

    Parameters:
    - frame: The video frame to draw on.
    - landmarks: The facial landmarks detected in the frame.
    - ear: The Eye Aspect Ratio (EAR) score to display.
    """
    # Define indices for the outer corners of the eyes in the landmarks array
    LEFT_EYE_OUTER_CORNER_IDX = 468  # Adjust based on your landmarks model
    RIGHT_EYE_OUTER_CORNER_IDX = 473  # Adjust based on your landmarks model

    # Extract the outer corner points of each eye
    left_eye_point = landmarks[LEFT_EYE_OUTER_CORNER_IDX][:2]
    right_eye_point = landmarks[RIGHT_EYE_OUTER_CORNER_IDX][:2]

    # Calculate the midpoint between the eyes
    midpoint_x = ((left_eye_point[0] + right_eye_point[0]) / 2)
    midpoint_y = ((left_eye_point[1] + right_eye_point[1]) / 2)

    # Convert midpoint coordinates to frame scale
    frame_midpoint_x = int(midpoint_x * frame.shape[1])
    frame_midpoint_y = int(midpoint_y * frame.shape[0])

    frame_left_x = int(left_eye_point[0] * frame.shape[1])
    frame_left_y = int(left_eye_point[1] * frame.shape[0]) + 20

    frame_right_x = int(right_eye_point[0] * frame.shape[1])
    frame_right_y = int(right_eye_point[1] * frame.shape[0]) + 20

    #print(f"Midpoint: ({frame_midpoint_x}, {frame_midpoint_y}) left: {left_eye_point} right: {right_eye_point} midpoint: {midpoint_x}, {midpoint_y}")

    # Display the EAR score at the calculated midpoint
    # cv2.putText(frame, f"EAR: {round(ear, 3)}", (frame_midpoint_x, frame_midpoint_y),
    #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    draw_on_screen_left_eye = 0
    draw_on_screen_right_eye = 0
    if flip_eye_mode:
        draw_on_screen_left_eye = ear_right
        draw_on_screen_right_eye = ear_left
    else:
        draw_on_screen_left_eye = ear_left
        draw_on_screen_right_eye = ear_right

    cv2.putText(frame, f"{round(draw_on_screen_left_eye, 3)}", (frame_left_x, frame_left_y),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    cv2.putText(frame, f"{round(draw_on_screen_right_eye, 3)}", (frame_right_x, frame_right_y),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)







gc = GlobalContext(args)

# Create and start the threads
thread_capture = threading.Thread(target=capture_frames, args=[gc], daemon=True)
thread_save = threading.Thread(target=save_frames, args=[gc], daemon=True)
thread_process = threading.Thread(target=process_frames, args=[gc], daemon=True)

if args.input == "":
    thread_capture.start()

thread_save.start()
thread_process.start()

# Keep the main thread alive
try:
    while gc.done is False:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
print("done1")
gc.done = True
thread_capture.join()
print("thread_capture ended")
thread_save.join()
print("thread_save ended")
thread_process.join()
print("thread_process ended")

os._exit(-1)
print("done2")