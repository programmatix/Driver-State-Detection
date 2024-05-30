# Must run under an admin prompt with
# start "" /high python main.py
# ($Process = Start-Process "python" -ArgumentList "main.py" -PassThru).PriorityClass = [System.Diagnostics.ProcessPriorityClass]::High


import argparse
import cv2
import math
import mediapipe as mp
import numpy as np
import os
import queue
import socket
import threading
import time

from keras.src.saving import load_model

from Attention_Scorer_Module import AttentionScorer as AttScorer
from BlinkDetector import BlinkDetector
from Eye_Dector_Module import EyeDetector as EyeDet
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from RealTimeEARPlot import RealTimeEARPlot
from RealTimePERCLOSPlot import RealTimePERCLOSPlot
from datetime import datetime
from dotenv import load_dotenv
from EyeImage import clean_eye
from influxdb_client_3 import InfluxDBClient3
import tensorflow as tf
from tensorflow.python.client import device_lib

import TrainingConstants
from ModelPredict import predict, predict_multi
from TrainingProcess import process_image

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

done = False

hostname = socket.gethostname()

# Load environment variables from .env file
load_dotenv()

INFLUXDB_URL = os.getenv('INFLUXDB_URL')
INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN')
INFLUXDB_ORG = os.getenv('INFLUXDB_ORG')
INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET')

# Initialize InfluxDB client
client = InfluxDBClient3(
    host=INFLUXDB_URL,
    token=INFLUXDB_TOKEN,
    database=INFLUXDB_BUCKET
)

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

def zoom_in(frame, zoom_factor=2):
    """
    Zooms in on the frame by the specified zoom factor.
    A zoom_factor of 2 means the image will be zoomed in by 100%, focusing on the center.

    Parameters:
    - frame: The original webcaframe.
    - zoom_factor: The factor by which to zoom in on the frame.

    Returns:
    - The zoomed-in frame.
    """
    height, width = frame.shape[:2]
    new_width, new_height = width // zoom_factor, height // zoom_factor

    # Calculate the region of interest
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2

    # Extract the zoomed-in region
    zoomed_in_region = frame[top:bottom, left:right]

    # Resize back to original frame size
    zoomed_in_frame = cv2.resize(zoomed_in_region, (width, height), interpolation=cv2.INTER_LINEAR)

    return zoomed_in_frame

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

# Create a thread-safe queue
frame_queue_for_saving = queue.Queue()
frame_queue_for_processing = queue.Queue()

print_timings = False
capture_fps = 0
save_fps = 0
process_fps = 0
capture_mode = args.input != ""
buffer_mode = False
dump_buffered_frames = False
flip_mode = 3
# Bit confused but I think OpenCV is working with screen space, e.g. left eye is on the left of the screen.
# So this flips it so it's in physical space - left eye is physically on my left, e.g. right of the screen.
flip_eye_mode = True
saving_to_influx = args.write_to_influx
print("Saving to InfluxDB: " + str(saving_to_influx) + " " + str(args.write_to_influx))

mode = args.mode


def open_camera():
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840) # 4k/high_res
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) # 4k/high_res

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # 4k/high_res
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 4k/high_res
    cap.set(cv2.CAP_PROP_FPS, 120) # 4k/high_res

    print(f"Camera supports CAP_PROP_ZOOM: {cap.get(cv2.CAP_PROP_ZOOM)}")
    print(f"Camera supports CAP_PROP_FOURCC: {cap.get(cv2.CAP_PROP_FOURCC)}")

    # Request compression
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # fourcc = cv2.VideoWriter_fourcc(*'X265')
    # print(f"Setting cap CAP_PROP_FOURCC to X265: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # print(f"Setting cap CAP_PROP_FOURCC to X264: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # print(f"Setting cap CAP_PROP_FOURCC to XVID: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print(f"Setting cap CAP_PROP_FOURCC to MJPG: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # print(f"Setting cap CAP_PROP_FOURCC to MP4V: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    #
    # fourcc = cv2.VideoWriter_fourcc(*'VP80')
    # print(f"Setting cap CAP_PROP_FOURCC to VP80: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'VP90')
    # print(f"Setting cap CAP_PROP_FOURCC to VP90: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    # print(f"Setting cap CAP_PROP_FOURCC to WMV1: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'WMV2')
    # print(f"Setting cap CAP_PROP_FOURCC to WMV2: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'AVC1')
    # print(f"Setting cap CAP_PROP_FOURCC to AVC1: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))

    #print(f"Setting cap CAP_MODE_GRAY: ", cap.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY))

    print(f"Setting cap VIDEO_ACCELERATION_ANY: ", cap.set(cv2.VIDEO_ACCELERATION_ANY, 1))
    #cap.set(cv2.VIDEO_ACCELERATION_ANY, 1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # 4k/high_res
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # 4k/high_res
    # cap.set(cv2.CAP_PROP_FPS, 60) # 4k/high_res
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"The default resolution of the webcam is {width}x{height} {int(fps)}FPS")

    return cap

def compress_frame(frame, quality=95):
    # Encode the frame into a memory buffer
    ret, jpeg_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    # Decode the memory buffer back into an image
    # compressed_frame = cv2.imdecode(jpeg_frame, 1)

    return jpeg_frame

def capture_frames():
    global capture_fps
    global done
    # p = psutil.Process(os.getpid())
    # # Set the process priority to above normal, this can be adjusted to your needs
    # p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    # # Additionally, set the thread priority if needed
    # threading.current_thread().priority = threading.PRIORITY_HIGHEST

    frame_idx = 0
    prev_second = None

    cap = open_camera()

    #histogram = HdrHistogram()
    while done is False:
        #tX = time.perf_counter()
        ret, frame = cap.read()
        #if (print_timings):
            #histogram.record_value((time.perf_counter() - tX) * 1000)
            #print(f"Time to read frame: {(time.perf_counter() - tX) * 1000} FPS: {capture_fps}")

        current_time = datetime.now()
        current_second = current_time.strftime("%S")

        if prev_second is None or prev_second != current_second:
            prev_second = current_second
            capture_fps = frame_idx
            frame_idx = 0  # Reset frame index for each new second
        else:
            frame_idx += 1

        # cv2.imshow(f"window", frame)
        # cv2.waitKey(1)

        if not ret:
            print("Can't receive frame from camera/stream end")
            time.sleep(1)
            cap = open_camera()
        else:
            frame_queue_for_processing.put(frame)

        #Every second display histogram
    print("Capture thread done")
    cap.release()

buffered_frames = []


def save_frames():
    global save_fps
    global capture_mode
    global buffer_mode
    global done
    global dump_buffered_frames
    global buffered_frames

    frame_idx = 0
    prev_second = None

    while done is False:
        try:
            frames = frame_queue_for_saving.get(timeout=1)
        except queue.Empty:
            continue

        #print(f"Got frames orig={int(frames[0].nbytes / 1024)}kb processed={int(frames[1].nbytes / 1024)}kb")

        current_time = datetime.now()
        current_second = current_time.strftime("%S")

        if prev_second is None or prev_second != current_second:
            prev_second = current_second
            save_fps = frame_idx
            frame_idx = 0  # Reset frame index for each new second
        else:
            frame_idx += 1

        now = time.time()
        # buffered_frames.append((frames[0], now, frame_idx, "orig"))

        # # We have to compressed the stored image, it's just way too much memory otherwise
        compress_frames = False
        # compressed_proc = compress_frame(frames[1], 95)
        # buffered_frames.append((compressed_proc, now, frame_idx, "proc", current_time))
        if buffer_mode:
            buffered_frames.append((frames[0], now, frame_idx, "orig", current_time))
        # five_minutes_ago = now - 10 * 60
        # buffered_frames = [(frame, timestamp, idx, desc, timestamp2) for frame, timestamp, idx, desc, timestamp2 in buffered_frames if timestamp >= five_minutes_ago]


        if dump_buffered_frames:
            dump_buffered_frames = False

            if compress_frames:
                folderName = "output_images"
            else:
                folderName = f"training/{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
            try:
                os.mkdir(folderName)
            except FileExistsError:
                pass

            # Save all frames in the buffer that are within the last 5 minutes
            for bframe, timestamp, idx, desc, timestamp2 in buffered_frames:

                fn = timestamp2.strftime("%Y-%m-%d_%H-%M-%S") + "-" + desc + "-" + str(idx)

                filename1 = f"{folderName}/{fn}.jpg"

                print(f"Writing {filename1}")
                if compress_frames:
                    with open(filename1, "wb") as f:
                        f.write(bframe.tobytes())
                else:
                    cv2.imwrite(filename1, bframe)

            buffered_frames = []

        # if (frame_idx == 0):
        #     print("New second " + timestamp)

        if capture_mode:
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S") + "-" + str(frame_idx)
            filename1 = f"output_images/{timestamp}-orig.jpg"
            filename2 = f"output_images/{timestamp}-processed.jpg"

            # Don't compress - hard enough to debug
            #cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(filename1, frames[0])
            cv2.imwrite(filename2, frames[1])

            print(f"Saved {filename1} and {filename2}")


def process_frames():
    global process_fps
    global capture_mode
    global flip_mode
    global flip_eye_mode
    global done
    global buffered_frames
    global dump_buffered_frames
    global buffer_mode
    global print_timings
    global saving_to_influx
    global model
    global mode
    try:

        if args.input:
            frame_queue_for_processing.put(cv2.imread(args.input))

        # if not cv2.useOptimized():
        #     try:
        #         cv2.setUseOptimized(True)  # set OpenCV optimization to True
        #     except:
        #         print(
        #             "OpenCV optimization could not be set to True, the script may be slower than expected")

        """instantiation of mediapipe face mesh model. This model give back 478 landmarks
        if the rifine_landmarks parameter is set to True. 468 landmarks for the face and
        the last 10 landmarks for the irises
        """
        detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                                   min_detection_confidence=0.5,
                                                   min_tracking_confidence=0.5,
                                                   # https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#attention-mesh-model
                                                   refine_landmarks=True)

        # instantiation of the eye detector and pose estimator objects
        Eye_det = EyeDet(show_processing=False)

        #Head_pose = HeadPoseEst(show_axis=args.show_axis)

        # instantiation of the attention scorer object, with the various thresholds
        # NOTE: set verbose to True for additional printed information about the scores
        t0 = time.perf_counter()
        Scorer = AttScorer(t_now=t0, ear_thresh=args.ear_thresh, gaze_time_thresh=args.gaze_time_thresh,
                           roll_thresh=args.roll_thresh, pitch_thresh=args.pitch_thresh,
                           yaw_thresh=args.yaw_thresh, ear_time_thresh=args.ear_time_thresh,
                           gaze_thresh=args.gaze_thresh, pose_time_thresh=args.pose_time_thresh,
                           verbose=False)

        i = 0
        time.sleep(0.01) # To prevent zero division error when calculating the FPS
        t_last_save = time.perf_counter()
        t_last_image_save = t_last_save

        save_to_influx_every_x_seconds = 5
        saving_to_influx = False

        ear_values = []
        ear_left_values = []
        ear_right_values = []
        gaze_values = []
        present_values = []

        blink_detector = BlinkDetector(ear_threshold=args.ear_thresh)  # Set your EAR threshold

        ear_plotter = RealTimeEARPlot()
        perclos_plotter = RealTimePERCLOSPlot()

        frame_idx = 0
        prev_second = None
        rolling_buffers = [[]]
        rolling_buffer = []

        while done is False:
            t_now = time.perf_counter()
            period_start_time = time.perf_counter()
            current_time = datetime.now()
            current_second = current_time.strftime("%S")
            text_list = []

            if prev_second is None or prev_second != current_second:
                prev_second = current_second
                process_fps = frame_idx
                frame_idx = 0  # Reset frame index for each new second
            else:
                frame_idx += 1

            fps = i / (t_now - t_last_save)
            if fps == 0:
                fps = 10

            try:
                frame = frame_queue_for_processing.get(timeout=1)
            except queue.Empty:
                continue


            #print(f"Got frame for processing {frame.nbytes / 1024}kb")

            # if the frame comes from webcam, flip it so it looks like a mirror.
            tX = time.perf_counter()
            # if args.camera == 0:
            #     frame = cv2.flip(frame, 2)
            #elif args.camera == 1:

            #print(f"Flip is {args.flip}")
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if flip_mode == 1:
                #print(f"Flipping")

                # This is 100% one of the modes I want, in this order!
                # The Pixel has its own algo for if/how it flips the output also.
                frame = cv2.flip(frame, 2)
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif flip_mode == 2:
                #print(f"Flipping")

                # This is 100% one of the modes I want, in this order!
                # The Pixel has its own algo for if/how it flips the output also.
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                frame = cv2.flip(frame, 1)
            elif flip_mode == 3:
                frame = cv2.flip(frame, 1)
            # frame = cv2.flip(frame, 2)
            # frame = zoom_in(frame, 2)
            if (print_timings):
                print(f"Time to flip frame: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            # start the tick counter for computing the processing time for each frame
            # e1 = cv2.getTickCount()

            # height, width = frame.shape[:2]  # Get the height and width of the image
            # tiny = cv2.resize(frame, (width//4, height//4))

            processed = frame.copy()


            # tX = time.perf_counter()
            # processed = compress_frame(frame, 80)
            # if (print_timings):
            #     print(f"Time to compress frame: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            # Do a quick pass to find the face and zoom in on the left eye
            # Doesn't work currently
            #
            # lms = detector.process(processed).multi_face_landmarks
            #
            # if processed is None or len(processed.shape) != 3:
            #     print("bad processed 426")
            #     exit(-1)
            #
            # if lms:
            #     landmarks = _get_landmarks(lms)
            #
            #     # Indices for the face landmarks in the 468-point model
            #     FACE_INDICES = list(range(0, 468))
            #
            #     # Get the face landmarks
            #     face_landmarks = landmarks[FACE_INDICES]
            #
            #     # Calculate the bounding box of the face
            #     face_bbox = (min(face_landmarks[:, 0]),  # x_min
            #                  min(face_landmarks[:, 1]),  # y_min
            #                  max(face_landmarks[:, 0]),  # x_max
            #                  max(face_landmarks[:, 1]))  # y_max
            #
            #     # Convert the bounding box coordinates to the frame scale
            #     face_bbox = (int(face_bbox[0] * processed.shape[1]),  # x_min
            #                  int(face_bbox[1] * processed.shape[0]),  # y_min
            #                  int(face_bbox[2] * processed.shape[1]),  # x_max
            #                  int(face_bbox[3] * processed.shape[0]))  # y_max
            #
            #     # Extract the ROI from the processed frame
            #     face_roi = processed[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
            #
            #     # Resize the ROI to the original frame size
            #     # face_zoomed_in = cv2.resize(face_roi, (tiny.shape[1], tiny.shape[0]), interpolation=cv2.INTER_LINEAR)
            #
            #     processed = face_roi
            #     if processed is None or len(processed.shape) != 3:
            #         print("bad processed 459")
            #         exit(-1)
            #
            #     processed = frame

                # Indices for the left eye landmarks in the 468-point model
                # LEFT_EYE_INDICES = list(range(33, 47))
                #
                # # Get the left eye landmarks
                # left_eye_landmarks = landmarks[LEFT_EYE_INDICES]
                #
                # # Calculate the bounding box of the left eye
                # left_eye_bbox = (min(left_eye_landmarks[:, 0]),  # x_min
                #                  min(left_eye_landmarks[:, 1]),  # y_min
                #                  max(left_eye_landmarks[:, 0]),  # x_max
                #                  max(left_eye_landmarks[:, 1]))  # y_max
                #
                # # Convert the bounding box coordinates to the frame scale
                # left_eye_bbox = (int(left_eye_bbox[0] * frame.shape[1]),  # x_min
                #                  int(left_eye_bbox[1] * frame.shape[0]),  # y_min
                #                  int(left_eye_bbox[2] * frame.shape[1]),  # x_max
                #                  int(left_eye_bbox[3] * frame.shape[0]))  # y_max
                #
                # # Extract the ROI from the processed frame
                # left_eye_roi = frame[left_eye_bbox[1]:left_eye_bbox[3], left_eye_bbox[0]:left_eye_bbox[2]]

                # Resize the ROI to the original frame size
                # left_eye_zoomed_in = cv2.resize(left_eye_roi, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

                # frame = left_eye_roi



            # if processed is None or len(processed.shape) != 3:
            #     print("bad processed 485")
            #     exit(-1)

            # height, width = processed.shape[:2]  # Get the height and width of the image
            # processed = cv2.resize(processed, (width//2, height//2))  # Resize the image to half its original size

            # transform the BGR frame in grayscale
            # tX = time.perf_counter()
            # processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            # print(f"gray={int(gray.nbytes / 1024)}kb")
            # if (print_timings):
            #     print(f"Time to convert to grayscale: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")


            # if processed is None or len(processed.shape) != 3:
            #     print("bad processed 572")
            #     exit(-1)
            # tX = time.perf_counter()
            # edges = cv2.Canny(processed, threshold1=50, threshold2=70)
            # if (print_timings):
            #     print(f"Time to Canny: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            #edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # processed = cv2.addWeighted(processed, 0.4, edges, 0.5, 0)

            # get the frame size
            frame_size = processed.shape[1], processed.shape[0]

            # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from processed image to give it to the model
            # tX = time.perf_counter()
            # filtered = cv2.bilateralFilter(processed, 5, 10, 10)

            # This takes the grayscale image and triples it so we end up with the same size image!
            # But without that, face processing crashes
            # processed = np.expand_dims(processed, axis=2)
            # processed = np.concatenate([processed, processed, processed], axis=2)

            # if (print_timings):
            #     print(f"Time to bilateral filter: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            # if processed is None or len(processed.shape) != 3:
            #     print("bad processed 525")
            #     exit(-1)

            prediction = None

            if mode == 0:
                tX = time.perf_counter()
                results = process_image(detector, "", processed, 0)
                if (print_timings):
                    print(f"Time to find eye: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                if results is not None:
                    prediction, cleaned, steps = clean_eye(results[1])

                    rolling_buffer.append(steps)

                num_cols = 10
                num_rows = 12
                img_width = 99
                img_height = 33

                for i in range(min(7, len(rolling_buffer))):
                    row = i
                    rb = rolling_buffer[i]
                    for col in range(0, len(rb)):
                        x = col * img_width
                        y = row * img_height
                        if y + img_height <= processed.shape[0] and x + img_width <= processed.shape[1]:
                            processed[y:y+img_height, x:x+img_width] = rb[col]
                            #cv2.putText(processed, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                # for i in range(min(num_rows * num_cols, len(rolling_buffer))):
                #     row = i // num_cols
                #     col = i % num_cols
                #     x = col * img_width
                #     y = row * img_height
                #     if y + img_height <= processed.shape[0] and x + img_width <= processed.shape[1]:
                #         processed[y:y+img_height, x:x+img_width] = rolling_buffer[i]
                #         cv2.putText(processed, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

                while (len(rolling_buffer) > 120):
                    rolling_buffer.pop(0)

            elif mode == 1:
                tX = time.perf_counter()
                _, just_eye_img = process_image(detector, "", processed, 0)

                rolling_buffers.append([just_eye_img])
                for i in range(0, len(rolling_buffers) - 1):
                    if (len(rolling_buffers[i]) < TrainingConstants.IMAGES_SHOWN_TO_MODEL):
                        rolling_buffers[i].append(just_eye_img)

                if len(rolling_buffer) < TrainingConstants.IMAGES_SHOWN_TO_MODEL:
                    rolling_buffer.append(i)

                # if (len(rolling_buffer) > TrainingConstants.IMAGES_SHOWN_TO_MODEL):
                #     rolling_buffer.pop(0)
                if (print_timings):
                    print(f"Time to find eye: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                processed[0:33,0:99] = just_eye_img

                prediction_multi = None

                # This isn't right anyway as we need to execute every frame against last few frames
                # if len(rolling_buffer) >= TrainingConstants.IMAGES_SHOWN_TO_MODEL:
                #     tX = time.perf_counter()
                #     prediction = predict2(rolling_buffer, model)
                #     rolling_buffers = []
                #     if (print_timings):
                #         print(f"Time to predict: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                if len(rolling_buffers) >= 1:
                    tX = time.perf_counter()
                    filled_rolling_buffers = []
                    to_remove = []
                    for i in range(0, len(rolling_buffers)):
                        if (len(rolling_buffers[i]) == TrainingConstants.IMAGES_SHOWN_TO_MODEL):
                            filled_rolling_buffers.append(rolling_buffers[i])
                            to_remove.insert(0, i)
                    for i in to_remove:
                        rolling_buffers.pop(i)
                    if (len(filled_rolling_buffers) > 0):
                        #print(f"filled_rolling_buffers={len(filled_rolling_buffers)}")
                        latest = filled_rolling_buffers[-1]
                        image_idx = 0
                        for image in latest:
                            start = 200 + image_idx * 99
                            processed[33:66,start:start+99] = image
                            image_idx += 1
                        prediction = predict_multi(filled_rolling_buffers, model)
                        if (print_timings):
                            print(f"Time to predict: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                text_list.append("rolling_buffers size: " + str(len(rolling_buffers)))

                filled_rolling_buffers_count = 0
                for i in range(0, len(rolling_buffers)):
                    if (len(rolling_buffers[i]) == TrainingConstants.IMAGES_SHOWN_TO_MODEL):
                        filled_rolling_buffers_count += 1

                text_list.append("filled rolling_buffers count: " + str(filled_rolling_buffers_count))

            # old way
            else:
                # find the faces using the face mesh model
                tX = time.perf_counter()
                # todo should be cv2.cvtColor(img, cv2.COLOR_BGR2RGB) as:
                # Converts the image from BGR to RGB color space because the FaceMesh model expects images in RGB format.
                lms = detector.process(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)).multi_face_landmarks
                if (print_timings):
                    print(f"Time to find faces: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                perclos_rolling_score_v2 = None

                if lms:  # process the frame only if at least a face is found
                    present_values.append(1)



                    # prediction = None
                    # if len(rolling_buffer) == TrainingConstants.IMAGES_SHOWN_TO_MODEL:
                    #     tX = time.perf_counter()
                    #     prediction = predict2(rolling_buffer, model)
                    #     if (print_timings):
                    #         print(f"Time to predict: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # getting face landmarks and then take only the bounding box of the biggest face
                    tX = time.perf_counter()
                    landmarks = _get_landmarks(lms)
                    if (print_timings):
                        print(f"Time to get landmarks: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # shows the eye keypoints (can be commented)
                    tX = time.perf_counter()
                    Eye_det.show_eye_keypoints(
                        color_frame=processed, landmarks=landmarks, frame_size=frame_size)
                    if (print_timings):
                        print(f"Time to show eye keypoints: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # compute the EAR score of the eyes
                    tX = time.perf_counter()

                    ear, ear_left, ear_right = Eye_det.get_EAR(frame=processed, landmarks=landmarks)
                    # Intentionally flipped here, into my physical left eye (not the eye on screen left)
                    if flip_eye_mode:
                        temp = ear_right
                        ear_right = ear_left
                        ear_left = temp
                    if (print_timings):
                        print(f"Time to get EAR: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")


                    # Assuming `frame` is your current video frame and `ear` is the current EAR score
                    # tX = time.perf_counter()
                    # ear_plotter.update_ear_scores(ear)  # Update the plot data
                    # ear_plotter.overlay_graph_on_frame(frame)  # Overlay the graph on the frame
                    # if (print_timings) print(f"Time to update and overlay graph: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")


                    # Display the frame with the overlay

                    #cv2.imshow('Frame with EAR Graph', frame)


                    # compute the PERCLOS score and state of tiredness
                    tX = time.perf_counter()
                    # tired, perclos_score = Scorer.get_PERCLOS(t_now, fps, ear)

                    # _, perclos_rolling_score_v2 = Scorer.get_PERCLOS_rolling_v2(t_now, fps, ear, save_to_influx_every_x_seconds)
                    _, perclos_rolling_score_v3 = Scorer.get_PERCLOS_rolling_v3(t_now, fps, ear_left)
                    if (print_timings):
                        print(f"Time to get PERCLOS: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # tX = time.perf_counter()
                    # perclos_plotter.update_ear_scores(perclos_rolling_score_v3)  # Update the plot data
                    # perclos_plotter.overlay_graph_on_frame(frame)  # Overlay the graph on the frame
                    # if (print_timings) print(f"Time to update and overlay graph: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # compute the Gaze Score
                    tX = time.perf_counter()
                    gaze = Eye_det.get_Gaze_Score(
                        frame=processed, landmarks=landmarks, frame_size=frame_size)
                    if (print_timings):
                        print(f"Time to get Gaze Score: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")


                    if ear is not None:
                        blink_detector.update_ear(ear_left)
                        ear_values.append(ear)
                        ear_left_values.append(ear_left)
                        ear_right_values.append(ear_right)
                    if gaze is not None:
                        gaze_values.append(gaze)

                    # if perclos_score is not None:
                    #     perclos_values.append(perclos_score)

                    # compute the head pose
                    # tX = time.perf_counter()
                    # frame_det, roll, pitch, yaw = Head_pose.get_pose(
                    #     frame=frame, landmarks=landmarks, frame_size=frame_size)
                    # if (print_timings):
                    #     print(f"Time to get head pose: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # evaluate the scores for EAR, GAZE and HEAD POSE
                    # tX = time.perf_counter()
                    # asleep, looking_away, distracted = Scorer.eval_scores(t_now=t_now,
                    #                                                       ear_score=ear,
                    #                                                       gaze_score=gaze,
                    #                                                       head_roll=roll,
                    #                                                       head_pitch=pitch,
                    #                                                       head_yaw=yaw)
                    # if (print_timings) print(f"Time to evaluate scores: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # show the real-time EAR score
                    if ear is not None:
                        text_list.append("EAR:" + str(round(ear, 3)))
                        text_list.append(f"EAR PHYSICAL LEFT: {round(ear_left, 3)}")
                        text_list.append(f"EAR PHYSICAL RIGHT: {round(ear_right, 3)}")
                        draw_ear_between_eyes(processed, landmarks, ear, ear_left, ear_right)

                    if perclos_rolling_score_v3 is not None:
                        text_list.append("PERCLOS ROLLING (V3):" + str(round(perclos_rolling_score_v3, 3)))

                    if perclos_rolling_score_v2 is not None:
                        text_list.append("PERCLOS ROLLING (V2):" + str(round(perclos_rolling_score_v2, 3)))

                    blink_count_per_min, blink_durations = blink_detector.get_blink_data_all()
                    blink_count_recent, blink_durations_recent = blink_detector.get_blink_data_recent(5)


                    if blink_count_per_min is not None:
                        text_list.append("BLINK COUNT (60s):" + str(round(blink_count_per_min, 3)))

                    if blink_durations is not None:
                        text_list.append("BLINK DURATION (60s):" + str(round(blink_durations, 3)))

                    text_list.append("BLINK COUNT (5s):" + str(round(blink_count_recent, 3)))
                    text_list.append("BLINK Duration (5s):" + str(round(blink_durations_recent, 3)))


                else:
                    present_values.append(0)

            text_list.append(f"FPS Capture: {capture_fps}")
            text_list.append(f"FPS Process: {process_fps}")
            text_list.append(f"FPS Store  : {save_fps}")
            text_list.append(f"Flip camera mode: {flip_mode}")
            text_list.append(f"Flip eye mode: {flip_eye_mode}")
            text_list.append(f"Capture mode: {capture_mode}")
            text_list.append(f"Dump mode: {dump_buffered_frames}")
            text_list.append(f"Buffer mode: {buffer_mode}")
            text_list.append(f"Save queue: {frame_queue_for_saving.qsize()}")
            text_list.append(f"Process queue: {frame_queue_for_processing.qsize()}")
            text_list.append(f"Saving to influx: {saving_to_influx}")
            text_list.append(f"Prediction: {prediction}")


            total_size = 0
            for bframe, _, _, _, _ in buffered_frames:
                total_size += bframe.nbytes

            text_list.append(f"Buffered to save: {len(buffered_frames)} frames {round(total_size / 1024 / 1024, 0)} MB")

            position = 1
            for text in text_list:
                cv2.putText(processed, text, (10, position * 23), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                position += 1

            # if processed is None or len(processed.shape) != 3:
            #     print("bad processed 658")
            #     exit(-1)

            if saving_to_influx and (time.perf_counter() - t_last_save) > save_to_influx_every_x_seconds:
                t_last_save = time.perf_counter()

                blink_count_per_min, blink_durations = blink_detector.get_blink_data_all()
                blink_count_recent, blink_durations_recent = blink_detector.get_blink_data_recent(5)
                #print(f"Blink Count: {blink_count_per_min}, Blink Durations: {blink_durations}")
                # Save blink_count and blink_durations to InfluxDB or any storage

                blink_durations = int(blink_durations * 1000)  # Convert to milliseconds

                average_ear = sum(ear_values) / len(ear_values) if ear_values else None
                average_ear_left = sum(ear_left_values) / len(ear_left_values) if ear_left_values else None
                average_ear_right = sum(ear_right_values) / len(ear_right_values) if ear_right_values else None
                average_gaze = sum(gaze_values) / len(gaze_values) if gaze_values else None

                # Just get a friendly number, and presumably less precision is cheaper to store
                average_ear = int(average_ear * 100) if (average_ear  is not None and not math.isnan(average_ear) and not math.isinf(average_ear)) else None
                average_ear_left = int(average_ear_left * 100) if (average_ear_left  is not None and not math.isnan(average_ear_left) and not math.isinf(average_ear_left)) else None
                average_ear_right = int(average_ear_right * 100) if (average_ear_right is not None and not math.isnan(average_ear_right) and not math.isinf(average_ear_right)) else None
                average_gaze = int(average_gaze * 1000) if (average_gaze is not None and not math.isnan(average_gaze) and not math.isinf(average_gaze)) else None

                # worst_perclos = max(perclos_values) if perclos_values else None

                # pct_tired = tired_values.count(True) / len(tired_values) if tired_values else None
                # pct_distracted = distracted_values.count(True) / len(distracted_values) if distracted_values else None
                # pct_looking_away = looking_away_values.count(True) / len(looking_away_values) if looking_away_values else None

                pct_present = sum(present_values) / len(present_values) if present_values else 0

                i = 0
                ear_values = []
                ear_left_values = []
                ear_right_values = []

                gaze_values = []
                # perclos_values = []
                # tired_values = []
                # distracted_values = []
                # looking_away_values = []
                present_values = []


                # Write data point to the "XL" bucket
                try:
                    # Names do go on the wire but take minimal space in db
                    value = f"fatigue,host={hostname} present={pct_present}"

                    # All the perclos values are already % closed over some time frame, so there seems no better to averaging
                    # them further

                    if (pct_present > 0):
                        if (perclos_rolling_score_v2 != None):
                            value += f",perclosV2={perclos_rolling_score_v2}"
                        if (perclos_rolling_score_v3 != None):
                            value += f",perclosV3={perclos_rolling_score_v3}"
                        if (average_ear != None):
                            value += f",ear={average_ear}"
                            value += f",earLeft={average_ear_left}"
                            value += f",earRight={average_ear_right}"
                        if (average_gaze != None):
                            value += f",gaze={average_gaze}"


                    # Don't record blinks if I'm not at the computer
                    if (pct_present > 0.8):
                        if (blink_count_per_min != None):
                            value += f",blinks={blink_count_per_min}"
                        # Already a rolling average
                        if (blink_durations != None):
                            value += f",blinkDurations={blink_durations}"
                        value += f",blinksV2={blink_count_recent}"
                        value += f",blinkDurationsV2={blink_durations_recent}"
                        value += f",fpsCapture={round(capture_fps)}"
                        value += f",fpsProcess={round(process_fps)}"
                        value += f",queue={round(frame_queue_for_processing.qsize())}"

                    value += f" {int(time.time())}"
                    print(f"Writing data to InfluxDB: {value}")
                    client.write([value],write_precision='s')
                except Exception as e:
                    # Improved error message
                    error_type = type(e).__name__
                    print(f"Failed to write data to InfluxDB due to {error_type}: {e}")
                    print("Please check your InfluxDB configurations, network connection, and ensure the InfluxDB service is running.")


            #print(f"Got frame from webcam orig={int(frame.nbytes / 1024)}kb processed={int(processed.nbytes / 1024)}kb")

            frame_queue_for_saving.put([frame, processed])

            # if processed is None or len(processed.shape) != 3:
            #     print("bad processed 756")
            #     exit(-1)

            # if (frame_idx % 60 == 0):
            if True:
                # show the frame on screen
                tX = time.perf_counter()
                cv2.imshow("Press 'q' to terminate, 'c' to toggle saving (for debug), 's' to save buffered frames, 'b' to buffer frames (for training), 'p' to print timings, 'f' to change flip_mode", processed)
                if (print_timings):
                    print(f"Time to draw frame: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                # if the key "q" is pressed on the keyboard, the program is terminated
                tX = time.perf_counter()
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    capture_mode = not capture_mode
                elif key == ord('f'):
                    flip_mode += 1
                    if flip_mode > 3:
                        flip_mode = 0
                elif key == ord('q'):
                    done = True
                    exit(0)
                elif key == ord('s'):
                    dump_buffered_frames = True
                elif key == ord('b'):
                    buffer_mode = not buffer_mode
                elif key == ord('p'):
                    print_timings = not print_timings
                elif key == ord('e'):
                    flip_eye_mode = not flip_eye_mode
                if (print_timings):
                    print(f"Time to wait key: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            if (print_timings):
                print(f"Time for total frame: {(time.perf_counter() - t_now) * 1000} {(time.perf_counter() - tX) * 1000}")

            i += 1
    except Exception as e:
        print("process_frames thread ended: " + e)

# Create and start the threads
thread_capture = threading.Thread(target=capture_frames, daemon=True)
thread_save = threading.Thread(target=save_frames, daemon=True)
thread_process = threading.Thread(target=process_frames, daemon=True)

if args.input == "":
    thread_capture.start()

thread_save.start()
thread_process.start()

# Keep the main thread alive
try:
    while done is False:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")
print("done1")
done = True
thread_capture.join()
print("thread_capture ended")
thread_save.join()
print("thread_save ended")
thread_process.join()
print("thread_process ended")

os._exit(-1)
print("done2")