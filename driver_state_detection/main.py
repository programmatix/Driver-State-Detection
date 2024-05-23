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
from Attention_Scorer_Module import AttentionScorer as AttScorer
from BlinkDetector import BlinkDetector
from Eye_Dector_Module import EyeDetector as EyeDet
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from RealTimeEARPlot import RealTimeEARPlot
from RealTimePERCLOSPlot import RealTimePERCLOSPlot
from datetime import datetime
from dotenv import load_dotenv
from influxdb_client_3 import InfluxDBClient3

#from hdrhistogram import HdrHistogram

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
    - frame: The original webcam frame.
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
    frame_left_y = int(left_eye_point[1] * frame.shape[0])

    frame_right_x = int(right_eye_point[0] * frame.shape[1])
    frame_right_y = int(right_eye_point[1] * frame.shape[0])

    #print(f"Midpoint: ({frame_midpoint_x}, {frame_midpoint_y}) left: {left_eye_point} right: {right_eye_point} midpoint: {midpoint_x}, {midpoint_y}")

    # Display the EAR score at the calculated midpoint
    cv2.putText(frame, f"EAR: {round(ear, 3)}", (frame_midpoint_x, frame_midpoint_y),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.putText(frame, f"{round(ear_left, 3)}", (frame_left_x, frame_left_y),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.putText(frame, f"{round(ear_right, 3)}", (frame_right_x, frame_right_y),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

# Create a thread-safe queue
frame_queue_for_saving = queue.Queue()
frame_queue_for_processing = queue.Queue()

print_timings = False
capture_fps = 0
save_fps = 0
process_fps = 0
capture_mode = False

def capture_frames():
    global capture_fps
    # p = psutil.Process(os.getpid())
    # # Set the process priority to above normal, this can be adjusted to your needs
    # p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    # # Additionally, set the thread priority if needed
    # threading.current_thread().priority = threading.PRIORITY_HIGHEST

    frame_idx = 0
    prev_second = None

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    cap.set(cv2.VIDEO_ACCELERATION_ANY, 1)
    #histogram = HdrHistogram()
    while True:
        tX = time.perf_counter()
        ret, frame = cap.read()

        current_time = datetime.now()
        current_second = current_time.strftime("%S")

        if prev_second is None or prev_second != current_second:
            prev_second = current_second
            capture_fps = frame_idx
            frame_idx = 0  # Reset frame index for each new second
        else:
            frame_idx += 1

        if not ret:
            print("Can't receive frame from camera/stream end")
            time.sleep(1)
            cap = cv2.VideoCapture(0)
        else:
            if (print_timings):
                #histogram.record_value((time.perf_counter() - tX) * 1000)
                print(f"Time to read frame: {(time.perf_counter() - tX) * 1000}")
            frame_queue_for_processing.put(frame)

        #Every second display histogram


def save_frames():
    global save_fps
    frame_idx = 0
    prev_second = None
    while True:
        frame = frame_queue_for_saving.get()
        current_time = datetime.now()
        current_second = current_time.strftime("%S")

        if prev_second is None or prev_second != current_second:
            prev_second = current_second
            save_fps = frame_idx
            frame_idx = 0  # Reset frame index for each new second
        else:
            frame_idx += 1

        # if (frame_idx == 0):
        #     print("New second " + timestamp)

        if capture_mode:
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S") + "-" + str(frame_idx)
            filename = f"output_images/{timestamp}.jpg"

            # Don't compress - hard enough to debug
            #cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(filename, frame)
        # print(f"Saved {filename}")

def process_frames():
    global process_fps
    global capture_mode
    try:
        parser = argparse.ArgumentParser(description='Driver State Detection')

        # selection the camera number, default is 0 (webcam)
        parser.add_argument('-c', '--camera', type=int,
                            default=0, metavar='', help='Camera number, default is 0 (webcam)')

        # TODO: add option for choose if use camera matrix and dist coeffs

        # visualisation parameters
        parser.add_argument('--show_fps', type=bool, default=True,
                            metavar='', help='Show the actual FPS of the capture stream, default is true')
        parser.add_argument('--show_proc_time', type=bool, default=True,
                            metavar='', help='Show the processing time for a single frame, default is true')
        parser.add_argument('--show_eye_proc', type=bool, default=False,
                            metavar='', help='Show the eyes processing, deafult is false')
        parser.add_argument('--show_axis', type=bool, default=True,
                            metavar='', help='Show the head pose axis, default is true')
        parser.add_argument('--verbose', type=bool, default=False,
                            metavar='', help='Prints additional info, default is false')

        # Attention Scorer parameters (EAR, Gaze Score, Pose)
        parser.add_argument('--smooth_factor', type=float, default=0.5,
                            metavar='', help='Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5')
        parser.add_argument('--ear_thresh', type=float, default=0.15,
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

        # parse the arguments and store them in the args variable dictionary
        args = parser.parse_args()

        if args.verbose:
            print(f"Arguments and Parameters used:\n{args}\n")

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
                                                   refine_landmarks=True)

        # instantiation of the eye detector and pose estimator objects
        Eye_det = EyeDet(show_processing=args.show_eye_proc)

        Head_pose = HeadPoseEst(show_axis=args.show_axis)

        # instantiation of the attention scorer object, with the various thresholds
        # NOTE: set verbose to True for additional printed information about the scores
        t0 = time.perf_counter()
        Scorer = AttScorer(t_now=t0, ear_thresh=args.ear_thresh, gaze_time_thresh=args.gaze_time_thresh,
                           roll_thresh=args.roll_thresh, pitch_thresh=args.pitch_thresh,
                           yaw_thresh=args.yaw_thresh, ear_time_thresh=args.ear_time_thresh,
                           gaze_thresh=args.gaze_thresh, pose_time_thresh=args.pose_time_thresh,
                           verbose=args.verbose)

        i = 0
        time.sleep(0.01) # To prevent zero division error when calculating the FPS
        t_last_save = time.perf_counter()
        t_last_image_save = t_last_save

        save_to_influx_every_x_seconds = 5

        ear_values = []
        ear_left_values = []
        ear_right_values = []
        gaze_values = []
        perclos_values = []
        tired_values = []
        distracted_values = []
        looking_away_values = []
        present_values = []
        average_ear = 0
        average_gaze = 0
        worst_perclos = 0
        pct_tired = 0
        pct_distracted = 0
        pct_looking_away = 0
        pct_present = 0


        # When average EAR was being used:
        # 0.20 avg too sensitive
        # 0.10 avg not picking up
        # 0.05 avg not picking up
        blink_detector = BlinkDetector(ear_threshold=0.08)  # Set your EAR threshold

        # Example usage
        ear_plotter = RealTimeEARPlot()
        perclos_plotter = RealTimePERCLOSPlot()

        frame_idx = 0
        prev_second = None
        while True:
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

            frame = frame_queue_for_processing.get()

            #print("Got frame for processing")

            # if the frame comes from webcam, flip it so it looks like a mirror.
            tX = time.perf_counter()
            if args.camera == 0:
                frame = cv2.flip(frame, 2)

            frame = zoom_in(frame, 2)
            if (print_timings):
                print(f"Time to process frame: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            # start the tick counter for computing the processing time for each frame
            e1 = cv2.getTickCount()

            # transform the BGR frame in grayscale
            tX = time.perf_counter()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (print_timings):
                print(f"Time to convert to grayscale: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            # get the frame size
            frame_size = frame.shape[1], frame.shape[0]

            # apply a bilateral filter to lower noise but keep frame details. create a 3D matrix from gray image to give it to the model
            tX = time.perf_counter()
            gray = np.expand_dims(cv2.bilateralFilter(gray, 5, 10, 10), axis=2)
            gray = np.concatenate([gray, gray, gray], axis=2)
            if (print_timings):
                print(f"Time to bilateral filter: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            # find the faces using the face mesh model
            tX = time.perf_counter()
            lms = detector.process(gray).multi_face_landmarks
            if (print_timings):
                print(f"Time to find faces: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            perclos_rolling_score_v2 = None

            if lms:  # process the frame only if at least a face is found
                present_values.append(1)

                # getting face landmarks and then take only the bounding box of the biggest face
                tX = time.perf_counter()
                landmarks = _get_landmarks(lms)
                if (print_timings):
                    print(f"Time to get landmarks: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                # shows the eye keypoints (can be commented)
                tX = time.perf_counter()
                Eye_det.show_eye_keypoints(
                    color_frame=frame, landmarks=landmarks, frame_size=frame_size)
                if (print_timings):
                    print(f"Time to show eye keypoints: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                # compute the EAR score of the eyes
                tX = time.perf_counter()
                ear, ear_left, ear_right = Eye_det.get_EAR(frame=gray, landmarks=landmarks)
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
                _, perclos_rolling_score_v3 = Scorer.get_PERCLOS_rolling_v3(t_now, fps, ear)
                if (print_timings):
                    print(f"Time to get PERCLOS: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                # tX = time.perf_counter()
                # perclos_plotter.update_ear_scores(perclos_rolling_score_v3)  # Update the plot data
                # perclos_plotter.overlay_graph_on_frame(frame)  # Overlay the graph on the frame
                # if (print_timings) print(f"Time to update and overlay graph: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                # compute the Gaze Score
                tX = time.perf_counter()
                gaze = Eye_det.get_Gaze_Score(
                    frame=gray, landmarks=landmarks, frame_size=frame_size)
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

                # tired_values.append(tired)
                # distracted_values.append(distracted)
                # looking_away_values.append(looking_away)


                # if the head pose estimation is successful, show the results
                # if frame_det is not None:
                #     frame = frame_det


                # adding_to_perclos = (ear is not None) and (ear <= Scorer.ear_thresh)
                # cv2.putText(frame, f"Adding to perclos: {adding_to_perclos}", (10, 260),
                #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                # show the real-time EAR score
                if ear is not None:
                    text_list.append("EAR:" + str(round(ear, 3)))
                    text_list.append(f"EAR LEFT: {round(ear_left, 3)}")
                    text_list.append(f"EAR RIGHT: {round(ear_right, 3)}")
                    # cv2.putText(frame, "EAR:" + str(round(ear, 3)), (10, 50),
                    #             cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 1, cv2.LINE_AA)
                    draw_ear_between_eyes(frame, landmarks, ear, ear_left, ear_right)

                # show the real-time Gaze Score
                # if gaze is not None:
                #     cv2.putText(frame, "Gaze Score:" + str(round(gaze, 3)), (10, 80),
                #                 cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)

                # show the real-time PERCLOS score
                # cv2.putText(frame, "PERCLOS:" + str(round(perclos_score, 3)), (10, 110),
                #             cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
                #
                if perclos_rolling_score_v3 is not None:
                    text_list.append("PERCLOS ROLLING (V3):" + str(round(perclos_rolling_score_v3, 3)))

                    # cv2.putText(frame, "PERCLOS ROLLING (V3):" + str(round(perclos_rolling_score_v3, 3)), (10, 140),
                    #                 cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)

                if perclos_rolling_score_v2 is not None:
                    # cv2.putText(frame, "PERCLOS ROLLING (V2):" + str(round(perclos_rolling_score_v2, 3)), (10, 170),
                    #             cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
                    text_list.append("PERCLOS ROLLING (V2):" + str(round(perclos_rolling_score_v2, 3)))

                blink_count_per_min, blink_durations = blink_detector.get_blink_data_all()
                blink_count_recent, blink_durations_recent = blink_detector.get_blink_data_recent(5)


                if blink_count_per_min is not None:
                    # cv2.putText(frame, "BLINK COUNT:" + str(round(blink_count_per_min, 3)), (10, 200),
                    #             cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
                    text_list.append("BLINK COUNT (60s):" + str(round(blink_count_per_min, 3)))

                if blink_durations is not None:
                    # cv2.putText(frame, "BLINK DURATION:" + str(round(blink_durations, 3)), (10, 230),
                    #             cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1, cv2.LINE_AA)
                    text_list.append("BLINK DURATION (60s):" + str(round(blink_durations, 3)))

                text_list.append("BLINK COUNT (5s):" + str(round(blink_count_recent, 3)))
                text_list.append("BLINK Duration (5s):" + str(round(blink_durations_recent, 3)))

                #text_list.append(f"Process frame time: ${proc_time_frame_ms}")
                text_list.append(f"FPS Capture: {capture_fps}")
                text_list.append(f"FPS Process: {process_fps}")
                text_list.append(f"FPS Store  : {save_fps}")
                text_list.append(f"Capture mode: {capture_mode}")

                position = 1
                for text in text_list:
                    cv2.putText(frame, text, (10, position * 23), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    position += 1


                # if roll is not None:
                #     cv2.putText(frame, "roll:"+str(roll.round(1)[0]), (450, 40),
                #                 cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
                # if pitch is not None:
                #     cv2.putText(frame, "pitch:"+str(pitch.round(1)[0]), (450, 70),
                #                 cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)
                # if yaw is not None:
                #     cv2.putText(frame, "yaw:"+str(yaw.round(1)[0]), (450, 100),
                #                 cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 1, cv2.LINE_AA)


                # if the driver is tired, show and alert on screen
                # if tired:
                #     cv2.putText(frame, "TIRED!", (10, 280),
                #                 cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                #
                # # if the state of attention of the driver is not normal, show an alert on screen
                # if asleep:
                #     cv2.putText(frame, "ASLEEP!", (10, 300),
                #                 cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                # if looking_away:
                #     cv2.putText(frame, "LOOKING AWAY!", (10, 320),
                #                 cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
                # if distracted:
                #     cv2.putText(frame, "DISTRACTED!", (10, 340),
                #                 cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                # cv2.putText(frame, str(time.perf_counter() - t_last_save), (10, 360),
                #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            else:
                present_values.append(0)

            # cv2.putText(frame, f"Avg ear: {average_ear}", (10, 360),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(frame, f"Avg gaze: {average_gaze}", (10, 380),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(frame, f"Worst perclos: {worst_perclos}", (10, 400),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(frame, f"Pct tired: {pct_tired}", (10, 420),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(frame, f"Pct distracted: {pct_distracted}", (10, 440),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(frame, f"Pct looking away: {pct_looking_away}", (10, 460),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.putText(frame, f"Pct present: {pct_present}", (10, 480),
            #             cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            if (time.perf_counter() - t_last_image_save) > (60 * 20):
                t_last_image_save = time.perf_counter()
                cv2.imwrite(f"output_images/{t_last_save}.jpg", frame)
                print("Image saved successfully.")

            if (time.perf_counter() - t_last_save) > save_to_influx_every_x_seconds:
                t_last_save = time.perf_counter()

                blink_count_per_min, blink_durations = blink_detector.get_blink_data_all()
                blink_count_recent, blink_durations_recent = blink_detector.get_blink_data_recent(5)
                print(f"Blink Count: {blink_count_per_min}, Blink Durations: {blink_durations}")
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
                    # if (worst_perclos != None):
                    #     value += f",perclos={worst_perclos}"
                    # if (pct_tired != None):
                    #     value += f",tired={pct_tired}"
                    # if (pct_distracted != None):
                    #     value += f",distracted={pct_distracted}"
                    # if (pct_looking_away != None):
                    #     value += f",lookingAway={pct_looking_away}"

                    value += f" {int(time.time())}"
                    print(f"Writing data to InfluxDB: {value}")
                    client.write([value],write_precision='s')
                except Exception as e:
                    # Improved error message
                    error_type = type(e).__name__
                    print(f"Failed to write data to InfluxDB due to {error_type}: {e}")
                    print("Please check your InfluxDB configurations, network connection, and ensure the InfluxDB service is running.")




            # stop the tick counter for computing the processing time for each frame
            e2 = cv2.getTickCount()
            # processign time in milliseconds
            proc_time_frame_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
            # print fps and processing time per frame on screen
            # if args.show_fps:
            #     cv2.putText(frame, "FPS:" + str(round(fps)), (10, 400), cv2.FONT_HERSHEY_PLAIN, 2,
            #                 (255, 0, 255), 1)
            # if args.show_proc_time:
            #     cv2.putText(frame, "PROC. TIME FRAME:" + str(round(proc_time_frame_ms, 0)) + 'ms', (10, 430), cv2.FONT_HERSHEY_PLAIN, 2,
            #                 (255, 0, 255), 1)

            frame_queue_for_saving.put(frame)

            # show the frame on screen
            tX = time.perf_counter()
            cv2.imshow("Press 'q' to terminate, 'c' to start saving, 'd' to stop", frame)
            if (print_timings):
                print(f"Time to draw frame: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            # if the key "q" is pressed on the keyboard, the program is terminated
            tX = time.perf_counter()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                capture_mode = True
            elif key == ord('d'):
                capture_mode = False
            elif key == ord('q'):
                break
            if (print_timings):
                print(f"Time to wait key: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            if (print_timings):
                print(f"Time for total frame: {(time.perf_counter() - t_now) * 1000} {(time.perf_counter() - tX) * 1000}")

            i += 1
    except  e:
        print("process_frames thread ended: " + e)

# Create and start the threads
thread_capture = threading.Thread(target=capture_frames, daemon=True)
thread_save = threading.Thread(target=save_frames, daemon=True)
thread_process = threading.Thread(target=process_frames, daemon=True)

thread_capture.start()
thread_save.start()
thread_process.start()

# Keep the main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopping...")