import math
import time

import cv2
import mediapipe as mp
import queue
import traceback
from datetime import datetime

import TrainingConstants
import approaches.MediapipeEARMultiFrame as MediapipeEARMultiFrame
from Attention_Scorer_Module import AttentionScorer as AttScorer
from BlinkDetector import BlinkDetector
from EyeImage import clean_eye
from Eye_Dector_Module import EyeDetector as EyeDet
from GlobalContext import GlobalContext
from ModelPredict import predict_multi
from RealTimeEARPlot import RealTimeEARPlot
from RealTimePERCLOSPlot import RealTimePERCLOSPlot
from TrainingProcess import process_image
from approaches.MediapipeEARMultiFrame.Approach import handle_new_second, handle_image, write_to_influx, \
    handle_new_minute
from approaches.MediapipeEARMultiFrame.ApproachContext import ApproachContext


# from influxdb_client_3 import InfluxDBClient3


def process_frames(gc: GlobalContext):
    try:
        print("Started process frames thread")
        args = gc.args
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
        time.sleep(0.01)  # To prevent zero division error when calculating the FPS
        t_last_save = time.perf_counter()
        t_last_image_save = t_last_save

        save_to_influx_every_x_seconds = 5
        saving_to_influx = True

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
        prev_minute = None
        rolling_buffers = [[]]
        rolling_buffer = []

        currently_blinking = False
        last_processed = None
        ac = None
        if gc.mode == 3:
            ac = ApproachContext(gc)

        while gc.done is False:
            t_now = time.perf_counter()
            period_start_time = time.perf_counter()
            current_time = datetime.now()
            current_second = current_time.strftime("%S")
            current_minute = current_time.strftime("%M")
            text_list = []

            if prev_second is None or prev_second != current_second:
                prev_second = current_second
                gc.process_fps = frame_idx

                handle_new_second(ac, current_time, frame_idx)

                frame_idx = 0  # Reset frame index for each new second
            else:
                frame_idx += 1

            if prev_minute is None or prev_minute != current_minute:
                prev_minute = current_minute
                handle_new_minute(ac, current_time, frame_idx)

            fps = i / (t_now - t_last_save)
            if fps == 0:
                fps = 10

            try:
                #print("Waiting for frame")
                frame = gc.frame_queue_for_processing.get(timeout=1)
            except queue.Empty:
                continue

            #print(f"Got frame for processing {frame.nbytes / 1024}kb")

            frame = flip_frame(frame, gc, t_now)

            # start the tick counter for computing the processing time for each frame
            # e1 = cv2.getTickCount()

            # height, width = frame.shape[:2]  # Get the height and width of the image
            # tiny = cv2.resize(frame, (width//4, height//4))

            processed = frame.copy()

            frame_size = processed.shape[1], processed.shape[0]

            prediction = None
            tMode = time.perf_counter()

            if gc.mode == 0:
                tX = time.perf_counter()
                results = process_image(detector, "", processed, 0)
                if (gc.print_timings):
                    print(
                        f"Time to find eye: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

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
                            processed[y:y + img_height, x:x + img_width] = rb[col]
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

            elif gc.mode == 1:
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
                if (gc.print_timings):
                    print(
                        f"Time to find eye: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                processed[0:33, 0:99] = just_eye_img

                prediction_multi = None

                # This isn't right anyway as we need to execute every frame against last few frames
                # if len(rolling_buffer) >= TrainingConstants.IMAGES_SHOWN_TO_MODEL:
                #     tX = time.perf_counter()
                #     prediction = predict2(rolling_buffer, model)
                #     rolling_buffers = []
                #     if (gc.print_timings):
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
                            processed[33:66, start:start + 99] = image
                            image_idx += 1
                        prediction = predict_multi(filled_rolling_buffers, model)
                        if (gc.print_timings):
                            print(
                                f"Time to predict: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                text_list.append("rolling_buffers size: " + str(len(rolling_buffers)))

                filled_rolling_buffers_count = 0
                for i in range(0, len(rolling_buffers)):
                    if (len(rolling_buffers[i]) == TrainingConstants.IMAGES_SHOWN_TO_MODEL):
                        filled_rolling_buffers_count += 1

                text_list.append("filled rolling_buffers count: " + str(filled_rolling_buffers_count))

            elif gc.mode == 2:
                # find the faces using the face mesh model
                tX = time.perf_counter()
                # todo should be cv2.cvtColor(img, cv2.COLOR_BGR2RGB) as:
                # Converts the image from BGR to RGB color space because the FaceMesh model expects images in RGB format.
                lms = detector.process(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)).multi_face_landmarks
                if (gc.print_timings):
                    print(
                        f"Time to find faces: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                perclos_rolling_score_v2 = None

                if lms:  # process the frame only if at least a face is found
                    present_values.append(1)

                    # prediction = None
                    # if len(rolling_buffer) == TrainingConstants.IMAGES_SHOWN_TO_MODEL:
                    #     tX = time.perf_counter()
                    #     prediction = predict2(rolling_buffer, model)
                    #     if (gc.print_timings):
                    #         print(f"Time to predict: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # getting face landmarks and then take only the bounding box of the biggest face
                    tX = time.perf_counter()
                    landmarks = _get_landmarks(lms)
                    if (gc.print_timings):
                        print(
                            f"Time to get landmarks: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # shows the eye keypoints (can be commented)
                    tX = time.perf_counter()
                    Eye_det.show_eye_keypoints(
                        color_frame=processed, landmarks=landmarks, frame_size=frame_size)
                    if (gc.print_timings):
                        print(
                            f"Time to show eye keypoints: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # compute the EAR score of the eyes
                    tX = time.perf_counter()

                    ear, ear_left, ear_right = Eye_det.get_EAR(frame=processed, landmarks=landmarks)
                    # Intentionally flipped here, into my physical left eye (not the eye on screen left)
                    if flip_eye_mode:
                        temp = ear_right
                        ear_right = ear_left
                        ear_left = temp
                    if (gc.print_timings):
                        print(
                            f"Time to get EAR: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # Assuming `frame` is your current video frame and `ear` is the current EAR score
                    # tX = time.perf_counter()
                    # ear_plotter.update_ear_scores(ear)  # Update the plot data
                    # ear_plotter.overlay_graph_on_frame(frame)  # Overlay the graph on the frame
                    # if (gc.print_timings) print(f"Time to update and overlay graph: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # Display the frame with the overlay

                    #cv2.imshow('Frame with EAR Graph', frame)

                    # compute the PERCLOS score and state of tiredness
                    tX = time.perf_counter()
                    # tired, perclos_score = Scorer.get_PERCLOS(t_now, fps, ear)

                    # _, perclos_rolling_score_v2 = Scorer.get_PERCLOS_rolling_v2(t_now, fps, ear, save_to_influx_every_x_seconds)
                    _, perclos_rolling_score_v3 = Scorer.get_PERCLOS_rolling_v3(t_now, fps, ear_left)
                    if (gc.print_timings):
                        print(
                            f"Time to get PERCLOS: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # tX = time.perf_counter()
                    # perclos_plotter.update_ear_scores(perclos_rolling_score_v3)  # Update the plot data
                    # perclos_plotter.overlay_graph_on_frame(frame)  # Overlay the graph on the frame
                    # if (gc.print_timings) print(f"Time to update and overlay graph: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # compute the Gaze Score
                    tX = time.perf_counter()
                    gaze = Eye_det.get_Gaze_Score(
                        frame=processed, landmarks=landmarks, frame_size=frame_size)
                    if (gc.print_timings):
                        print(
                            f"Time to get Gaze Score: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

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
                    # if (gc.print_timings):
                    #     print(f"Time to get head pose: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                    # evaluate the scores for EAR, GAZE and HEAD POSE
                    # tX = time.perf_counter()
                    # asleep, looking_away, distracted = Scorer.eval_scores(t_now=t_now,
                    #                                                       ear_score=ear,
                    #                                                       gaze_score=gaze,
                    #                                                       head_roll=roll,
                    #                                                       head_pitch=pitch,
                    #                                                       head_yaw=yaw)
                    # if (gc.print_timings) print(f"Time to evaluate scores: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

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
            elif gc.mode == 3:
                ac: ApproachContext = ac
                handle_image(ac, processed, text_list)

            if (gc.print_timings):
                print(
                    f"Time to do mode: {(time.perf_counter() - tMode) * 1000} {(time.perf_counter() - t_now) * 1000}")

            text_list.append(f"FPS Capture: {gc.capture_fps}")
            text_list.append(f"FPS Process: {gc.process_fps}")
            text_list.append(f"FPS Store  : {gc.save_fps}")
            text_list.append(f"Flip camera mode: {gc.flip_mode}")
            text_list.append(f"Flip eye mode: {gc.flip_eye_mode}")
            text_list.append(f"Capture mode: {gc.capture_mode}")
            text_list.append(f"Dump mode: {gc.dump_buffered_frames}")
            text_list.append(f"Buffer mode: {gc.buffer_mode}")
            text_list.append(f"Debug mode: {gc.debug_mode}")
            text_list.append(f"Save queue: {gc.frame_queue_for_saving.qsize()}")
            text_list.append(f"Process queue: {gc.frame_queue_for_processing.qsize()}")
            text_list.append(f"Saving to influx: {gc.saving_to_influx}")
            text_list.append(f"Prediction: {prediction}")
            text_list.append(f"Mode: {gc.mode}")

            total_size = 0
            for bframe, _, _, _, _ in gc.buffered_frames:
                total_size += bframe.nbytes

            text_list.append(
                f"Buffered to save: {len(gc.buffered_frames)} frames {round(total_size / 1024 / 1024, 0)} MB")

            position = 1
            for text in text_list:
                cv2.putText(processed, text, (10, position * 23), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                position += 1

            # if processed is None or len(processed.shape) != 3:
            #     print("bad processed 658")
            #     exit(-1)

            if saving_to_influx and (time.perf_counter() - t_last_save) > save_to_influx_every_x_seconds:
                t_last_save = time.perf_counter()

                if gc.mode == 2:
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
                    average_ear = int(average_ear * 100) if (
                                average_ear is not None and not math.isnan(average_ear) and not math.isinf(
                            average_ear)) else None
                    average_ear_left = int(average_ear_left * 100) if (
                                average_ear_left is not None and not math.isnan(average_ear_left) and not math.isinf(
                            average_ear_left)) else None
                    average_ear_right = int(average_ear_right * 100) if (
                                average_ear_right is not None and not math.isnan(average_ear_right) and not math.isinf(
                            average_ear_right)) else None
                    average_gaze = int(average_gaze * 1000) if (
                                average_gaze is not None and not math.isnan(average_gaze) and not math.isinf(
                            average_gaze)) else None

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
                            value += f",fpsCapture={round(gc.capture_fps)}"
                            value += f",fpsProcess={round(gc.process_fps)}"
                            value += f",queue={round(gc.frame_queue_for_processing.qsize())}"

                        value += f" {int(time.time())}"
                        print(f"Writing data to InfluxDB: {value}")
                        client.write([value], write_precision='s')
                    except Exception as e:
                        # Improved error message
                        error_type = type(e).__name__
                        print(f"Failed to write data to InfluxDB due to {error_type}: {e}")
                        print(
                            "Please check your InfluxDB configurations, network connection, and ensure the InfluxDB service is running.")
                elif gc.mode == 3:
                    write_to_influx(gc, ac)

            #print(f"Got frame from webcam orig={int(frame.nbytes / 1024)}kb processed={int(processed.nbytes / 1024)}kb")

            gc.frame_queue_for_saving.put([frame, processed])

            # if processed is None or len(processed.shape) != 3:
            #     print("bad processed 756")
            #     exit(-1)

            if True:
            #if (frame_idx % 20 == 0):
                # show the frame on screen
                tX = time.perf_counter()
                cv2.imshow(
                    "Press 'q' to terminate, 'c' to toggle saving (for debug), 's' to save buffered frames, 'b' to buffer frames (for training), 'p' to print timings, 'l' to save one frame, 'd' for debug",
                    processed)
                if (gc.print_timings):
                    print(
                        f"Time to draw frame: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

                # if the key "q" is pressed on the keyboard, the program is terminated
                tX = time.perf_counter()
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    gc.capture_mode = not gc.capture_mode
                elif key == ord('f'):
                    gc.flip_mode += 1
                    if gc.flip_mode > 3:
                        gc.flip_mode = 0
                elif key == ord('q'):
                    gc.done = True
                    exit(0)
                elif key == ord('s'):
                    gc.dump_buffered_frames = True
                elif key == ord('b'):
                    gc.buffer_mode = not gc.buffer_mode
                elif key == ord('p'):
                    gc.print_timings = not gc.print_timings
                elif key == ord('e'):
                    gc.flip_eye_mode = not gc.flip_eye_mode
                elif key == ord('l'):
                    gc.capture_single_frame_mode = True
                elif key == ord('d'):
                    gc.debug_mode = not gc.debug_mode
                if (gc.print_timings):
                    print(
                        f"Time to wait key: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")

            if (gc.print_timings):
                print(
                    f"Time for total frame: {(time.perf_counter() - t_now) * 1000} {(time.perf_counter() - tX) * 1000}")

            i += 1
    except Exception as e:
        print("process_frames thread ended: " + str(e))
        print("Stack trace: " + traceback.format_exc())


def flip_frame(frame, gc, t_now):
    # if the frame comes from webcam, flip it so it looks like a mirror.
    tX = time.perf_counter()
    # if args.camera == 0:
    #     frame = cv2.flip(frame, 2)
    # elif args.camera == 1:
    # print(f"Flip is {args.flip}")
    # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if gc.flip_mode == 1:
        # print(f"Flipping")

        # This is 100% one of the modes I want, in this order!
        # The Pixel has its own algo for if/how it flips the output also.
        frame = cv2.flip(frame, 2)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif gc.flip_mode == 2:
        # print(f"Flipping")

        # This is 100% one of the modes I want, in this order!
        # The Pixel has its own algo for if/how it flips the output also.
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = cv2.flip(frame, 1)
    elif gc.flip_mode == 3:
        frame = cv2.flip(frame, 1)
    # frame = cv2.flip(frame, 2)
    # frame = zoom_in(frame, 2)
    if (gc.print_timings):
        print(f"Time to flip frame: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - t_now) * 1000}")
    return frame
