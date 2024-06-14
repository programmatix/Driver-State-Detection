# Expanding on the previous EAR single-frame approach, to consider EAR of previous frames too.
import time

import cv2
import numpy as np
import os
from typing_extensions import List

import approaches.MediapipeEARHeadPoseEpochs.TrainingConstants as tc
from GlobalContext import GlobalContext
from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst
from approaches.MediapipeEARHeadPoseEpochs import ApproachContext
from approaches.MediapipeEARHeadPoseEpochs.EyeDetector import EyeDetector2
from approaches.MediapipeEARHeadPoseEpochs.Model import ProcessedImage, EpochOngoing, EpochFinished


def get_landmarks(detector, img, debug):
    # if debug:
    #     print(f"Processing image with detector")
    # results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    results = detector.process(img)
    if results.multi_face_landmarks:
        return results.multi_face_landmarks
    else:
        return None


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


# def calculate_bounding_box(landmarks, img, eye_landmarks, debug):
#     min_x = max_x = int(landmarks[eye_landmarks[0]].x * img.shape[1])
#     min_y = max_y = int(landmarks[eye_landmarks[0]].y * img.shape[0])
#     # if debug:
#     #     print(f"Initial bounding box: {min_x}x{min_y} to {max_x}x{max_y}")
#     for i in eye_landmarks:
#         x = int(landmarks[i].x * img.shape[1])
#         y = int(landmarks[i].y * img.shape[0])
#         min_x = min(min_x, x)
#         max_x = max(max_x, x)
#         min_y = min(min_y, y)
#         max_y = max(max_y, y)
#         if debug:
#             # print(f"Landmark {i} ({eye_landmark_names[i]}) at {x}x{y} (min {min_x}x{min_y}, max {max_x}x{max_y})")
#             cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
#     return min_x, min_y, max_x, max_y
#
#
# def adjust_bounding_box(min_x, min_y, max_x, max_y, img, debug):
#     width = max_x - min_x
#     height = max_y - min_y
#     center_x = min_x + width // 2
#     center_y = min_y + height // 2
#     # if debug:
#     #     print(
#     #         f"Bounding box dimensions: {min_x}x{min_y} to {max_x}x{max_y} {width}x{height} ratio: {round(width / height, 2)}, want to centre on {center_x}x{center_y}")
#     if width / height < 3:
#         width = height * 3
#         min_x = center_x - width // 2
#         max_x = center_x + width // 2
#     elif height < width / 3:
#         height = width / 3
#         min_y = center_y - height // 2
#         max_y = center_y + height // 2
#     # if debug:
#     #     print(f"Adjusted bounding box dimensions: {width}x{height} ratio: {round(width / height, 2)}")
#     min_x = int(max(0, min_x))
#     min_y = int(max(0, min_y))
#     max_x = int(min(img.shape[1], max_x))
#     max_y = int(min(img.shape[0], max_y))
#     # if debug:
#     #     print(f"Final bounding box: {min_x}x{min_y} to {max_x}x{max_y}")
#     return min_x, min_y, max_x, max_y


def process_image(ac: ApproachContext, detector, original: any, debug=False, profile=False) -> ProcessedImage:
    tStart = time.perf_counter()
    tX = time.perf_counter()
    img_colour = original  #.copy()
    out = ProcessedImage()
    Eye_det2 = EyeDetector2(show_processing=False)
    #raise Exception("Not implemented")
    tX = time.perf_counter()
    cvt = cv2.cvtColor(img_colour, cv2.COLOR_BGR2RGB)
    if (profile):
        print(
            f"\tTime to process frame - convert: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - tStart) * 1000}")


    Head_pose = HeadPoseEst(show_axis=True)
    frame_size = img_colour.shape[1], img_colour.shape[0]

    tX = time.perf_counter()


    lms = get_landmarks(detector, cvt, debug)
    if lms is not None:
        if (profile):
            print(
                f"\tTime to process frame - get_landmarks: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - tStart) * 1000}")

        tX = time.perf_counter()
        landmarks = _get_landmarks(lms)
        landmarks2 = lms[0].landmark

        # Draws head pose, useful
        Eye_det2.get_Gaze_Score(frame=img_colour, landmarks=landmarks, frame_size=frame_size)

        frame_det, roll, pitch, yaw = Head_pose.get_pose(frame=img_colour, landmarks=landmarks, frame_size=frame_size)
        out.roll = roll
        out.pitch = pitch
        out.yaw = yaw

        if (profile):
            print(
                f"\tTime to process frame - _get_landmarks: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - tStart) * 1000}")

        # if debug:
        #     tX = time.perf_counter()
        #
        #     landmarks_to_use_bounding_box = frame_right_eye_landmarks if ac.gc.flip_eye_mode else frame_left_eye_landmarks
        #
        #     min_x, min_y, max_x, max_y = calculate_bounding_box(landmarks2, img_colour, landmarks_to_use_bounding_box, debug)
        #     min_x, min_y, max_x, max_y = adjust_bounding_box(min_x, min_y, max_x, max_y, img_colour, debug)
        #
        #     if (profile):
        #         print(
        #             f"\tTime to process frame - calc and adjust bounding box: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - tStart) * 1000}")
        #
        #     annotated = img_colour  #.copy()
        #     #Eye_det2.show_eye_keypoints(color_frame=annotated, landmarks=landmarks)

        tX = time.perf_counter()

        ear, ear_frame_left, ear_frame_right = Eye_det2.get_EAR(landmarks=landmarks)
        # ear, ear_right, ear_left = Eye_det2.get_EAR(landmarks=landmarks)

        if (profile):
            print(
                f"\tTime to process frame - EAR: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - tStart) * 1000}")

        tX = time.perf_counter()

        out.ear = ear
        out.ear_left = ear_frame_left if not ac.gc.flip_eye_mode else ear_frame_right
        out.ear_right = ear_frame_right if not ac.gc.flip_eye_mode else ear_frame_left


        # if debug:
        #     eye_img = annotated[min_y:max_y, min_x:max_x]
        #     if eye_img is not None:
        #         eye_img = cv2.resize(eye_img, (tc.EYE_IMAGE_WIDTH, tc.EYE_IMAGE_HEIGHT))
        #         out.eye_img_final = eye_img
        #     out.eye_img_orig = original[min_y:max_y, min_x:max_x]

        if (profile):
            print(
                f"\tTime to process frame - image: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - tStart) * 1000}")

        if (profile):
            print(f"Time to process frame: {(time.perf_counter() - tStart) * 1000}")
        return out


# def analyse_images(ac: ApproachContext, images: List[ProcessedImage], latest: ProcessedImage, profile=False) -> AnalysedImage:
#     tX = time.perf_counter()
#     avg_ear_left = ac.avg_ear_left()
#
#     highest_diff: AnalysedImage = None
#     for i in range(0, len(images)):
#         pi = images[i]
#
#         ear_left_diff = latest.ear_left - pi.ear_left
#         ear_left_diff_ratio = abs(ear_left_diff) / avg_ear_left
#
#         if highest_diff is None or ear_left_diff_ratio > highest_diff.ear_left_diff_ratio:
#             highest_diff = AnalysedImage(latest, avg_ear_left, pi)
#
#     if (profile):
#         print(f"Time to analyse images: {(time.perf_counter() - tX) * 1000}")
#     return highest_diff
#
#
def cram_homogenous_images(images: List[any], output_x, image_selector, image_annotator):
    if len(images) > 0:
        first = image_selector(images[0])
        if first is not None:
            img_width = first.shape[1]
            img_height = first.shape[0]

            num_cols = int(output_x / img_width)
            num_rows = (len(images) // num_cols) + 1
            output_y = num_rows * img_height

            out = np.zeros((output_y, output_x, 3), dtype=np.uint8)

            for i in range(min(num_rows * num_cols, len(images))):
                row = i // num_cols
                col = i % num_cols
                x = col * img_width
                y = row * img_height

                # Use the image_selector function to select the image
                img = image_selector(images[i])
                if img is not None:
                    selected_image = img.copy()

                    # Use the image_annotator function to annotate the image
                    annotated_image = image_annotator(images[i], selected_image, i)

                    # Resize and place the annotated image in the output image
                    out[y:y + img_height, x:x + img_width] = cv2.resize(annotated_image, (img_width, img_height))

            return out
    return None


# def process_and_analyse_frames(images: list[any]) -> list[AnalysedImage]:
#     processed = process_frames(images)
#     return analyse_images(processed)


# def image_annotator(ai: AnalysedImage, img, idx: int):
#     colour = (255, 255, 255)
#     if (ai.ear_left_diff_ratio > 0.5):
#         if (ai.ear_left_diff < 0):
#             colour = (0, 0, 255)
#         else:
#             colour = (0, 255, 0)
#     cv2.putText(img,
#                 f"{round(ai.processed.ear_left * 100)} {round(ai.ear_left_diff * 100)} {round(ai.ear_left_diff_ratio, 1)}",
#                 (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1)
#     return img

    self.ear_left_avg = ear_left_avg
    self.ear_left_median = ear_left_median
    self.frames_eye_visible = frames_eye_visible
    self.frames_good_pitch = frames_good_pitch
    self.frames_ear_left_below_threshold = frames_ear_left_below_threshold

def write_to_influx(gc: GlobalContext, ac: ApproachContext, finished: EpochFinished):
    try:
        # Names do go on the wire but take minimal space in db
        value = f"""fatigue,host="{gc.hostname}",vers="4V1" """

        value += f"epochFrames={finished.length_frames()}"
        value += f",framesEyeVisible={finished.frames_eye_visible}"
        value += f",framesGoodPitch={finished.frames_good_pitch}"
        value += f",framesLeftEyeBelowThreshold={finished.frames_ear_left_below_threshold}"
        value += f",earLeftAverage={finished.ear_left_avg}"
        value += f",earLeftMedian={finished.ear_left_median}"

        value += f",fpsCapture={round(gc.capture_fps)}"
        value += f",fpsProcess={round(gc.process_fps)}"
        value += f",queue={round(gc.frame_queue_for_processing.qsize())}"

        value += f" {int(time.time())}"
        print(f"Writing data to InfluxDB: {value}")
        gc.influx_client.write([value], write_precision='s')
    except Exception as e:
        # Improved error message
        error_type = type(e).__name__
        print(f"Failed to write data to InfluxDB due to {error_type}: {e}")
        print(
            "Please check your InfluxDB configurations, network connection, and ensure the InfluxDB service is running.")


def handle_image(ac: ApproachContext, processed, text_list: [str], frame_idx: int):
    if ac.epoch is None:
        ac.epoch = EpochOngoing(frame_idx)

    if processed is not None:
        img: ProcessedImage = process_image(ac, ac.detector, processed, debug=ac.gc.debug_mode, profile=ac.gc.print_timings)
        if img is not None:
            ac.epoch.add(img.ear_left, img.pitch)

            good_pitch = img.pitch >= ac.params.good_pitch_threshold
            left_eye_below_threshold = img.ear_left < ac.params.ear_left_threshold_for_closed

            text_list.append(f"Roll: {img.roll} Pitch: {img.pitch} (good={good_pitch}) Yaw: {img.yaw}")
            text_list.append(f"Frame idx: {frame_idx}")
            text_list.append(f"Ear left: {img.ear_left} (below = {left_eye_below_threshold})")

    if (frame_idx - ac.epoch.start_frame_idx) + 1 >= ac.params.epoch_time_frames:
        finished = ac.epoch.finish(frame_idx, ac.params.ear_left_threshold_for_closed, ac.params.good_pitch_threshold)
        write_to_influx(ac.gc, ac, finished)
        #ac.finished_epochs.append(finished)
        #print(f"Finished epoch:\n\t{ac.epoch}\n\t{finished}")
        print(finished)
        ac.epoch = None


# def handle_image_for_training(ac: ApproachContext, processed, all_images: List[ImageAndFilename], current_index: int, text_list: [str]):
#     if processed is not None:
#         img: ProcessedImage = process_image(ac, ac.detector, processed, debug=ac.gc.debug_mode, profile=ac.gc.print_timings)
#         if img is not None and current_index > ac.params.frames_lookback
#             images = all_images[current_index - ac.params.frames_lookback:current_index - 1]
#
#             latest: AnalysedImage = analyse_images(ac, ac.rolling_buffer[-ac.params.frames_lookback:-1], img)
#             a = ac.blink_recorder.record(ac, latest)
#
#             if ac.gc.debug_mode:
#                 # if True:
#                 ac.rolling_buffer_for_debug.append(a)
#
#             text_list.append(f"Avg ear left: {ac.avg_ear_left()}")
#             text_list.append(f"Ear left: {latest.processed.ear_left}")
#             text_list.append(f"Prev ear left: {latest.prev_ear_left}")
#             text_list.append(f"Ear left diff: {latest.ear_left_diff}")
#             text_list.append(f"Ear left diff ratio: {latest.ear_left_diff_ratio}")
#             text_list.append(f"Currently blinking: {a.bc.currently_blinking}")
#             text_list.append(f"Blinks in last {ac.blink_recorder.period_seconds}s: {a.bc.blinks_in_last_period}")
#             text_list.append(f"Blinks ever: {a.bc.blinks_total}")


# def select_image2(x: AnalysedImageAndTimeAndContext) -> np.ndarray:
#     if x.ai.ai.processed.eye_img_final is None:
#         return None
#
#     resized_orig = cv2.resize(x.ai.ai.processed.eye_img_orig, (tc.EYE_IMAGE_WIDTH, tc.EYE_IMAGE_HEIGHT))
#     # Create a black image of the same size as the original image
#     black_image = np.zeros_like(resized_orig)
#
#     # Stack the original image and the black image vertically
#     out = np.vstack((black_image, x.ai.ai.processed.eye_img_final, resized_orig))
#
#     return out
#
# def image_annotator2(ai: AnalysedImageAndTimeAndContext, img, idx: int):
#     colour = (255, 255, 255)
#
#     if (ai.ai.ai.ear_left_diff_ratio > 0.5):
#         if (ai.ai.ai.ear_left_diff < 0):
#             colour = (0, 0, 255)
#         else:
#             colour = (0, 255, 0)
#     cv2.putText(img,
#                 f"{round(ai.ai.ai.processed.ear_left * 100)} {round(ai.ai.ai.ear_left_diff * 100)} {round(ai.ai.ai.ear_left_diff_ratio, 1)}",
#                 (1, 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
#     cv2.putText(img,
#                 f"b={'T' if ai.bc.currently_blinking() else 'F'} blinks={ai.bc.blinks_total}",
#                 (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
#     # cv2.putText(img,
#     #             f"{ai.ai.timestamp.strftime('%H:%M:%S.%f')[:-3]}-{idx}",
#     #             (1, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
#     return img


def handle_new_second(ac: ApproachContext, current_time, frame_idx):
    # handle_new_minute(ac, current_time, frame_idx)
    pass

def handle_new_minute(ac: ApproachContext, current_time, frame_idx):
    pass
    # print(f"New minute {current_time} ${ac.gc.debug_mode} ${len(ac.rolling_buffer_for_debug)}")
    # #if ac.gc.debug_mode:
    # if True:
    #     debug_draw = cram_homogenous_images(ac.rolling_buffer_for_debug, 1000,
    #                                         select_image2,
    #                                         image_annotator2)
    #
    #
    #     if debug_draw is not None:
    #         output_dir = "output_images"
    #         try:
    #             os.mkdir(output_dir)
    #         except FileExistsError:
    #             pass
    #         timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    #         filename1 = f"{output_dir}/{timestamp}-proc.jpg"
    #         print(f"Writing {filename1}")
    #         cv2.imwrite(filename1, debug_draw)
    #     else:
    #         print("debug draw was None")
    #
    # ac.rolling_buffer_for_debug = []
