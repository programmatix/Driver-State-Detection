# Expanding on the previous EAR single-frame approach, to consider EAR of previous frames too.
import os
import time
from glob import glob
from numpy import linalg as LA

import cv2
import mediapipe as mp
import numpy as np
from typing import List
from typing_extensions import List

# import Eye_Dector_Module2
import approaches.MediapipeEARMultiFrame.TrainingConstants as tc
from GlobalContext import GlobalContext
from approaches.MediapipeEARMultiFrame import ApproachContext
from approaches.MediapipeEARMultiFrame.Model import ProcessedImage, AnalysedImage, AnalysedImageAndTimeAndContext, \
    BlinkContext, AnalysedImageAndTime, ImageAndFilename

from Pose_Estimation_Module import HeadPoseEstimator as HeadPoseEst

# This has both left and right eyes
EYES_LMS_NUMS = [33, 133, 160, 144, 158, 153, 362, 263, 385, 380, 387, 373]

LEFT_IRIS_NUM = 468
RIGHT_IRIS_NUM = 473


class EyeDetector2:

    def __init__(self, show_processing: bool = False):
        """
        Eye dector class that contains various method for eye aperture rate estimation and gaze score estimation

        Parameters
        ----------
        show_processing: bool
            If set to True, shows frame images during the processing in some steps (default is False)

        Methods
        ----------
        - show_eye_keypoints: shows eye keypoints in the frame/image
        - get_EAR: computes EAR average score for the two eyes of the face
        - get_Gaze_Score: computes the Gaze_Score (normalized euclidean distance between center of eye and pupil)
            of the eyes of the face
        """

        self.show_processing = show_processing

    @staticmethod
    def _calc_EAR_eye(eye_pts):
        """
        Computer the EAR score for a single eyes given it's keypoints
        :param eye_pts: numpy array of shape (6,2) containing the keypoints of an eye
        :return: ear_eye
            EAR of the eye
        """
        ear_eye = (LA.norm(eye_pts[2] - eye_pts[3]) + LA.norm(
            eye_pts[4] - eye_pts[5])) / (2 * LA.norm(eye_pts[0] - eye_pts[1]))
        '''
        EAR is computed as the mean of two measures of eye opening (see mediapipe face keypoints for the eye)
        divided by the eye lenght
        '''
        return ear_eye

    @staticmethod
    def _calc_1eye_score(landmarks, eye_lms_nums, eye_iris_num, frame_size, frame):
        """Gets each eye score and its picture."""
        iris = landmarks[eye_iris_num, :2]

        eye_x_min = landmarks[eye_lms_nums, 0].min()
        eye_y_min = landmarks[eye_lms_nums, 1].min()
        eye_x_max = landmarks[eye_lms_nums, 0].max()
        eye_y_max = landmarks[eye_lms_nums, 1].max()

        eye_center = np.array(((eye_x_min+eye_x_max)/2,
                               (eye_y_min+eye_y_max)/2))

        eye_gaze_score = LA.norm(iris - eye_center) / eye_center[0]

        eye_x_min_frame = int(eye_x_min * frame_size[0])
        eye_y_min_frame = int(eye_y_min * frame_size[1])
        eye_x_max_frame = int(eye_x_max * frame_size[0])
        eye_y_max_frame = int(eye_y_max * frame_size[1])

        eye = frame[eye_y_min_frame:eye_y_max_frame,
              eye_x_min_frame:eye_x_max_frame]

        # Draw each item on the frame
        for i in eye_lms_nums:
            x = int(landmarks[i, 0] * frame_size[0])
            y = int(landmarks[i, 1] * frame_size[1])
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Draw the eye_gaze_score below the eye
        cv2.putText(frame, f"Gaze Score: {eye_gaze_score}", (eye_x_min_frame, eye_y_max_frame + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return eye_gaze_score, eye

    def get_Gaze_Score(self, frame, landmarks, frame_size):
        """
        Computes the average Gaze Score for the eyes
        The Gaze Score is the mean of the l2 norm (euclidean distance) between the center point of the Eye ROI
        (eye bounding box) and the center of the eye-pupil

        Parameters
        ----------
        frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: numpy array
            List of 478 face mesh keypoints of the face

        Returns
        --------
        avg_gaze_score: float
            If successful, returns the float gaze score
            If unsuccessful, returns None

        """

        left_gaze_score, left_eye = self._calc_1eye_score(
            landmarks, EYES_LMS_NUMS[:6], LEFT_IRIS_NUM, frame_size, frame)
        right_gaze_score, right_eye = self._calc_1eye_score(
            landmarks, EYES_LMS_NUMS[6:], RIGHT_IRIS_NUM, frame_size, frame)

        # if show_processing is True, shows the eyes ROI, eye center, pupil center and line distance

        # computes the average gaze score for the 2 eyes
        avg_gaze_score = (left_gaze_score + right_gaze_score) / 2

        # if self.show_processing and (left_eye is not None) and (right_eye is not None):
        #     left_eye = resize(left_eye, 1000)
        #     right_eye = resize(right_eye, 1000)
        #     cv2.imshow("left eye", left_eye)
        #     cv2.imshow("right eye", right_eye)

        return avg_gaze_score

    def show_eye_keypoints(self, color_frame, landmarks):
        """
        Shows eyes keypoints found in the face, drawing red circles in their position in the frame/image

        Parameters
        ----------
        color_frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: landmarks: numpy array
            List of 478 mediapipe keypoints of the face
        """

        # cv2.circle(color_frame, (landmarks[LEFT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
        #            3, (255, 255, 255), cv2.FILLED)
        # cv2.circle(color_frame, (landmarks[RIGHT_IRIS_NUM, :2] * frame_size).astype(np.uint32),
        #            3, (255, 255, 255), cv2.FILLED)

        frame_size = color_frame.shape[1], color_frame.shape[0]
        for n in EYES_LMS_NUMS:
            x = int(landmarks[n, 0] * frame_size[0])
            y = int(landmarks[n, 1] * frame_size[1])
            cv2.circle(color_frame, (x, y), 1, (0, 255, 0), -1)
        return

    def get_EAR(self, landmarks):
        """
        Computes the average eye aperture rate of the face

        Parameters
        ----------
        frame: numpy array
            Frame/image in which the eyes keypoints are found
        landmarks: landmarks: numpy array
            List of 478 mediapipe keypoints of the face

        Returns
        --------
        ear_score: float
            EAR average score between the two eyes
            The EAR or Eye Aspect Ratio is computed as the eye opennes divided by the eye lenght
            Each eye has his scores and the two scores are averaged
        """

        # numpy array for storing the keypoints positions of the left and right eyes
        eye_pts_l = np.zeros(shape=(6, 2))
        eye_pts_r = eye_pts_l.copy()

        # get the face mesh keypoints
        for i in range(len(EYES_LMS_NUMS) // 2):
            # array of x,y coordinates for the left eye reference point
            eye_pts_l[i] = landmarks[EYES_LMS_NUMS[i], :2]
            # array of x,y coordinates for the right eye reference point
            eye_pts_r[i] = landmarks[EYES_LMS_NUMS[i + 6], :2]

        # Frame left and right
        ear_frame_left = self._calc_EAR_eye(eye_pts_l)  # computing the left eye EAR score
        ear_frame_right = self._calc_EAR_eye(eye_pts_r)  # computing the right eye EAR score

        # computing the average EAR score
        ear_avg = (ear_frame_left + ear_frame_right) / 2

        return ear_avg, ear_frame_left, ear_frame_right


class OriginalImage:
    def __init__(self, filename, original_image):
        self.filename = filename
        self.original_image = original_image


def load_images(folder, max=None) -> list[OriginalImage]:
    images: list[OriginalImage] = []
    print(f'Loading from {folder}')
    count = 0
    # Primary sort on timestamp, secondary sort on frame number
    # Filename: 2024-05-27_09-10-39-orig-97.jpg
    for filename in sorted(glob(folder.replace("\\", "/") + '/*.jpg'),
                           key=lambda f: (f.split('-orig-')[0], int(f.replace(".jpg", "").split('-')[-1]))):
        if max is not None and count > max:
            break
        print(f'Loading {filename}')
        img = cv2.imread(filename)
        count += 1

        if img is not None:
            images.append(OriginalImage(filename, img))
    print(f"Loaded {len(images)} images")
    return images


# def process_frames(images: list[any]):
#     detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5,
#                                                min_tracking_confidence=0.5, refine_landmarks=True)
#     processed_images: list[ProcessedImage] = []
#     image_idx = 0
#     debug = False
#     for img in images:
#         #debug = image_idx == 1
#         processed_image = process_image(detector, img, debug)
#         image_idx += 1
#         if processed_image is not None:
#             processed_images.append(processed_image)
#     return processed_images


# https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/src/tfjs/constants.ts
#left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 468, 246, 161, 160, 159, 158, 157, 173, 474]
frame_left_eye_landmarks = [
    # Lower contour.
    # 33, 7, 163, 144, 145, 153, 154, 155, 133,
    # upper contour (excluding corners).
    # 246, 161, 160, 159, 158, 157, 173,
    # Halo x2 lower contour.
    130, 25, 110, 24, 23, 22, 26, 112, 243,
    # Halo x2 upper contour (excluding corners).
    247, 30, 29, 27, 28, 56, 190,
    # Halo x3 lower contour.
    # 226, 31, 228, 229, 230, 231, 232, 233, 244,
    # Halo x3 upper contour (excluding corners).
    # 113, 225, 224, 223, 222, 221, 189,
    # Halo x4 upper contour (no lower because of mesh structure) or
    # eyebrow inner contour.
    # 35, 124, 46, 53, 52, 65,
    # Halo x5 lower contour.
    # 143, 111, 117, 118, 119, 120, 121, 128, 245,
    # Halo x5 upper contour (excluding corners) or eyebrow outer contour.
    # 156, 70, 63, 105, 66, 107, 55, 193
]

frame_left_eye_iris_landmarks = [
    # Center.
    468,
    # Iris right edge.
    469,
    # Iris top edge.
    470,
    # Iris left edge.
    471,
    # Iris bottom edge.
    472
]

frame_right_eye_landmarks = [
    # Lower contour
    # 263, 249, 390, 373, 374, 380, 381, 382, 362,
    # Upper contour (excluding corners).
    # 466, 388, 387, 386, 385, 384, 398,
    # Halo x2 lower contour.
    359, 255, 339, 254, 253, 252, 256, 341, 463,
    # Halo x2 upper contour (excluding corners).
    467, 260, 259, 257, 258, 286, 414,
    # Halo x3 lower contour.
    # 446, 261, 448, 449, 450, 451, 452, 453, 464,
    # # Halo x3 upper contour (excluding corners).
    # 342, 445, 444, 443, 442, 441, 413,
    # # Halo x4 upper contour (no lower because of mesh structure) or
    # # eyebrow inner contour.
    # 265, 353, 276, 283, 282, 295,
    # # Halo x5 lower contour.
    # 372, 340, 346, 347, 348, 349, 350, 357, 465,
    # # Halo x5 upper contour (excluding corners) or eyebrow outer contour.
    # 383, 300, 293, 334, 296, 336, 285, 417
]

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


def calculate_bounding_box(landmarks, img, eye_landmarks, debug):
    min_x = max_x = int(landmarks[eye_landmarks[0]].x * img.shape[1])
    min_y = max_y = int(landmarks[eye_landmarks[0]].y * img.shape[0])
    # if debug:
    #     print(f"Initial bounding box: {min_x}x{min_y} to {max_x}x{max_y}")
    for i in eye_landmarks:
        x = int(landmarks[i].x * img.shape[1])
        y = int(landmarks[i].y * img.shape[0])
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        if debug:
            # print(f"Landmark {i} ({eye_landmark_names[i]}) at {x}x{y} (min {min_x}x{min_y}, max {max_x}x{max_y})")
            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
    return min_x, min_y, max_x, max_y


def adjust_bounding_box(min_x, min_y, max_x, max_y, img, debug):
    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + width // 2
    center_y = min_y + height // 2
    # if debug:
    #     print(
    #         f"Bounding box dimensions: {min_x}x{min_y} to {max_x}x{max_y} {width}x{height} ratio: {round(width / height, 2)}, want to centre on {center_x}x{center_y}")
    if width / height < 3:
        width = height * 3
        min_x = center_x - width // 2
        max_x = center_x + width // 2
    elif height < width / 3:
        height = width / 3
        min_y = center_y - height // 2
        max_y = center_y + height // 2
    # if debug:
    #     print(f"Adjusted bounding box dimensions: {width}x{height} ratio: {round(width / height, 2)}")
    min_x = int(max(0, min_x))
    min_y = int(max(0, min_y))
    max_x = int(min(img.shape[1], max_x))
    max_y = int(min(img.shape[0], max_y))
    # if debug:
    #     print(f"Final bounding box: {min_x}x{min_y} to {max_x}x{max_y}")
    return min_x, min_y, max_x, max_y


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

        gaze = Eye_det2.get_Gaze_Score(
            frame=img_colour, landmarks=landmarks, frame_size=frame_size)
        #print(gaze)

        frame_det, roll, pitch, yaw = Head_pose.get_pose(
            frame=img_colour, landmarks=landmarks, frame_size=frame_size)
        out.roll = roll
        out.pitch = pitch
        out.yaw = yaw

        if (profile):
            print(
                f"\tTime to process frame - _get_landmarks: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - tStart) * 1000}")

        if debug:
            tX = time.perf_counter()

            landmarks_to_use_bounding_box = frame_right_eye_landmarks if ac.gc.flip_eye_mode else frame_left_eye_landmarks

            min_x, min_y, max_x, max_y = calculate_bounding_box(landmarks2, img_colour, landmarks_to_use_bounding_box, debug)
            min_x, min_y, max_x, max_y = adjust_bounding_box(min_x, min_y, max_x, max_y, img_colour, debug)

            if (profile):
                print(
                    f"\tTime to process frame - calc and adjust bounding box: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - tStart) * 1000}")

            annotated = img_colour  #.copy()
            #Eye_det2.show_eye_keypoints(color_frame=annotated, landmarks=landmarks)

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

        if out.ear_left is not None:
            ac.total_ear_left += out.ear_left
            ac.ear_left_count += 1

        #print(f"EAR frame left: {ear_frame_left} EAR frame right: {ear_frame_right} EAR left: {out.ear_left} EAR right: {out.ear_right}")

        if debug:
            eye_img = annotated[min_y:max_y, min_x:max_x]
            if eye_img is not None:
                eye_img = cv2.resize(eye_img, (tc.EYE_IMAGE_WIDTH, tc.EYE_IMAGE_HEIGHT))
                out.eye_img_final = eye_img
            out.eye_img_orig = original[min_y:max_y, min_x:max_x]

        if (profile):
            print(
                f"\tTime to process frame - image: {(time.perf_counter() - tX) * 1000} {(time.perf_counter() - tStart) * 1000}")

        if (profile):
            print(f"Time to process frame: {(time.perf_counter() - tStart) * 1000}")
        return out


def analyse_images(ac: ApproachContext, images: List[ProcessedImage], latest: ProcessedImage, profile=False) -> AnalysedImage:
    tX = time.perf_counter()
    avg_ear_left = ac.avg_ear_left()

    highest_diff: AnalysedImage = None
    for i in range(0, len(images)):
        pi = images[i]

        ear_left_diff = latest.ear_left - pi.ear_left
        ear_left_diff_ratio = abs(ear_left_diff) / avg_ear_left

        if highest_diff is None or ear_left_diff_ratio > highest_diff.ear_left_diff_ratio:
            highest_diff = AnalysedImage(latest, avg_ear_left, pi)

    if (profile):
        print(f"Time to analyse images: {(time.perf_counter() - tX) * 1000}")
    return highest_diff


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


def image_annotator(ai: AnalysedImage, img, idx: int):
    colour = (255, 255, 255)
    if (ai.ear_left_diff_ratio > 0.5):
        if (ai.ear_left_diff < 0):
            colour = (0, 0, 255)
        else:
            colour = (0, 255, 0)
    cv2.putText(img,
                f"{round(ai.processed.ear_left * 100)} {round(ai.ear_left_diff * 100)} {round(ai.ear_left_diff_ratio, 1)}",
                (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colour, 1)
    return img


def write_to_influx(gc: GlobalContext, ac: ApproachContext):
    try:
        # Names do go on the wire but take minimal space in db
        value = f"""fatigue,host="{gc.hostname}",blinksVers="3V1",medianBlinkDurationFramesVers="3V1" blinks={len(ac.blink_recorder._blinks_in_last_period)}"""
        md = ac.blink_recorder.get_median_blink_duration()
        if md is not None:
            value += f",medianBlinkDurationFrames={md}"

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


def handle_image(ac: ApproachContext, processed, text_list: [str], frame_idx: int) -> AnalysedImageAndTimeAndContext:
    recorded_blink_frame = False

    if processed is not None:
        img: ProcessedImage = process_image(ac, ac.detector, processed, debug=ac.gc.debug_mode, profile=ac.gc.print_timings)
        if img is not None:
            img.frame_idx = frame_idx
            ac.rolling_buffer.append(img)

            if len(ac.rolling_buffer) > ac.params.frames_lookback:
                while (len(ac.rolling_buffer) > ac.params.frames_lookback):
                    ac.rolling_buffer.pop(0)

                latest: AnalysedImage = analyse_images(ac, ac.rolling_buffer, img,  profile=ac.gc.print_timings)
                a = ac.blink_recorder.record(ac, latest, frame_idx)
                recorded_blink_frame = True

                if ac.gc.debug_mode:
                # if True:
                    ac.rolling_buffer_for_debug.append(a)

                text_list.append(f"Roll: {latest.processed.roll} Pitch: {latest.processed.pitch} Yaw: {latest.processed.yaw}")
                text_list.append(f"Frame idx: {frame_idx}")
                text_list.append(f"Prev frame idx: {latest.prev.frame_idx}")
                text_list.append(f"Avg ear left: {ac.avg_ear_left()}")
                text_list.append(f"Ear left: {latest.processed.ear_left}")
                text_list.append(f"Prev ear left: {latest.prev_ear_left}")
                text_list.append(f"Ear left diff: {latest.ear_left_diff}")
                text_list.append(f"Ear left diff ratio: {latest.ear_left_diff_ratio}")
                text_list.append(f"Blink state: {a.bc.blink_state}")
                text_list.append(f"Current blink duration frames: {a.bc.current_blink_duration_frames}")
                text_list.append(f"Blinks in last {ac.blink_recorder.period_frames} frames: {a.bc.blinks_in_last_period}")
                text_list.append(f"Median blink duration in last {ac.blink_recorder.period_frames} frames: {a.bc.median_blink_duration_in_last_period}")
                text_list.append(f"Blinks ever: {a.bc.blinks_total}")

                return a

                # if ac.gc.debug_mode:
                #     image_selector = lambda x: x.processed.eye_img_final
                #     debug_draw = cram_homogenous_images(analysed, 1000, image_selector, image_annotator)
                #
                #     if debug_draw is not None:
                #         processed[0:debug_draw.shape[0], 0:debug_draw.shape[1]] = debug_draw
                #         last_processed = processed.copy()

    if not recorded_blink_frame:
        ac.blink_recorder.record_empty(ac)

    ac.gc.save_with_training_set['avg_ear_left'] = ac.avg_ear_left()

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


def select_image2(x: AnalysedImageAndTimeAndContext) -> np.ndarray:
    if x.ai.ai.processed.eye_img_final is None:
        return None

    resized_orig = cv2.resize(x.ai.ai.processed.eye_img_orig, (tc.EYE_IMAGE_WIDTH, tc.EYE_IMAGE_HEIGHT))
    # Create a black image of the same size as the original image
    black_image = np.zeros_like(resized_orig)

    # Stack the original image and the black image vertically
    out = np.vstack((black_image, x.ai.ai.processed.eye_img_final, resized_orig))

    return out

def image_annotator2(ai: AnalysedImageAndTimeAndContext, img, idx: int):
    colour = (255, 255, 255)

    if (ai.ai.ai.ear_left_diff_ratio > 0.5):
        if (ai.ai.ai.ear_left_diff < 0):
            colour = (0, 0, 255)
        else:
            colour = (0, 255, 0)
    cv2.putText(img,
                f"{round(ai.ai.ai.processed.ear_left * 100)} {round(ai.ai.ai.ear_left_diff * 100)} {round(ai.ai.ai.ear_left_diff_ratio, 1)}",
                (1, 7), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    cv2.putText(img,
                f"b={'T' if ai.bc.currently_blinking() else 'F'} blinks={ai.bc.blinks_total}",
                (1, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    # cv2.putText(img,
    #             f"{ai.ai.timestamp.strftime('%H:%M:%S.%f')[:-3]}-{idx}",
    #             (1, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1)
    return img


def handle_new_second(ac: ApproachContext, current_time, frame_idx):
    # handle_new_minute(ac, current_time, frame_idx)
    pass

def handle_new_minute(ac: ApproachContext, current_time, frame_idx):
    print(f"New minute {current_time} ${ac.gc.debug_mode} ${len(ac.rolling_buffer_for_debug)}")
    #if ac.gc.debug_mode:
    if True:
        debug_draw = cram_homogenous_images(ac.rolling_buffer_for_debug, 1000,
                                            select_image2,
                                            image_annotator2)


        if debug_draw is not None:
            output_dir = "output_images"
            try:
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            filename1 = f"{output_dir}/{timestamp}-proc.jpg"
            print(f"Writing {filename1}")
            cv2.imwrite(filename1, debug_draw)
        else:
            print("debug draw was None")

    ac.rolling_buffer_for_debug = []
