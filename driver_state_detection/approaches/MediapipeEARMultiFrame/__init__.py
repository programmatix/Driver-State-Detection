# Expanding on the previous EAR single-frame approach, to consider EAR of previous frames too.
import socket
import time
from datetime import datetime
from glob import glob

import cv2
import mediapipe as mp
import numpy as np
from Eye_Dector_Module import EyeDetector as EyeDet2
import TrainingConstants as tc

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


class ProcessedImage:
    # All images are in colour.  Easier to work with, and can put debug info in them.
    eye_img_orig = None
    eye_img_final = None
    eye_img_steps = []
    original_image: any = None
    pupil_center_pixel = None
    ear = None
    ear_left = None
    ear_right = None

def process_frames(images: list[any]):
    detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5, refine_landmarks=True)
    processed_images: list[ProcessedImage] = []
    image_idx = 0
    debug = False
    for img in images:
        #debug = image_idx == 1
        processed_image = process_image(detector, img, debug)
        image_idx += 1
        if processed_image is not None:
            processed_images.append(processed_image)
    return processed_images


# https://github.com/tensorflow/tfjs-models/blob/master/face-landmarks-detection/src/tfjs/constants.ts
#left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 468, 246, 161, 160, 159, 158, 157, 173, 474]
left_eye_landmarks = [
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

left_eye_iris_landmarks = [
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


def get_landmarks(detector, img, debug):
    if debug:
        print(f"Processing image with detector")
    # results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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


def calculate_bounding_box(landmarks, img, left_eye_landmarks, debug):
    min_x = max_x = int(landmarks[left_eye_landmarks[0]].x * img.shape[1])
    min_y = max_y = int(landmarks[left_eye_landmarks[0]].y * img.shape[0])
    if debug:
        print(f"Initial bounding box: {min_x}x{min_y} to {max_x}x{max_y}")
    for i in left_eye_landmarks:
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
    if debug:
        print(
            f"Bounding box dimensions: {min_x}x{min_y} to {max_x}x{max_y} {width}x{height} ratio: {round(width / height, 2)}, want to centre on {center_x}x{center_y}")
    if width / height < 3:
        width = height * 3
        min_x = center_x - width // 2
        max_x = center_x + width // 2
    elif height < width / 3:
        height = width / 3
        min_y = center_y - height // 2
        max_y = center_y + height // 2
    if debug:
        print(f"Adjusted bounding box dimensions: {width}x{height} ratio: {round(width / height, 2)}")
    min_x = int(max(0, min_x))
    min_y = int(max(0, min_y))
    max_x = int(min(img.shape[1], max_x))
    max_y = int(min(img.shape[0], max_y))
    if debug:
        print(f"Final bounding box: {min_x}x{min_y} to {max_x}x{max_y}")
    return min_x, min_y, max_x, max_y


def process_image(detector, original: any, debug=False, profile=False) -> ProcessedImage:
    tX = time.perf_counter()
    img_colour = original.copy()
    out = ProcessedImage()
    Eye_det = EyeDet2(show_processing=False)
    #raise Exception("Not implemented")
    lms = get_landmarks(detector, img_colour, debug)
    if lms is not None:
        landmarks = _get_landmarks(lms)
        landmarks2 = lms[0].landmark

        min_x, min_y, max_x, max_y = calculate_bounding_box(landmarks2, img_colour, left_eye_landmarks, debug)
        min_x, min_y, max_x, max_y = adjust_bounding_box(min_x, min_y, max_x, max_y, img_colour, debug)

        annotated = img_colour.copy()
        #Eye_det.show_eye_keypoints(color_frame=annotated, landmarks=landmarks)

        ear, ear_left, ear_right = Eye_det.get_EAR(None, landmarks=landmarks)
        out.ear = ear
        out.ear_left = ear_left
        out.ear_right = ear_right

        eye_img = annotated[min_y:max_y, min_x:max_x]
        eye_img = cv2.resize(eye_img, (tc.EYE_IMAGE_WIDTH, tc.EYE_IMAGE_HEIGHT))

        out.eye_img_final = eye_img

        if (profile):
            print(f"Time to process frame: {(time.perf_counter() - tX) * 1000}")
        return out



class AnalysedImage:
    processed: ProcessedImage
    prev_ear_left: float
    ear_left_diff: float
    ear_left_diff_ratio: float
    avg_ear_left: float

    def __init__(self, processed: ProcessedImage, avg_ear_left: float, prev: ProcessedImage):
        self.processed = processed
        self.avg_ear_left = avg_ear_left
        self.prev_ear_left = prev.ear_left
        self.ear_left_diff = self.processed.ear_left - self.prev_ear_left
        self.ear_left_diff_ratio = abs(self.ear_left_diff) / self.avg_ear_left

def analyse_images(images: list[ProcessedImage], N: int, profile=False) -> list[AnalysedImage]:
    tX = time.perf_counter()
    avg_ear_left = sum([pi.ear_left for pi in images]) / len(images)
    out: list[AnalysedImage] = []

    for i in range(1, len(images)):
        pi = images[i]
        highest_diff: AnalysedImage = None
        for n in range(max(0, i - N), min(len(images), i)):
            prev = images[n]
            ai = AnalysedImage(pi, avg_ear_left, prev)
            if highest_diff is None or ai.ear_left_diff_ratio > highest_diff.ear_left_diff_ratio:
                highest_diff = ai
        if highest_diff is not None:
            out.append(highest_diff)

    if (profile):
        print(f"Time to analyse images: {(time.perf_counter() - tX) * 1000}")
    return out

def cram_homogenous_images(images: list[AnalysedImage], output_x, image_selector, image_annotator):
    img_width = images[0].processed.eye_img_final.shape[1]
    img_height = images[0].processed.eye_img_final.shape[0]

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
        selected_image = image_selector(images[i]).copy()

        # Use the image_annotator function to annotate the image
        annotated_image = image_annotator(images[i], selected_image, i)

        # Resize and place the annotated image in the output image
        out[y:y + img_height, x:x + img_width] = cv2.resize(annotated_image, (img_width, img_height))

    return out

def process_and_analyse_frames(images: list[any]) -> list[AnalysedImage]:
    processed = process_frames(images)
    return analyse_images(processed)


def image_annotator(ai: AnalysedImage, img, idx: int):
    colour = (255, 255, 255)
    if (ai.ear_left_diff_ratio > 0.5):
        if (ai.ear_left_diff < 0):
            colour = (0, 0, 255)
        else:
            colour = (0, 255, 0)
    cv2.putText(img, f"{round(ai.processed.ear_left * 100)} {round(ai.ear_left_diff * 100)} {round(ai.ear_left_diff_ratio,1)}", (1, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return img

class AnalysedImageAndTime:
    ai: AnalysedImage

    def __init__(self, ai: AnalysedImage, timestamp):
        self.ai = ai
        self.timestamp = timestamp

class BlinkRecorder:
    blinks_total = 0
    _blinks_in_last_period: list[AnalysedImageAndTime] = []
    currently_blinking = False

    def __init__(self, period_seconds):
        self.period_seconds = period_seconds
        self.hostname = socket.gethostname()

    def record(self, latest: AnalysedImage):
        # Only record blink starts currently
        current_time = datetime.now()
        is_blinking = latest.ear_left_diff_ratio > 0.5 and latest.ear_left_diff < 0
        self._blinks_in_last_period = [b for b in self._blinks_in_last_period if (current_time - b.timestamp).seconds <= self.period_seconds]
        if is_blinking and not self.currently_blinking:
            self.currently_blinking = True
            self.blinks_total += 1
            self._blinks_in_last_period.append(AnalysedImageAndTime(latest, current_time))
            print(f"{current_time} Blink start, total blinks {self.blinks_total}, blinks in last period {len(self._blinks_in_last_period)}")
        if self.currently_blinking and not is_blinking:
            print(f"{current_time} Blink end, total blinks {self.blinks_total}, blinks in last period {len(self._blinks_in_last_period)}")
            self.currently_blinking = False
        return (self.currently_blinking, len(self._blinks_in_last_period), self.blinks_total)
