import cv2
import mediapipe as mp
import numpy as np

import TrainingConstants as tc

def process_frames(images):
    detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
    processed_images = []
    image_idx = 0
    debug = False
    for filename, img in images:
        #debug = image_idx == 1
        processed_image = process_image(detector, filename, img, image_idx, debug)
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
        return results.multi_face_landmarks[0].landmark
    else:
        return None

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

def calculate_center_of_iris(landmarks, img, iris_landmark, debug):
    center_x = int(landmarks[iris_landmark].x * img.shape[1])
    center_y = int(landmarks[iris_landmark].y * img.shape[0])
    if debug:
        print(f"Center of iris: {center_x}x{center_y}")
    return center_x, center_y

def adjust_bounding_box(min_x, min_y, max_x, max_y, img, center_x, center_y, debug):

    width = max_x - min_x
    height = max_y - min_y
    center_x = min_x + width // 2
    center_y = min_y + height // 2
    if debug:
        print(f"Bounding box dimensions: {min_x}x{min_y} to {max_x}x{max_y} {width}x{height} ratio: {round(width / height, 2)}, want to centre on {center_x}x{center_y}")
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

def process_image(detector, filename, img, image_idx, debug=False):
    #raise Exception("Not implemented")
    landmarks = get_landmarks(detector, img, debug)
    if landmarks is not None:
        min_x, min_y, max_x, max_y = calculate_bounding_box(landmarks, img, left_eye_landmarks, debug)
        center_x, center_y = calculate_center_of_iris(landmarks, img, 468, debug)
        min_x, min_y, max_x, max_y = adjust_bounding_box(min_x, min_y, max_x, max_y, img, center_x, center_y, debug)

        pixel = None

        # Just normalize the eye portion
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # pupil_x = int(landmarks[468].x * gray.shape[1])
        # pupil_y = int(landmarks[468].y * gray.shape[0])
        # pixel = gray[pupil_y, pupil_x]
        # eye_img = gray[min_y:max_y, min_x:max_x]
        # eye_img = cv2.normalize(eye_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # img[min_y:max_y, min_x:max_x] = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2BGR)

        for i in left_eye_iris_landmarks:
            x = int(landmarks[i].x * img.shape[1])
            y = int(landmarks[i].y * img.shape[0])
            if i == 0:
                pixel = img[y, x]
            cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

        mask = np.zeros_like(img)
        points = np.array([(landmarks[i].x * img.shape[1], landmarks[i].y * img.shape[0]) for i in left_eye_landmarks], np.int32)
        ellipse = cv2.fitEllipse(points)
        cv2.ellipse(mask, ellipse, (255, 255, 255), -1)
        masked = cv2.bitwise_and(img, mask)


        eye_img = masked[min_y:max_y, min_x:max_x]
        eye_img = cv2.resize(eye_img, (tc.EYE_IMAGE_WIDTH, tc.EYE_IMAGE_HEIGHT))

        # Not gray at this point because may have colour debug info

        return (filename, eye_img, pixel)