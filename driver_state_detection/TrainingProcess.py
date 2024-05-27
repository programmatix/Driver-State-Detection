import cv2
import mediapipe as mp


def process_frames(images):
    detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)
    processed_images = []
    for filename, img in images:
        results = detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            left_eye = landmarks[33]
            x, y = int(left_eye.x * img.shape[1]), int(left_eye.y * img.shape[0])
            eye_img = img[max(0, y - 25):min(img.shape[0], y + 25), max(0, x - 25):min(img.shape[1], x + 25)]

            # Calculate the new height based on the original aspect ratio
            h, w = eye_img.shape[:2]
            new_w = 100
            new_h = int(h * new_w / w)

            print(f"Resizing image from {w}x{h} to {new_w}x{new_h}")
            # Resize the image
            eye_img = cv2.resize(eye_img, (new_w, new_h))

            processed_images.append((filename, eye_img))
    return processed_images
