import argparse
import shutil

import cv2
import os
import mediapipe as mp
from glob import glob
import TrainingConstants as tc
from TrainingDisplay import display_frames
from TrainingProcess import process_frames
from tensorflow.keras.models import load_model
from Thresholds import Thresholds

parser = argparse.ArgumentParser(description='Process images in a folder')
parser.add_argument('input', type=str, help='The path to the folder containing the images')
parser.add_argument('model', type=str, help='The path to the saved model')
args = parser.parse_args()


import re

def load_images(folder):
    images = []
    print(f'Loading from {folder}')
    count = 0
    # Primary sort on timestamp, secondary sort on frame number
    # Filename: 2024-05-27_09-10-39-orig-97.jpg
    for filename in sorted(glob(folder.replace("\\", "/") + '/*.jpg'), key=lambda f: (f.split('-orig-')[0], int(f.replace(".jpg", "").split('-')[-1]))):
        if count > 100:
            break
        print(f'Loading {filename}')
        img = cv2.imread(filename)
        count += 1

        if img is not None:
            images.append((filename, img))
    print(f"Loaded {len(images)} images")
    return images

def save_images(images, current_index, output_dir, label):
    start_index = max(0, current_index - 4)

    filename, img = images[current_index]
    dirname, basename = os.path.split(filename)
    print(f"Processing {filename} with dirname {dirname} and basename {basename}")
    outdir = filename.replace("training", "labels").replace("orig\\", "").replace(".jpg", "")
    print(f"Output directory: {outdir}")
    try:
        shutil.rmtree(outdir)
    except FileNotFoundError:
        pass
    os.makedirs(outdir, exist_ok=True)

    for i in range(start_index, current_index + 1):
        filename, img = images[i]

        # Split the filename into directory and base name
        dirname, basename = os.path.split(filename)
        print(f"Processing {filename} with dirname {dirname} and basename {basename}")

        # # Remove the 'orig' part from the directory name
        # dirname = dirname.replace('orig', '')
        # od = os.path.join(output_dir, dirname, basename)
        # print(f"Output directory: {od}")
        # # Create the output directory if it doesn't exist
        # output_dirname = os.path.join(od, dirname)
        # Add "-blink-" only to the current frame's filename
        if i == current_index:
            output_filename = os.path.join(outdir, basename.replace(".jpg", f'-{label}.jpg'))
        else:
            output_filename = os.path.join(outdir, basename)
        print(f"Saving image {output_filename}")
        cv2.imwrite(output_filename, img)


# Use the FaceMesh model to detect faces in each frame and zoom into the left eye
# Handle user input
def handle_input(images, current_index, model):
    thresholds = Thresholds()
    while True:
        print(str(thresholds.__dict__))
        display_frames(images, current_index, model, thresholds)
        key = cv2.waitKey(0)
        if key == ord('a'):  # Left key
            current_index = max(0, current_index - 1)
        elif key == ord('d'):  # Right key
            current_index = min(len(images) - 1, current_index + 1)
        elif key == ord(' '):  # Spacebar
            save_images(images, current_index, "labels", "not-blink")
            current_index = min(len(images) - 1, current_index + 1)
        # elif key == ord('5'):
        #     for i in range(0, 5):
        #         save_images(images, i, "labels", "not-blink")
        #         current_index = min(len(images) - 1, current_index + 1)
        elif key == ord('s'):  # 's' key
            save_images(images, current_index, "labels", "blink")
            current_index = min(len(images) - 1, current_index + 1)

        elif key == ord('2'):
            thresholds.canny_1 = min(255, thresholds.canny_1 + 5)
        elif key == ord('1'):
            thresholds.canny_1 = max(0, thresholds.canny_1 - 5)

        elif key == ord('4'):
            thresholds.canny_2 = min(255, thresholds.canny_2 + 5)
        elif key == ord('3'):
            thresholds.canny_2 = max(0, thresholds.canny_2 - 5)

        elif key == ord('6'):
            thresholds.blur_1 = min(255, thresholds.blur_1 + 2)
        elif key == ord("5"):
            thresholds.blur_1 = max(0, thresholds.blur_1 - 2)

        elif key == ord('8'):
            thresholds.blur_sigma = min(255, thresholds.blur_sigma + 2)
        elif key == ord("7"):
            thresholds.blur_sigma = max(0, thresholds.blur_sigma - 2)

        elif key == ord('0'):
            thresholds.hoare_param1 = min(255, thresholds.hoare_param1 + 1)
        elif key == ord("9"):
            thresholds.hoare_param1 = max(1, thresholds.hoare_param1 - 1)

        elif key == ord('='):
            thresholds.hoare_param2 = min(255, thresholds.hoare_param2 + 1)
        elif key == ord("-"):
            thresholds.hoare_param2 = max(1, thresholds.hoare_param2 - 1)

        elif key == ord('w'):
            thresholds.hoare_dp = min(255, thresholds.hoare_dp + 1)
        elif key == ord("q"):
            thresholds.hoare_dp = max(1, thresholds.hoare_dp - 1)

        elif key == ord('r'):
            thresholds.hoare_min_dist = min(255, thresholds.hoare_min_dist + 1)
        elif key == ord("e"):
            thresholds.hoare_min_dist = max(0, thresholds.hoare_min_dist - 1)

        elif key == ord('y'):
            thresholds.hoare_min_radius = min(255, thresholds.hoare_min_radius + 1)
        elif key == ord("t"):
            thresholds.hoare_min_radius = max(1, thresholds.hoare_min_radius - 1)

        elif key == ord('i'):
            thresholds.hoare_max_radius = min(255, thresholds.hoare_max_radius + 1)
        elif key == ord("u"):
            thresholds.hoare_max_radius = max(1, thresholds.hoare_max_radius - 1)

        elif key == ord('p'):
            thresholds.min_ellipse = min(255, thresholds.min_ellipse + 1)
        elif key == ord("o"):
            thresholds.min_ellipse = max(0, thresholds.min_ellipse - 1)

        elif key == ord(']'):
            thresholds.min_ellipse_aspect = min(255.0, thresholds.min_ellipse_aspect + 0.1)
        elif key == ord("["):
            thresholds.min_ellipse_aspect = max(0.0, thresholds.min_ellipse_aspect - 0.1)

        # elif key == ord("'"):
        #     thresholds.max_ellipse_aspect = min(255.0, thresholds.max_ellipse_aspect + 0.1)
        # elif key == ord(";"):
        #     thresholds.max_ellipse_aspect = max(0.0, thresholds.max_ellipse_aspect - 0.1)

        elif key == ord("l"):
            thresholds.max_ellipse = min(255.0, thresholds.max_ellipse + 0.1)
        elif key == ord("k"):
            thresholds.max_ellipse = max(0.0, thresholds.max_ellipse - 0.1)

        elif key == ord("#"):
            thresholds.draw_original = not thresholds.draw_original


        elif key == ord('x'):  # 'q' key
            break
        else:
            print(f"Unknown key: {key}")
    cv2.destroyAllWindows()

# Main function
def main():
    global args
    model = load_model(args.model)
    images = load_images(args.input)
    images = process_frames(images)
    handle_input(images, 4, model)

if __name__ == "__main__":
    main()