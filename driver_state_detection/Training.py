import argparse

import cv2
import os
import mediapipe as mp
from glob import glob
import TrainingConstants as tc
from TrainingDisplay import display_frames
from TrainingProcess import process_frames

parser = argparse.ArgumentParser(description='Process images in a folder')
parser.add_argument('input', type=str, help='The path to the folder containing the images')
args = parser.parse_args()


# Load all images from a folder in name-sorted order
def load_images(folder):
    images = []
    print(f'Loading from {folder}')
    count = 0
    for filename in sorted(glob(folder.replace("\\", "/") + '/*.jpg')):
        if count >= 20:
            break
        print(f'Loading {filename}')
        img = cv2.imread(filename)
        count += 1

        if img is not None:
            images.append((filename, img))
    print(f"Loaded {len(images)} images")
    return images


def save_images(images, current_index, output_dir):
    start_index = max(0, current_index - 4)
    for i in range(start_index, current_index + 1):
        filename, img = images[i]
        # Split the filename into directory and base name
        dirname, basename = os.path.split(filename)
        # Remove the 'orig' part from the directory name
        dirname = dirname.replace('orig', '')
        # Create the output directory if it doesn't exist
        output_dirname = os.path.join(output_dir, dirname)
        os.makedirs(output_dirname, exist_ok=True)
        # Add "-blink-" only to the current frame's filename
        if i == current_index:
            output_filename = os.path.join(output_dirname, basename.replace('-orig-', '-blink-'))
        else:
            output_filename = os.path.join(output_dirname, basename)
        print(f"Saving image {output_filename}")
        cv2.imwrite(output_filename, img)

# Use the FaceMesh model to detect faces in each frame and zoom into the left eye
# Handle user input
def handle_input(images, current_index):
    while True:
        display_frames(images, current_index)
        key = cv2.waitKey(0)
        if key == ord('a'):  # Left key
            current_index = max(0, current_index - 1)
        elif key == ord('d'):  # Right key
            current_index = min(len(images) - 1, current_index + 1)
        elif key == ord(' '):  # Spacebar
            save_images(images, current_index, "labels")
            current_index = min(len(images) - 1, current_index + 1)
        elif key == ord('s'):  # 's' key
            current_index = min(len(images) - 1, current_index + 1)
        elif key == ord('q'):  # 'q' key
            break
    cv2.destroyAllWindows()

# Main function
def main():
    global args
    images = load_images(args.input)
    images = process_frames(images)
    handle_input(images, 4)

if __name__ == "__main__":
    main()