import sys

import argparse
import cv2
import os
import shutil
from glob import glob
from typing_extensions import List


# Yeah this is great coding keep it up
sys.path.insert(0, "C:\\dev\\Projects\\Driver-State-Detection\\driver_state_detection")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from GlobalContext import GlobalContext
from approaches.MediapipeEARHeadPoseEpochs.Training.TrainingDisplay import display_frames
from Model import ImageAndFilename

from Approach import handle_image
from ApproachContext import ApproachContext
from approaches.MediapipeEARHeadPoseEpochs.Model import ImageAndFilenameAndContext, AnalysedImageAndTimeAndContext, GoodBad, \
    TrainingSet


def load_training_sets(folder, max=None) -> List[TrainingSet]:
    out: List[TrainingSet] = []

    has_subdirectories = any(entry.is_dir() for entry in os.scandir(folder))

    if has_subdirectories:
        folders = [f.path for f in os.scandir(folder) if f.is_dir()]
        for folder in folders:
            out.append(load_training_set(folder, max))
    else:
        out.append(load_training_set(folder, max))
    return out


def load_training_set(folder, max=None) -> TrainingSet:
    images = []
    print(f'Loading from {folder}')
    count = 0
    # Primary sort on timestamp, secondary sort on frame number
    # Filename: 2024-05-27_09-10-39-frame-97-orig.jpg
    for filename in sorted(glob(folder.replace("\\", "/") + '/*.jpg'), key=lambda f: (f.split('-frame-')[0], int(f.split("-frame-")[1].split('-')[0]))):
        if max is not None and count > max:
            break
        img = cv2.imread(filename)
        count += 1

        if img is not None:
            images.append(ImageAndFilename(filename, img))
    print(f"Loaded {len(images)} images")
    return TrainingSet(images, folder)

def save_images(images: List[ImageAndFilename], current_index, output_dir, label):
    image = images[current_index]
    outfilename = image.filename.split("-end-")[0] + "-end-" + label + ".jpg"
    print(f"Saving {image.filename} to {outfilename}")
    os.rename(image.filename, outfilename)
    images[current_index].filename = outfilename


# Use the FaceMesh model to detect faces in each frame and zoom into the left eye
# Handle user input
def handle_input(ac: ApproachContext, images: List[ImageAndFilename], analysed: List[ImageAndFilenameAndContext], current_index):
    while True:
        display_frames(images, analysed, current_index)
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
        elif key == ord('b'):
            save_images(images, current_index, "labels", "blink")
            current_index = min(len(images) - 1, current_index + 1)
        elif key == ord('z'):
            save_images(images, current_index, "labels", "ambiguous-down")
            current_index = min(len(images) - 1, current_index + 1)
        elif key == ord('x'):
            save_images(images, current_index, "labels", "ambiguous-up")
            current_index = min(len(images) - 1, current_index + 1)
        elif key == ord('f'):
            save_images(images, current_index, "labels", "ambiguous-fakeout")
            current_index = min(len(images) - 1, current_index + 1)
        elif key == ord('q'):
            break
        else:
            print(f"Unknown key: {key}")
    cv2.destroyAllWindows()


# Main function
def main():
    parser = argparse.ArgumentParser(description='Process images in a folder')
    parser.add_argument('input', type=str, help='The path to the folder containing the images')
    args = parser.parse_args()

    gc = GlobalContext(args)
    gc.debug_mode = True
    gc.flip_eye_mode = True
    ac = ApproachContext(gc)
    training_sets = load_training_sets(args.input)
    good_bad_count = {good_bad: 0 for good_bad in GoodBad}

    analysed: List[ImageAndFilenameAndContext] = []
    for t in range(0, len(training_sets)):
        ts = training_sets[t]
        good_bad_count_for_training_set = {good_bad: 0 for good_bad in GoodBad}

        for i in range(0, len(ts.images)):
            text_list = []
            img = ts.images[i]
            ai: AnalysedImageAndTimeAndContext = handle_image(ac, img.image, text_list, i)

            good = classify(ai, img, text_list)

            out = ImageAndFilenameAndContext(ts.images[i], ai, text_list, good)
            analysed.append(out)
            good_bad_count[good] += 1
            good_bad_count_for_training_set[good] += 1

        print(f"Training set: {ts.folder}")
        print_good_bad(good_bad_count_for_training_set)

    print_good_bad(good_bad_count)
    if len(training_sets) == 1:
        handle_input(ac, training_sets[0].images, analysed, 0)


def classify(ai, img, text_list):
    good = GoodBad.UNKNOWN
    if ai is not None:
        if ai.bc is not None:
            is_labelled_not_blink = "-not-blink" in img.filename
            is_labelled_blink = not is_labelled_not_blink and "-blink" in img.filename
            is_labelled_ambiguous = "-ambiguous" in img.filename

            if is_labelled_ambiguous:
                text_list.append(
                    f"Ambiguous is labelled so we don't mind what the algo thinks (which is blinking={ai.bc.blink_state})")
                good = GoodBad.LABEL_AMBIGUOUS

            if not is_labelled_ambiguous:
                correctly_matches_blink = ai.bc.currently_blinking() and is_labelled_blink
                correctly_matches_not_blink = not ai.bc.currently_blinking() and is_labelled_not_blink

                if correctly_matches_blink:
                    text_list.append("Blink is labelled and algo matches it - good!")
                    good = GoodBad.MATCHED_LABEL
                if correctly_matches_not_blink:
                    text_list.append("Not-blink is labelled and algo matches it - good!")
                    good = GoodBad.MATCHED_LABEL

                if not (correctly_matches_blink or correctly_matches_not_blink):
                    good = GoodBad.DID_NOT_MATCH_LABEL
                    if is_labelled_blink and not ai.bc.currently_blinking():
                        text_list.append("Blink is labelled but algo thinks not-blink - bad!")
                    if is_labelled_not_blink and ai.bc.currently_blinking():
                        text_list.append("Not-blink is labelled but algo thinks blinking - bad!")
    return good


def print_good_bad(good_bad_count):
    total = good_bad_count[GoodBad.MATCHED_LABEL] + good_bad_count[GoodBad.DID_NOT_MATCH_LABEL]

    print(f"Matched label: {good_bad_count[GoodBad.MATCHED_LABEL]} ({good_bad_count[GoodBad.MATCHED_LABEL] / total * 100}%)")
    print(f"Failed to match label: {good_bad_count[GoodBad.DID_NOT_MATCH_LABEL]} ({good_bad_count[GoodBad.DID_NOT_MATCH_LABEL] / total * 100}%)")
    print(f"Label was ambiguous so model result ignored: {good_bad_count[GoodBad.LABEL_AMBIGUOUS]}")
    print(f"Unknown: {good_bad_count[GoodBad.UNKNOWN]}")


if __name__ == "__main__":
    main()