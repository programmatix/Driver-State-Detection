import sys

import argparse
import numpy as np
import os
from typing_extensions import List


# Yeah this is great coding keep it up
sys.path.insert(0, "C:\\dev\\Projects\\Driver-State-Detection\\driver_state_detection")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from GlobalContext import GlobalContext

from approaches.MediapipeEARMultiFrame.Training.Training import load_training_sets, classify, print_good_bad
from approaches.MediapipeEARMultiFrame.Approach import handle_image

from ApproachContext import ApproachContext
from approaches.MediapipeEARMultiFrame.Model import ImageAndFilenameAndContext, AnalysedImageAndTimeAndContext, GoodBad


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

    best = None

    permutations = 0
    for frames_lookback in range(1, 20):
        for ear_left_threshold_for_blink_start in np.arange(0.3, 0.8, 0.05):
            for ear_left_threshold_for_blink_stop in np.arange(0.2, 0.8, 0.05):
                permutations += 1

    permutation = 0
    for frames_lookback in range(1, 20):
        for ear_left_threshold_for_blink_start in np.arange(0.3, 0.8, 0.05):
            for ear_left_threshold_for_blink_stop in np.arange(0.2, 0.8, 0.05):
                ac = ApproachContext(gc)
                ac.frames_lookback = frames_lookback
                ac.ear_left_threshold_for_blink_start = ear_left_threshold_for_blink_start
                ac.ear_left_threshold_for_blink_stop = ear_left_threshold_for_blink_stop

                good_bad_count = {good_bad: 0 for good_bad in GoodBad}

                for t in range(0, len(training_sets)):
                    ts = training_sets[t]
                    good_bad_count_for_training_set = {good_bad: 0 for good_bad in GoodBad}

                    for i in range(0, len(ts.images)):
                        text_list = []
                        img = ts.images[i]
                        ai: AnalysedImageAndTimeAndContext = handle_image(ac, img.image, text_list, i)

                        good = classify(ai, img, text_list)

                        good_bad_count[good] += 1
                        good_bad_count_for_training_set[good] += 1

                    print(f"Training set: {ts.folder} frames_lookback={frames_lookback} ear_left_threshold_for_blink_start={ear_left_threshold_for_blink_start} ear_left_threshold_for_blink_stop={ear_left_threshold_for_blink_stop}")
                    print_good_bad(good_bad_count_for_training_set)

                print(f"Overall {permutation} / {permutations} frames_lookback={frames_lookback} ear_left_threshold_for_blink_start={ear_left_threshold_for_blink_start} ear_left_threshold_for_blink_stop={ear_left_threshold_for_blink_stop}")
                print_good_bad(good_bad_count)
                permutation += 1

                if best is None or good_bad_count[GoodBad.MATCHED_LABEL] > best[1]:
                    best = ((frames_lookback, ear_left_threshold_for_blink_start, ear_left_threshold_for_blink_stop), good_bad_count[GoodBad.MATCHED_LABEL])

    print(f"Best: {best}")

if __name__ == "__main__":
    main()