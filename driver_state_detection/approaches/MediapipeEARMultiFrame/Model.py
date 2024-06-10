from typing_extensions import List


class ProcessedImage:
    frame_idx: int
    # All images are in colour.  Easier to work with, and can put debug info in them.
    eye_img_orig = None
    eye_img_final = None
    eye_img_steps = []
    original_image: any = None
    pupil_center_pixel = None
    ear = None
    # Physical left and right, not frame left and right
    ear_left = None
    ear_right = None


class AnalysedImage:
    processed: ProcessedImage
    prev: ProcessedImage
    prev_frame_idx: int
    ear_left_diff: float
    ear_left_diff_ratio: float
    avg_ear_left: float

    def __init__(self, processed: ProcessedImage, avg_ear_left: float, prev: ProcessedImage):
        self.processed = processed
        self.avg_ear_left = avg_ear_left
        # A recent ear_left that is furthest away from current ear_left
        self.prev_ear_left = prev.ear_left
        self.prev = prev
        self.ear_left_diff = self.processed.ear_left - self.prev_ear_left
        self.ear_left_diff_ratio = abs(self.ear_left_diff) / self.avg_ear_left


class AnalysedImageWithContext:
    analysed: AnalysedImage


class AnalysedImageAndTime:
    ai: AnalysedImage

    def __init__(self, ai: AnalysedImage, frame_idx):
        self.ai = ai
        self.frame_idx = frame_idx

from enum import Enum

class BlinkState(Enum):
    BLINK_JUST_STARTED = 1
    BLINK_IN_PROGRESS = 2
    BLINK_JUST_ENDED = 3
    NOT_BLINKING = 4

class BlinkContext:
    def __init__(self, blink_state: BlinkState, blinks_in_last_period, blinks_total, current_blink_duration_frames):
        self.blink_state = blink_state
        self.blinks_in_last_period = blinks_in_last_period
        self.blinks_total = blinks_total
        self.current_blink_duration_frames = current_blink_duration_frames

    def currently_blinking(self):
        return self.blink_state == BlinkState.BLINK_IN_PROGRESS or self.blink_state == BlinkState.BLINK_JUST_STARTED


class AnalysedImageAndTimeAndContext:
    def __init__(self, ai: AnalysedImageAndTime, bc: BlinkContext):
        self.ai = ai
        self.bc = bc

class ImageAndFilename:
    def __init__(self, filename, image):
        self.filename = filename
        self.image = image


class GoodBad(Enum):
    MATCHED_LABEL = 1
    DID_NOT_MATCH_LABEL = 2
    LABEL_AMBIGUOUS = 3
    UNKNOWN = 4

class ImageAndFilenameAndContext:
    def __init__(self, img: ImageAndFilename, ai: AnalysedImageAndTimeAndContext, text_list: List[str], good: GoodBad):
        self.img = img
        self.ai = ai
        self.text_list = text_list
        self.good = good


class TrainingSet:
    def __init__(self, images: List[ImageAndFilename], folder: str):
        self.images = images
        self.folder = folder