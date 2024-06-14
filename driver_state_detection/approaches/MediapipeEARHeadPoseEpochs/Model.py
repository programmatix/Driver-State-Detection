from typing_extensions import List

class OriginalImage:
    def __init__(self, filename, original_image):
        self.filename = filename
        self.original_image = original_image

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
    roll = None
    pitch = None
    yaw = None



# class AnalysedImage:
#     processed: ProcessedImage
#     prev: ProcessedImage
#     prev_frame_idx: int
#     ear_left_diff: float
#     ear_left_diff_ratio: float
#     avg_ear_left: float
#
#     def __init__(self, processed: ProcessedImage, avg_ear_left: float, prev: ProcessedImage):
#         self.processed = processed
#         self.avg_ear_left = avg_ear_left
#         # A recent ear_left that is furthest away from current ear_left
#         self.prev_ear_left = prev.ear_left
#         self.prev = prev
#         self.ear_left_diff = self.processed.ear_left - self.prev_ear_left
#         self.ear_left_diff_ratio = abs(self.ear_left_diff) / self.avg_ear_left
#
#
# class AnalysedImageWithContext:
#     analysed: AnalysedImage
#
#
# class AnalysedImageAndTime:
#     ai: AnalysedImage
#
#     def __init__(self, ai: AnalysedImage, frame_idx):
#         self.ai = ai
#         self.frame_idx = frame_idx

# from enum import Enum
#
# class BlinkState(Enum):
#     BLINK_JUST_STARTED = 1
#     BLINK_IN_PROGRESS = 2
#     BLINK_JUST_ENDED = 3
#     NOT_BLINKING = 4
#
# class BlinkContext:
#     def __init__(self, blink_state: BlinkState, blinks_in_last_period, median_blink_duration_in_last_period, blinks_total, current_blink_duration_frames):
#         self.blink_state = blink_state
#         self.blinks_in_last_period = blinks_in_last_period
#         self.median_blink_duration_in_last_period = median_blink_duration_in_last_period
#         self.blinks_total = blinks_total
#         self.current_blink_duration_frames = current_blink_duration_frames
#
#     def currently_blinking(self):
#         return self.blink_state == BlinkState.BLINK_IN_PROGRESS or self.blink_state == BlinkState.BLINK_JUST_STARTED
#

# class AnalysedImageAndTimeAndContext:
#     def __init__(self, ai: AnalysedImageAndTime, bc: BlinkContext):
#         self.ai = ai
#         self.bc = bc
#
# class ImageAndFilename:
#     def __init__(self, filename, image):
#         self.filename = filename
#         self.image = image
#
#
# class GoodBad(Enum):
#     MATCHED_LABEL = 1
#     DID_NOT_MATCH_LABEL = 2
#     LABEL_AMBIGUOUS = 3
#     UNKNOWN = 4

# class ImageAndFilenameAndContext:
#     def __init__(self, img: ImageAndFilename, ai: AnalysedImageAndTimeAndContext, text_list: List[str], good: GoodBad):
#         self.img = img
#         self.ai = ai
#         self.text_list = text_list
#         self.good = good
#
#
# class TrainingSet:
#     def __init__(self, images: List[ImageAndFilename], folder: str):
#         self.images = images
#         self.folder = folder

# class Blink:
#     def __init__(self, start_frame, end_frame):
#         self.start_frame = start_frame
#         self.end_frame = end_frame
#
#     def duration_frames(self):
#         return self.end_frame - self.start_frame + 1
#

class EpochFinished:
    def __init__(self, start_frame_idx: int,
                 end_frame_idx_inclusive: int,
                 ear_left_avg: float,
                 ear_left_median: float,
                 frames_eye_visible: int,
                 frames_good_pitch: int,
                 frames_ear_left_below_threshold: int):
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx_inclusive = end_frame_idx_inclusive
        self.ear_left_avg = ear_left_avg
        self.ear_left_median = ear_left_median
        self.frames_eye_visible = frames_eye_visible
        self.frames_good_pitch = frames_good_pitch
        self.frames_ear_left_below_threshold = frames_ear_left_below_threshold

    def length_frames(self) -> int:
        return self.end_frame_idx_inclusive - self.start_frame_idx + 1

    def __str__(self):
        return f"EpochFinished: start_frame_idx={self.start_frame_idx}, end_frame_idx_inclusive={self.end_frame_idx_inclusive}, frames_eye_visible={self.frames_eye_visible}, frames_good_pitch={self.frames_good_pitch}, frames_ear_left_below_threshold={self.frames_ear_left_below_threshold}"


class EpochOngoing:

    def __init__(self, start_frame_idx: int):
        self.ear_left = []
        self.pitch = []
        self.start_frame_idx = start_frame_idx

    def add(self, ear_left, pitch):
        self.ear_left.append(ear_left)
        self.pitch.append(pitch)

    def frames(self) -> int:
        return len(self.ear_left)

    def finish(self, end_frame_idx_inclusive: int, ear_left_threshold: float, good_pitch_threshold: float) -> EpochFinished:
        ear_left_avg = sum(self.ear_left) / len(self.ear_left) if len(self.ear_left) > 0 else 0
        ear_left_median = sorted(self.ear_left)[len(self.ear_left) // 2] if len(self.ear_left) > 0 else 0
        frames_good_pitch = sum(1 for p in self.pitch if p >= good_pitch_threshold)
        frames_ear_left_below_threshold = sum(1 for e in self.ear_left if e < ear_left_threshold)

        return EpochFinished(self.start_frame_idx,
                             end_frame_idx_inclusive,
                             ear_left_avg,
                             ear_left_median,
                             len(self.ear_left),
                             frames_good_pitch,
                             frames_ear_left_below_threshold)

    def __str__(self):
        return f"EpochOngoing: start_frame_idx={self.start_frame_idx}, frames={self.frames()}"