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


class AnalysedImageWithContext:
    analysed: AnalysedImage


class AnalysedImageAndTime:
    ai: AnalysedImage

    def __init__(self, ai: AnalysedImage, timestamp):
        self.ai = ai
        self.timestamp = timestamp


class BlinkContext:
    def __init__(self, currently_blinking, blinks_in_last_period, blinks_total):
        self.currently_blinking = currently_blinking
        self.blinks_in_last_period = blinks_in_last_period
        self.blinks_total = blinks_total


class AnalysedImageAndTimeAndContext:
    def __init__(self, ai: AnalysedImageAndTime, bc: BlinkContext):
        self.ai = ai
        self.bc = bc
