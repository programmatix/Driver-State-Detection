import time

import cv2
import numpy as np
from typing import List

import approaches.MediapipeEARMultiFrame.BlinkRecorder
from GlobalContext import GlobalContext
from approaches import MediapipeEARMultiFrame
import mediapipe as mp

from approaches.MediapipeEARMultiFrame.Model import AnalysedImageAndTimeAndContext

class ApproachParams:
    # How many previous frames to compare current frame against
    frames_lookback = 10
    ear_left_threshold_for_blink_start = 0.5
    ear_left_threshold_for_blink_stop = 0.4

class ApproachContext:
    params = ApproachParams()
    rolling_buffer_for_debug: List[AnalysedImageAndTimeAndContext] = []
    rolling_buffer = []
    total_ear_left = 0.0
    ear_left_count = 0


    detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5,
                                               # https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#attention-mesh-model
                                               refine_landmarks=True)

    def __init__(self, gc: GlobalContext):
        self.gc = gc
        self.blink_recorder = approaches.MediapipeEARMultiFrame.BlinkRecorder.BlinkRecorder(self.gc.required_capture_fps * 60)


    def avg_ear_left(self):
        return self.total_ear_left / self.ear_left_count

