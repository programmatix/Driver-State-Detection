import time

import cv2
import numpy as np
from typing import List

import approaches.MediapipeEARMultiFrame.BlinkRecorder
from GlobalContext import GlobalContext
from approaches import MediapipeEARMultiFrame
import mediapipe as mp

from approaches.MediapipeEARMultiFrame.Model import AnalysedImageAndTimeAndContext


class ApproachContext:
    rolling_buffer_for_debug: List[AnalysedImageAndTimeAndContext] = []
    rolling_buffer = []

    images_to_keep = 150

    blink_recorder = approaches.MediapipeEARMultiFrame.BlinkRecorder.BlinkRecorder(60)
    detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5,
                                               # https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#attention-mesh-model
                                               refine_landmarks=True)

    def __init__(self, gc: GlobalContext):
        self.gc = gc

