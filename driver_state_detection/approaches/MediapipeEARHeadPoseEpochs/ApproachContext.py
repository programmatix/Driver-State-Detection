import mediapipe as mp
from typing import List

from GlobalContext import GlobalContext
from approaches.MediapipeEARHeadPoseEpochs.Model import EpochOngoing
from approaches.MediapipeEARHeadPoseEpochs.TrainingConstants import REQUIRED_FPS


class ApproachParams:
    epoch_time_frames = REQUIRED_FPS * 5
    ear_left_threshold_for_closed = 0.2
    good_pitch_threshold = 15

class ApproachContext:
    params = ApproachParams()
    epoch: EpochOngoing = None
    finished_epochs: List[EpochOngoing] = []

    detector = mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
                                               min_detection_confidence=0.5,
                                               min_tracking_confidence=0.5,
                                               # https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_mesh.md#attention-mesh-model
                                               refine_landmarks=True)

    def __init__(self, gc: GlobalContext):
        self.gc = gc

