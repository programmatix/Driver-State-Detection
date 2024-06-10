from datetime import datetime

import socket
from typing import List

from approaches.MediapipeEARMultiFrame.Model import AnalysedImageAndTime, AnalysedImage, AnalysedImageAndTimeAndContext, \
    BlinkContext, BlinkState


# We tie BlinkRecorder to frames not time as it makes training simpler
class BlinkRecorder:
    blinks_total = 0
    _blinks_in_last_period: List[AnalysedImageAndTime] = []
    blink_state = BlinkState.NOT_BLINKING
    current_blink_duration_frames = None

    def currently_blinking(self):
        return self.blink_state == BlinkState.BLINK_IN_PROGRESS or self.blink_state == BlinkState.BLINK_JUST_STARTED

    def __init__(self, period_frames):
        self.period_frames = period_frames
        self.hostname = socket.gethostname()

    def record(self, ac, latest: AnalysedImage, frame_idx: int) -> AnalysedImageAndTimeAndContext:
        self._blinks_in_last_period = [b for b in self._blinks_in_last_period if (frame_idx - b.frame_idx) <= self.period_frames]

        is_blinking = False
        has_stopped_blinking = False

        if not self.currently_blinking():
            # Have we started blinking
            is_blinking = latest.ear_left_diff_ratio > ac.params.ear_left_threshold_for_blink_start and latest.ear_left_diff < 0
        else:
            # Have we stopped blinking
            has_stopped_blinking = latest.ear_left_diff_ratio > ac.params.ear_left_threshold_for_blink_stop and latest.ear_left_diff > 0

        if self.blink_state == BlinkState.BLINK_JUST_ENDED:
            self.blink_state = BlinkState.NOT_BLINKING
            self.current_blink_duration_frames = None
        elif is_blinking and not self.currently_blinking():
            self.blink_state = BlinkState.BLINK_JUST_STARTED
            self.blinks_total += 1
            self.current_blink_duration_frames = 1
            self._blinks_in_last_period.append(AnalysedImageAndTime(latest, frame_idx))
            print(f"{frame_idx} Blink start at frame {frame_idx}, total blinks {self.blinks_total}, blinks in last period {len(self._blinks_in_last_period)}")
        elif self.currently_blinking() and not has_stopped_blinking:
            self.blink_state = BlinkState.BLINK_IN_PROGRESS
            self.current_blink_duration_frames += 1
        elif self.currently_blinking() and has_stopped_blinking:
            self.current_blink_duration_frames += 1
            self.blink_state = BlinkState.BLINK_JUST_ENDED
            print(f"{frame_idx} Blink end at frame {frame_idx} after {self.current_blink_duration_frames} frames, total blinks {self.blinks_total}, blinks in last period {len(self._blinks_in_last_period)}")
        bc = BlinkContext(self.blink_state, len(self._blinks_in_last_period), self.blinks_total, self.current_blink_duration_frames)

        return AnalysedImageAndTimeAndContext(AnalysedImageAndTime(latest, frame_idx), bc)

    def record_empty(self, ac):
        if self.currently_blinking and self.current_blink_duration_frames is not None:
            self.current_blink_duration_frames += 1
