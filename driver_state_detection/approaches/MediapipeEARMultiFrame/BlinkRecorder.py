from datetime import datetime

import socket

from approaches.MediapipeEARMultiFrame.Model import AnalysedImageAndTime, AnalysedImage, AnalysedImageAndTimeAndContext, \
    BlinkContext


class BlinkRecorder:
    blinks_total = 0
    _blinks_in_last_period: list[AnalysedImageAndTime] = []
    currently_blinking = False

    def __init__(self, period_seconds):
        self.period_seconds = period_seconds
        self.hostname = socket.gethostname()

    def record(self, latest: AnalysedImage) -> AnalysedImageAndTimeAndContext:
        # Only record blink starts currently
        current_time = datetime.now()
        is_blinking = latest.ear_left_diff_ratio > 0.5 and latest.ear_left_diff < 0
        self._blinks_in_last_period = [b for b in self._blinks_in_last_period if (current_time - b.timestamp).seconds <= self.period_seconds]
        if is_blinking and not self.currently_blinking:
            self.currently_blinking = True
            self.blinks_total += 1
            self._blinks_in_last_period.append(AnalysedImageAndTime(latest, current_time))
            print(f"{current_time} Blink start, total blinks {self.blinks_total}, blinks in last period {len(self._blinks_in_last_period)}")
        if self.currently_blinking and not is_blinking:
            print(f"{current_time} Blink end, total blinks {self.blinks_total}, blinks in last period {len(self._blinks_in_last_period)}")
            self.currently_blinking = False
        bc = BlinkContext(self.currently_blinking, len(self._blinks_in_last_period), self.blinks_total)
        return AnalysedImageAndTimeAndContext(AnalysedImageAndTime(latest, current_time), bc)
