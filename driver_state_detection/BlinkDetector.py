import time

class BlinkDetector:
    def __init__(self, ear_threshold):
        self.ear_threshold = ear_threshold
        self.blink_start_time = None
        self.blinks = []  # Stores tuples of (start_time, end_time)
        self.current_ear = 1.0  # Assume eyes are open initially
        self.creation_time = time.perf_counter()

    def update_ear(self, ear):
        current_time = time.perf_counter()
        # Remove blinks older than 60 seconds
        self.blinks = [blink for blink in self.blinks if current_time - blink[1] <= 60]

        self.current_ear = ear
        if ear < self.ear_threshold and self.blink_start_time is None:
            print("Blink started")
            # Eye just closed
            self.blink_start_time = current_time
        elif ear >= self.ear_threshold and self.blink_start_time is not None:
            # Eye just opened, blink ended
            blink_end_time = current_time
            self.blinks.append((self.blink_start_time, blink_end_time))
            print("Blink ended duration:", blink_end_time - self.blink_start_time, "seconds")
            self.blink_start_time = None

    def get_blink_data_all(self):
        # Filter blinks that occurred in the last 60 seconds
        current_time = time.perf_counter()
        recent_blinks = [blink for blink in self.blinks if current_time - blink[1] <= 60]
        blink_count = len(recent_blinks)
        if blink_count > 0:
            average_duration = sum(end - start for start, end in recent_blinks) / blink_count
        else:
            average_duration = 0
        if current_time - self.creation_time < 60:
            return None, average_duration
        else:
            return blink_count, average_duration

    def get_blink_data_recent(self, seconds):
        current_time = time.perf_counter()
        recent_blinks = [blink for blink in self.blinks if current_time - blink[1] <= seconds]
        blink_count = len(recent_blinks)
        if blink_count > 0:
            average_duration = sum(end - start for start, end in recent_blinks) / blink_count
        else:
            average_duration = 0
        return blink_count, average_duration
