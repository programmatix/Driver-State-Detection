import cv2
import os
import queue
import socket
from dotenv import load_dotenv

load_dotenv()

class GlobalContext:
    hostname = socket.gethostname()


    INFLUXDB_URL = os.getenv('INFLUXDB_URL')
    INFLUXDB_TOKEN = os.getenv('INFLUXDB_TOKEN')
    INFLUXDB_ORG = os.getenv('INFLUXDB_ORG')
    INFLUXDB_BUCKET = os.getenv('INFLUXDB_BUCKET')

    # Some approaches make assumptions/requirements of this
    required_capture_fps = 60

    capture_fps = 0
    save_fps = 0
    process_fps = 0

    frame_queue_for_saving = queue.Queue()
    frame_queue_for_processing = queue.Queue()

    done = False
    capture_mode = False
    capture_single_frame_mode = False
    buffer_mode = False
    dump_buffered_frames = False
    buffered_frames = []
    debug_mode = False
    flip_mode = 0
    # False means use frame left eye (which Mediapipe also says is left) as the physical left eye (e.g. the one we
    # are interested in).  True means use the frame right eye as the physical left eye.
    flip_eye_mode = True
    print_timings = False
    saving_to_influx = False
    model = None
    mode = 0

    def __init__(self, args):
        self.args = args
        self.saving_to_influx = getattr(args, 'write_to_influx', False) or None
        self.mode = getattr(args, 'mode', False) or None
        if args.input:
            self.frame_queue_for_processing.put(cv2.imread(args.input))
