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
    flip_mode = 3
    flip_eye_mode = False
    print_timings = False
    saving_to_influx = False
    model = None
    mode =  0

    def __init__(self, args):
        self.args = args
        self.saving_to_influx = args.write_to_influx
        self.mode = args.mode
        if args.input:
            self.frame_queue_for_processing.put(cv2.imread(args.input))
