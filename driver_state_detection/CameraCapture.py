import time

import cv2
import traceback
from datetime import datetime

import GlobalContext


def open_camera(gc: GlobalContext):
    #cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(gc.args.camera)
    # cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840) # 4k/high_res
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160) # 4k/high_res

    cv2.setUseOptimized(True)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # 4k/high_res
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 4k/high_res
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # 4k/high_res
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080) # 4k/high_res
    cap.set(cv2.CAP_PROP_FPS, 60) # 4k/high_res

    print(f"Camera supports CAP_PROP_ZOOM: {cap.get(cv2.CAP_PROP_ZOOM)}")
    print(f"Camera supports CAP_PROP_FOURCC: {cap.get(cv2.CAP_PROP_FOURCC)}")

    # Request compression
    # fourcc = cv2.VideoWriter_fourcc(*'X265')
    # print(f"Setting cap CAP_PROP_FOURCC to X265: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'X264')
    # print(f"Setting cap CAP_PROP_FOURCC to X264: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # print(f"Setting cap CAP_PROP_FOURCC to XVID: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    print(f"Setting cap CAP_PROP_FOURCC to MJPG: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # print(f"Setting cap CAP_PROP_FOURCC to MP4V: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    #
    # fourcc = cv2.VideoWriter_fourcc(*'VP80')
    # print(f"Setting cap CAP_PROP_FOURCC to VP80: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'VP90')
    # print(f"Setting cap CAP_PROP_FOURCC to VP90: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'WMV1')
    # print(f"Setting cap CAP_PROP_FOURCC to WMV1: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'WMV2')
    # print(f"Setting cap CAP_PROP_FOURCC to WMV2: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))
    #
    # fourcc = cv2.VideoWriter_fourcc(*'AVC1')
    # print(f"Setting cap CAP_PROP_FOURCC to AVC1: ", cap.set(cv2.CAP_PROP_FOURCC, fourcc))

    #print(f"Setting cap CAP_MODE_GRAY: ", cap.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY))

    print(f"Setting cap VIDEO_ACCELERATION_ANY: ", cap.set(cv2.VIDEO_ACCELERATION_ANY, 1))
    #cap.set(cv2.VIDEO_ACCELERATION_ANY, 1)

    # cap.set(cv2.CAP_PROP_FPS, 60) # 4k/high_res
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)


    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"The default resolution of the webcam is {width}x{height} {int(fps)}FPS")

    return cap


def capture_frames(gc: GlobalContext):
    try:
        # p = psutil.Process(os.getpid())
        # # Set the process priority to above normal, this can be adjusted to your needs
        # p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        # # Additionally, set the thread priority if needed
        # threading.current_thread().priority = threading.PRIORITY_HIGHEST

        frame_idx = 0
        prev_second = None

        cap = open_camera(gc)

        #histogram = HdrHistogram()
        while gc.done is False:
            #tX = time.perf_counter()
            ret, frame = cap.read()
            #if (print_timings):
            #histogram.record_value((time.perf_counter() - tX) * 1000)
            #print(f"Time to read frame: {(time.perf_counter() - tX) * 1000} FPS: {capture_fps}")

            current_time = datetime.now()
            current_second = current_time.strftime("%S")

            if prev_second is None or prev_second != current_second:
                prev_second = current_second
                gc.capture_fps = frame_idx
                frame_idx = 0  # Reset frame index for each new second
            else:
                frame_idx += 1

            # cv2.imshow(f"window", frame)
            # cv2.waitKey(1)

            if not ret:
                print("Can't receive frame from camera/stream end")
                time.sleep(1)
                cap = open_camera(gc)
            else:
                gc.frame_queue_for_processing.put(frame)
                #print(f"Frame queue {gc.frame_queue_for_processing.qsize()}")

            #Every second display histogram
        print("Capture thread done")
        cap.release()
    except Exception as e:
        print("Stack trace: " + traceback.format_exc())
    print("capture_frames thread ended")

