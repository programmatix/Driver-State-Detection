import time

import cv2
import os
import queue
from datetime import datetime

import GlobalContext


def save_frames(gc: GlobalContext):
    print("Starting save_frames")

    frame_idx = 0
    prev_second = None

    while gc.done is False:
        try:
            frames = gc.frame_queue_for_saving.get(timeout=1)
        except queue.Empty:
            continue

        #print(f"Got frames orig={int(frames[0].nbytes / 1024)}kb processed={int(frames[1].nbytes / 1024)}kb")

        current_time = datetime.now()
        current_second = current_time.strftime("%S")

        if prev_second is None or prev_second != current_second:
            prev_second = current_second
            save_fps = frame_idx
            frame_idx = 0  # Reset frame index for each new second
        else:
            frame_idx += 1

        now = time.time()
        # buffered_frames.append((frames[0], now, frame_idx, "orig"))

        # # We have to compressed the stored image, it's just way too much memory otherwise
        compress_frames = False
        # compressed_proc = compress_frame(frames[1], 95)
        # buffered_frames.append((compressed_proc, now, frame_idx, "proc", current_time))
        if gc.buffer_mode:
            gc.buffered_frames.append((frames[0], now, frame_idx, "orig", current_time))
        # five_minutes_ago = now - 10 * 60
        # buffered_frames = [(frame, timestamp, idx, desc, timestamp2) for frame, timestamp, idx, desc, timestamp2 in buffered_frames if timestamp >= five_minutes_ago]


        if gc.dump_buffered_frames:
            gc.dump_buffered_frames = False

            if compress_frames:
                folderName = "output_images"
            else:
                folderName = f"training/{current_time.strftime('%Y-%m-%d_%H-%M-%S')}"
            try:
                os.mkdir(folderName)
            except FileExistsError:
                pass

            # Save all frames in the buffer that are within the last 5 minutes
            for bframe, timestamp, idx, desc, timestamp2 in gc.buffered_frames:

                fn = timestamp2.strftime("%Y-%m-%d_%H-%M-%S") + "-" + desc + "-" + str(idx)

                filename1 = f"{folderName}/{fn}.jpg"

                print(f"Writing {filename1}")
                if compress_frames:
                    with open(filename1, "wb") as f:
                        f.write(bframe.tobytes())
                else:
                    cv2.imwrite(filename1, bframe)

            gc.buffered_frames = []

        # if (frame_idx == 0):
        #     print("New second " + timestamp)

        if gc.capture_mode or gc.capture_single_frame_mode:
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S") + "-" + str(frame_idx)
            filename1 = f"output_images/{timestamp}-orig.jpg"
            filename2 = f"output_images/{timestamp}-processed.jpg"

            # Don't compress - hard enough to debug
            #cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            cv2.imwrite(filename1, frames[0])
            cv2.imwrite(filename2, frames[1])

            print(f"Saved {filename1} and {filename2}")

            if gc.capture_single_frame_mode:
                cv2.imshow("Single capture", frames[1])

            gc.capture_single_frame_mode = False
