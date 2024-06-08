import cv2
import numpy as np


def cram_homogenous_images(images, output_x):
    img_width = images[0].shape[1]
    img_height = images[0].shape[0]
    num_cols = int(output_x / img_width)
    num_rows = (len(images) // num_cols) + 1
    output_y = num_rows * img_height

    out = np.zeros((output_y, output_x, 3))

    for i in range(min(num_rows * num_cols, len(images))):
        row = i // num_cols
        col = i % num_cols
        x = col * img_width
        y = row * img_height
        # if y + img_height <= out.shape[0] and x + img_width <= processed.shape[1]:
        out[y:y+img_height, x:x+img_width] = images[i]
        #cv2.putText(processed, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


def zoom_in(frame, zoom_factor=2):
    """
    Zooms in on the frame by the specified zoom factor.
    A zoom_factor of 2 means the image will be zoomed in by 100%, focusing on the center.

    Parameters:
    - frame: The original webcaframe.
    - zoom_factor: The factor by which to zoom in on the frame.

    Returns:
    - The zoomed-in frame.
    """
    height, width = frame.shape[:2]
    new_width, new_height = width // zoom_factor, height // zoom_factor

    # Calculate the region of interest
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2

    # Extract the zoomed-in region
    zoomed_in_region = frame[top:bottom, left:right]

    # Resize back to original frame size
    zoomed_in_frame = cv2.resize(zoomed_in_region, (width, height), interpolation=cv2.INTER_LINEAR)

    return zoomed_in_frame


def compress_frame(frame, quality=95):
    # Encode the frame into a memory buffer
    ret, jpeg_frame = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

    # Decode the memory buffer back into an image
    # compressed_frame = cv2.imdecode(jpeg_frame, 1)

    return jpeg_frame

