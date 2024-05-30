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
