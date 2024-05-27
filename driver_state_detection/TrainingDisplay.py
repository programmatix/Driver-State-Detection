import cv2
import numpy as np

import TrainingConstants as tc
import ModelPredict

def display_frames(images, current_index, model):
    # Create a new image to hold all tc.IMAGES_DISPLAYED images
    new_image = np.zeros((500, (tc.IMAGES_DISPLAYED * tc.EYE_IMAGE_WIDTH) + 100, 3), dtype=np.uint8)

    start_idx = max(0, current_index - 4)
    end_idx = min(current_index + 5, len(images))

    print(f"Displaying images {start_idx} to {end_idx} for {current_index}")

    predicted = ModelPredict.predict(images[start_idx:current_index + 1], model)

    for i in range(start_idx, end_idx):
        img = images[i][1]
        # if i > current_index:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Calculate the position of the image on the new image
        x = (i - start_idx) * tc.EYE_IMAGE_WIDTH
        y = 0
        x2 = x + img.shape[1]
        y2 = y + img.shape[0]

        print(f"Blitting image of size {img.shape[1]}x{img.shape[0]} into position {x}x{y} to {x2}x{y2}")
        # Copy the image into the correct position on the new image
        new_image[y:y2, x:x2] = img

        # Add the index under the image
        font_color = (255, 255, 255)
        if i == current_index:
            font_color = (0, 255, 0)
        cv2.putText(new_image, str(i), (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 2)


    # Display the current frame's full filename further down the screen
    filename = images[current_index][0]
    cv2.putText(new_image, filename, (10, new_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the new image
    cv2.imshow('Frames', new_image)