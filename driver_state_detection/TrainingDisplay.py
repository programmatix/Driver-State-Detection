import cv2
import numpy as np

import TrainingConstants as tc
import ModelPredict
from Thresholds import Thresholds

def fiddle_image(img, thresholds: Thresholds):
    img_orig = img.copy()
    if thresholds.invert:
        img = cv2.bitwise_not(img)
    
    if thresholds.blur_1 > 0:
        img = cv2.GaussianBlur(img, (thresholds.blur_1, thresholds.blur_1), thresholds.blur_sigma)
    # Normalize the image so that the lightest point is pure white, and the darkest point is pure black
    img = cv2.normalize(img, None, alpha=thresholds.normalize_alpha, beta=thresholds.normalize_beta, norm_type=cv2.NORM_MINMAX)
    #img = cv2.normalize(img, None, alpha=thresholds[1], beta=thresholds[2], norm_type=cv2.NORM_MINMAX)
    if thresholds.canny_1 > 0:
        img = cv2.Canny(img, thresholds.canny_1, thresholds.canny_2)
    # Apply Gaussian Blur
    #blurred = img

    #converted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


    # Use HoughCircles to detect the iris- not working
    if thresholds.hoare_min_dist > 0:
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=thresholds.hoare_dp, minDist=thresholds.hoare_min_dist, param1=thresholds.hoare_param1, param2=thresholds.hoare_param2, minRadius=thresholds.hoare_min_radius, maxRadius=thresholds.hoare_max_radius)
    
    
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # Draw the circle on the mask
                cv2.circle(img_color, (x, y), r, (255, 255, 255), thickness=-1)

    # Apply the mask to the original image
    # masked_image = cv2.bitwise_and(img_color, img_color, mask=mask)

    # _, thresholded = cv2.threshold(blurred, thresholds[1], 255, cv2.THRESH_BINARY_INV)
    #
    # # Find contours in the threshold image
    # contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # print(f"Found {len(contours)} contours")
    # # From the detected contours, find the one with the maximum area
    # if contours:
    #     max_contour = max(contours, key=cv2.contourArea)
    #     # Calculate the center and radius of the minimum enclosing circle
    #     ((x, y), radius) = cv2.minEnclosingCircle(max_contour)
    #     # Draw the circle on the original image
    #     cv2.circle(img_color, (int(x), int(y)), int(radius), (0, 255, 0), 2)

    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the original image to draw the ellipse
    # output = image.copy()

    # Loop over the contours
    # print(contours)
    # for contour in contours:
    #     if len(contour) >= 5:  # Fit ellipse needs at least 5 points
    #         ellipse = cv2.fitEllipse(contour)
    #         cv2.ellipse(img_color, ellipse, (0, 0, 255), 1)  # Draw the ellipse

    # Initialize variables to store the most circular contour and its aspect ratio
    most_circular_contour = None
    most_circular_aspect_ratio = float('inf')

    # Loop over the contours
    for contour in contours:
        if len(contour) >= 5:  # Fit ellipse needs at least 5 points
            ellipse = cv2.fitEllipse(contour)
            (x, y), (MA, ma), angle = ellipse

            # Calculate the aspect ratio of the ellipse
            aspect_ratio = float(MA) / float(ma)

            if ma < thresholds.min_ellipse:
                continue

            if MA < thresholds.min_ellipse:
                continue

            if ma > thresholds.max_ellipse:
                continue

            if MA > thresholds.max_ellipse:
                continue

            if aspect_ratio < thresholds.min_ellipse_aspect:
                continue

            if aspect_ratio > thresholds.max_ellipse_aspect:
                continue

            cv2.ellipse(img_color, ellipse, (0, 0, 255), 1)  # Draw the ellipse

            # If the aspect ratio is closer to 1 than the current most circular contour, update the most circular contour and aspect ratio
            if abs(aspect_ratio - 1) < abs(most_circular_aspect_ratio - 1):
                most_circular_contour = contour
                most_circular_aspect_ratio = aspect_ratio

    # If a most circular contour was found, draw it
    if most_circular_contour is not None:
        ellipse = cv2.fitEllipse(most_circular_contour)
        cv2.ellipse(img_color, ellipse, (0, 255, 0), 1)  # Draw the ellipse
        cv2.ellipse(img_orig, ellipse, (0, 255, 0), 1)  # Draw the ellipse

    # # Create a binary mask where the pixels with an intensity above the threshold are set to 255 (white), and all other pixels are set to 0 (black)
    mask = cv2.inRange(img, thresholds.draw_red, 255)

    # Create a copy of the original image
    # img_copy = img_color.copy()

    # Set the pixels in the original image where the mask is white to red
    # img_color[mask == 255] = (0, 0, 255)

    if thresholds.draw_original:
        return cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
    else:
        return img_color


def display_frames(images, current_index, model, thresholds):
    # Define the number of images per row
    images_per_row = 10
    # Calculate the number of rows needed to display 50 images
    num_rows = 50 // images_per_row

    # Create a new image to hold all images
    min_height = (num_rows + 1) * tc.EYE_IMAGE_HEIGHT

    # Keep in BGR so we can display text in colour
    colour_channels = 3
    new_image = np.zeros((min_height * 4, (images_per_row * tc.EYE_IMAGE_WIDTH) + 100, colour_channels), dtype=np.uint8)

    start_idx = max(0, current_index - 4)
    end_idx = min(current_index + 50, len(images))

    threshold_draw_red = thresholds.draw_red

    print(f"Displaying images {start_idx} to {end_idx} for {current_index}")

    #predicted = ModelPredict.predict(images[start_idx:current_index + 1], model)

    for i in range(start_idx, end_idx):
        img = images[i][1]

        # Calculate the position of the image on the new image
        row = (i - start_idx) // images_per_row
        col = (i - start_idx) % images_per_row
        x = int(col * tc.EYE_IMAGE_WIDTH)
        y = int(row * tc.EYE_IMAGE_HEIGHT)
        x2 = int(x + img.shape[1])
        y2 = int(y + img.shape[0])

        img_color = fiddle_image(img, thresholds)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

        # Convert the image to grayscale
        #print(f"Image shape: {img_color.shape} y: {y} y2: {y2} x: {x} x2: {x2}")
        new_image[y:y2, x:x2] = img_color





        #print(f"Blitting image of size {img.shape[1]}x{img.shape[0]} into position {x}x{y} to {x2}x{y2}")
        # Copy the image into the correct position on the new image
        # new_image[y:y2, x:x2] = blurred

        # Count the number of pixels over 200
        num_pixels_over_200 = cv2.countNonZero(cv2.inRange(img_gray, thresholds.draw_red, 255))
        #
        # # Display the count above the eye
        cv2.putText(new_image, f"{num_pixels_over_200}", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add the index and prediction under the image
        font_color = (255, 255, 255)
        if i == current_index:
            font_color = (0, 255, 0)
            # cv2.putText(new_image, f"{i} Pred: {predicted}", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 2)
            cv2.putText(new_image, f"{i}", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, font_color, 2)

    # Display the current frame's full filename further down the screen
    filename = images[current_index][0]
    cv2.putText(new_image, filename, (10, new_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Calculate histogram for the current frame
    current_img_gray = images[current_index][1]
    # current_img_gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
    hist_buckets = 256
    hist = cv2.calcHist([current_img_gray],[0],None,[hist_buckets],[0,256])

    # Create a blank image to draw the histogram
    # Create a blank image to draw the histogram with double width
    hist_img = np.zeros((300, 512, 3), dtype=np.uint8)

    # threshold = 0.75 * np.max(hist)

    # Draw the histogram
    for i in range(1, hist_buckets):
        color = (255, 255, 255)  # Default color is white
        if i > threshold_draw_red:
            color = (0, 0, 255)  # If the histogram value is above the threshold, set the color to red
        # Draw two lines for each bin to double the width of each line
        cv2.line(hist_img, ((i-1)*2, hist_img.shape[0] - int(hist[i-1])), ((i)*2, hist_img.shape[0] - int(hist[i])), color, 1)
        cv2.line(hist_img, ((i-1)*2+1, hist_img.shape[0] - int(hist[i-1])), ((i)*2+1, hist_img.shape[0] - int(hist[i])), color, 1)

    # Display the current threshold value above the histogram
    cv2.putText(hist_img, f"Thresholds: {str(thresholds.__dict__)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the histogram below the rows of images
    new_image[min_height:min_height+hist_img.shape[0], :hist_img.shape[1]] = hist_img

    # Display the new image
    cv2.imshow('Frames', new_image)