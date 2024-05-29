import cv2

def scale_down(img, scale_factor):
    # Get the original size of the image
    original_size = img.shape
        
    # Calculate the new size of the image
    new_width = int(original_size[1]*scale_factor)
    new_height = int(original_size[0]*scale_factor)
    #new_size = (int(original_size[1]*scale_factor), int(original_size[0]*scale_factor))
    
    # Resize (shrink) the image
    shrunken = cv2.resize(img, (new_width, new_height))
    
    # Calculate the size of the padding
    pad_top = (original_size[0] - new_height) // 2
    pad_bottom = original_size[0] - new_height - pad_top
    pad_left = (original_size[1] - new_width) // 2
    pad_right = original_size[1] - new_width - pad_left
    
    # print(f"Original size: {original_size} {original_size[0]}, shrunk {shrunken_mask_outside_eye.shape}, padding: {pad_top, pad_bottom, pad_left, pad_right}")
    
    # Pad the shrunken image with white pixels to bring it back to the original size
    return cv2.copyMakeBorder(shrunken, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=255)


def clean_eye(raw):
    steps = []

    orig = raw.copy()
    steps.append(raw)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    steps.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    # img = gray
    # img = cv2.GaussianBlur(img, (3, 3), 1)
    invert = cv2.bitwise_not(gray)
    steps.append(cv2.cvtColor(invert, cv2.COLOR_GRAY2BGR))
    mask_outside_eye = cv2.inRange(invert, 255, 255)
    mask_outside_eye = scale_down(mask_outside_eye, 0.9)
    steps.append(cv2.cvtColor(mask_outside_eye, cv2.COLOR_GRAY2BGR))
    mask_light_areas = cv2.inRange(invert, 150, 255)
    steps.append(cv2.cvtColor(mask_light_areas, cv2.COLOR_GRAY2BGR))
    orig[mask_light_areas == 255] = (0, 0, 255)
    orig[mask_outside_eye == 255] = (255, 0, 0)

    pixels_over_threshold = cv2.countNonZero(cv2.inRange(orig, (0, 0, 150), (0, 0, 255)))
    cv2.putText(orig, str(pixels_over_threshold), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)
    prediction = str(pixels_over_threshold)

    steps.append(orig)

    return prediction, orig, steps
