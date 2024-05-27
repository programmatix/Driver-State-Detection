import time

import TrainingConstants as tc
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import TrainingConstants as tc

def predict(images, model):
    imgs = list(map(lambda img: img[1], images))
    return predict_single(imgs, model)

# A single batch of 5 images
def predict_single(images, model):
    X = prepare_images_for_model(images)
    expanded = np.expand_dims(X, axis=0)
    y_pred = model.predict(expanded)
    out = round(float(y_pred) * 100, 1)
    print(f"Prediction: {out} for {expanded.shape}")
    return out

# Predicting for multiple batches of 5 images
def predict_multi(batches, model):
    X = []
    for batch in batches:
        X.append(prepare_images_for_model(batch))
    # Convert list to numpy array and add an extra dimension for the batch size
    X = np.array(X)
    y_pred = model.predict_on_batch(X)
    # y_pred is now a batch of outputs, so we need to round each one
    out = [round(float(pred) * 100, 1) for pred in y_pred]
    #print(f"Predictions: {out} for {X.shape}")
    return out

def prepare_images_for_model(images):
    tX = time.perf_counter()
    if (len(images) != tc.IMAGES_SHOWN_TO_MODEL):
        raise Exception(f"Insufficient images to predict: {len(images)}")

    to_concat = []
    for i in range(0, len(images)):
        img = images[i]
        #print(f"Preparing image {filename} for model")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        to_concat.append(x)

    out = np.concatenate(to_concat, axis=0)
    #print(f"Time to prepared {len(out)} images and produced tensor {out.shape}: {(time.perf_counter() - tX) * 1000}")
    return out