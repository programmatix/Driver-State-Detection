import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import TrainingConstants as tc
from driver_state_detection.ModelPredict import prepare_images_for_model


class ImageInfo:
    def __init__(self, filename, full_filename, image):
        self.filename = filename
        self.full_filename = full_filename
        self.image = image

def load_images_from_folder(folder) -> [ImageInfo]:
    print(f"Loading images from folder: {folder}")
    images = []
    for filename in sorted(os.listdir(folder)):
        print(f"Processing file: {filename}")
        full_filename = os.path.join(folder, filename)
        img = image.load_img(full_filename, target_size=(tc.EYE_IMAGE_WIDTH, tc.EYE_IMAGE_HEIGHT))
        # x = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # x = preprocess_input(x)
        images.append(ImageInfo(filename, full_filename, img))
    print(f"Loaded {len(images)} images")
    return images

def get_label(filename):
    if "-not-blink" in filename:
        return 0
    elif "-blink" in filename:
        return 1
    else:
        return None


def load_data_from_folder(folder):
    print(f"Loading data from folder: {folder}")
    X = []
    y = []
    for subfolder1 in os.listdir(folder):
        for subfolder in os.listdir(os.path.join(folder, subfolder1)):
            subfolder_path = os.path.join(folder, subfolder1, subfolder)
            print(f"Processing subfolder: {subfolder_path}")
            if os.path.isdir(subfolder_path):
                image_info = load_images_from_folder(subfolder_path)
                print(image_info)
                images = []
                for io in image_info:
                    label = get_label(io.filename)
                    print(f"Getting label for file: {io.filename}: {label}")
                    images.append(io.image)
                    if label is not None:
                        y.append(label)
                x = prepare_images_for_model(images)
                X.append(x)

                # X.append(image_info.image)
                # label = get_label(image_info.filename)
                # print(f"Getting label for file: {image_info.filename}: {label}")
                # y.append(label)

    print(f"Loaded {len(X)} images and {len(y)} labels")
    return np.array(X), np.array(y)

def train_model(folder):
    print(f"Training model with data from folder: {folder}")
    X, y = load_data_from_folder(folder)
    model = tf.keras.models.Sequential([
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'), input_shape=(tc.IMAGES_SHOWN_TO_MODEL, tc.EYE_IMAGE_WIDTH, tc.EYE_IMAGE_HEIGHT, 3)),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print(f"Model compiled {model.summary()}")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(f"Starting model training {X.shape} {y.shape}")
    model.fit(X, y, epochs=10)
    print("Model training completed.")
    model.save('trained_model.h5')
    print("Model saved to trained_model.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model on images in a folder.')
    parser.add_argument('folder', type=str, help='The folder containing the training images.')
    args = parser.parse_args()
    print(f"Starting training with folder: {args.folder}")
    train_model(args.folder)