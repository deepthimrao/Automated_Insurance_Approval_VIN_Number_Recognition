import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
import visualkeras

from difflib import SequenceMatcher

SEED = 45632
np.random.seed(SEED)
tf.random.set_seed(SEED)
GPU_ID = 0

gpus = tf.config.list_physical_devices('GPU')
if(gpus):
    tf.config.set_visible_devices(gpus[GPU_ID], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[GPU_ID], True)

data_dir = Path("./niss_vin_imgs_edge_preprocess_900_100/") #path to input data

images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]
labels = [l.split("_")[0] for l in labels]
# print(labels)
characters = set(char for label in labels for char in label)
characters = sorted(list(characters))

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

characters = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
# Batch size for training and validation
batch_size = 8

# Desired image dimensions
img_width = 900
img_height = 100

max_length = max([len(label) for label in labels])
print("Max length = ", max_length)

# Mapping characters to integers
char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


# Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))


def encode_single_sample(img_path, label):
    # Read image
    img = tf.io.read_file(img_path)
    # Decode and convert to grayscale
    # img = tf.io.decode_png(img, channels=1)
    img = tf.io.decode_jpeg(img, channels=1)
    # Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.transpose(img, perm=[1, 0, 2])
    # Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    label=tf.one_hot(tf.cast(label, tf.int64), len(characters))
    label=tf.split(label,17)
    labels={}
    for i in range(len(label)):
        labels["char"+str(i)]=tf.squeeze(label[i])
        labels["char"+str(i)].set_shape([36])

    return img, labels


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


# function to build model
def build_model_2():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), dtype="float32"
    )

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Third conv block
    x = layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv3",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    out = [layers.Dense(36, name="char%d" % i, activation="softmax")(x) for i in range(17)]
    # out = layers.Dense(17, name="dense2", activation="softmax")(x)

    model = keras.models.Model(
        inputs=input_img, outputs=out, name="ocr_model_v1"
    )

    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=['accuracy'])
    return model

# Get the model
model = build_model_2()
model.summary()
model.save("./models/model_2_nissan_8_900_100_edge_cnn.h5") #only for visualization in netron

epochs = 200

# define callbacks
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath="./models/ckpt_2_nissan_8_900_100_edge_cnn/",
    monitor="val_acc",
    mode="max",
    save_best_only=True,
    save_weights_only=True
)

tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./models_logs/logs_2_nissan_8_900_100_edge_cnn/")

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[checkpoint_callback, tensorboard_callback],
    batch_size=batch_size
)

# save model
model.save("E:/OMSA/Practicum/Assurant/models/model_2_nissan_8_900_100_edge_cnn")
