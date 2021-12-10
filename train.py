#!/usr/bin/env python
# coding: utf-8

# Facial Expression Recognition (FER)
# This project is the Capstone project for the ML-bookcamp (with Alexey Grigorev).
#
# The dataset come from an old Kaggle competition:
# https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

import argparse
import numpy as np
import pandas as pd

from tensorflow import keras
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    # Activation,
)
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tqdm.keras import TqdmCallback

import matplotlib.pyplot as plt


emotion_names = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


# --- PREPARE ---


def load_prepare_data(in_name):

    print(f">>> LOAD AND PREPARE DATA FROM {in_name}")

    # Load data
    data = pd.read_csv(in_name)
    data.columns = data.columns.str.replace(" ", "_").str.lower()

    # > The `Disgust` value seem under-represented. The dataset is clearly imbalanced.
    # > So we will need to use an appropriate solution *(use upsampling and downsampling techniques)* or metric.

    data = data[data["emotion"] != 1]

    # --- Encode target variable labels ---
    print(f">>> ENCODE TARGET VARIABLE")

    le = LabelEncoder()
    img_labels = le.fit_transform(data["emotion"])
    img_labels = np_utils.to_categorical(img_labels)
    img_labels.shape

    # --- Convert `pixels` input `strings` to `arrays` ---
    # /!\ The process takes a lot of ressources, it may slow down your computer.
    print(">>> CONVERT STRING IMAGES TO ARRAYS")

    img_array = data["pixels"].apply(
        lambda x: np.array(x.split(" ")).reshape(48, 48, 1).astype("float32")
    )
    img_array = np.stack(img_array, axis=0)

    # --- Split the dataset into Train, Validation & Test sets ---
    print(">>> SPLIT THE DATASET")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        img_array, img_labels, test_size=0.2
    )  # , random_state=42)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_full, y_train_full, test_size=X_test.shape[0]
    )  # , random_state=42)

    assert X_valid.shape[0] == X_test.shape[0]
    assert img_array.shape[0] == X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]

    # --- Check some variables used in the model architecture ---

    img_width = X_train.shape[1]
    img_height = X_train.shape[2]
    img_depth = X_train.shape[3]
    num_classes = y_train.shape[1]

    img_width, img_height, img_depth, num_classes

    assert img_width == 48
    assert img_height == 48
    assert img_depth == 1
    assert num_classes == 6

    # --- Setup the ImageDataGenerators with the selected transformations ---
    print(">>> SETUP IMAGEDATAGENERATORS")

    gen_batch_size = 64

    train_datagen_extra = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    valid_datagen_extra = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )

    train_ds_extra = train_datagen_extra.flow(
        x=X_train,
        y=y_train,
        batch_size=gen_batch_size,
        shuffle=True,
    )

    valid_ds_extra = valid_datagen_extra.flow(
        x=X_valid,
        y=y_valid,
        batch_size=gen_batch_size,
        shuffle=True,
    )

    # Define  Convolutional Neural Network - Architecture 2
    # from CNN architecture from: https://github.com/gitshanks/fer2013
    print(">>> DEFINE CNN ARCHITECTURE")

    num_features = 64

    model = Sequential()

    # 1st convolution
    model.add(
        Conv2D(
            num_features,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(img_width, img_height, img_depth),
            data_format="channels_last",
            kernel_regularizer=l2(0.01),
        )
    )
    model.add(
        Conv2D(num_features, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 2nd convolution
    model.add(
        Conv2D(2 * num_features, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(2 * num_features, kernel_size=(3, 3), activation="relu", padding="same")
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 3rd convolution
    model.add(
        Conv2D(
            2 * 2 * num_features, kernel_size=(3, 3), activation="relu", padding="same"
        )
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            2 * 2 * num_features, kernel_size=(3, 3), activation="relu", padding="same"
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # 4rd convolution
    model.add(
        Conv2D(
            2 * 2 * 2 * num_features,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(
        Conv2D(
            2 * 2 * 2 * num_features,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    # Flatten layer
    model.add(Flatten())

    # 1st fully connected layer
    model.add(Dense(2 * 2 * 2 * num_features, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(2 * 2 * num_features, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(2 * num_features, activation="relu"))
    model.add(Dropout(0.5))

    # output for up to 7 expressions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
    model.add(Dense(num_classes, activation="softmax"))

    return model, train_ds_extra, valid_ds_extra


def train_model(
            model,
            out_name,
            train_ds,
            valid_ds,
            epochs=1,
            batch_size=64,
            steps_per_epoch=100,
            validation_steps=50):

    def rocauc(yTrue, yPred):
        auc = roc_auc_score(yTrue, yPred)
        return auc[0]

    # Define Optimizer & Loss

    f_opti = keras.optimizers.Adam(learning_rate=0.0005)
    f_loss = keras.losses.CategoricalCrossentropy(from_logits=False)

    # Define savepoints
    filepath = (
        out_name
        + ".epoch{epoch:02d}-categorical_accuracy{val_categorical_accuracy:.2f}.hdf5"
    )

    checkpoint = ModelCheckpoint(
        filepath=filepath,
        monitor="val_categorical_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    # Compile model
    model.compile(
        optimizer=f_opti,
        loss=f_loss,
        metrics=[keras.metrics.CategoricalAccuracy(), "accuracy"],
    )

    # Save model
    model.save(out_name + ".h5")

    # Fit model
    history = model.fit(
        train_ds,
        batch_size=batch_size,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_ds,
        validation_steps=validation_steps,
        callbacks=[TqdmCallback(), checkpoint],
        verbose=1,
    )

    # Save weights
    model_json = model.to_json()
    with open(out_name + ".json", "w") as yaml_file:
        yaml_file.write(model_json)

    return history


# --- Plot the models' results ---


def summarize_diagnostics(history, out_name):

    # figure = plt.figure(figsize=(8, 8))

    # plot loss
    plt.subplot(211)
    plt.title("Cross Entropy Loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="orange", label="val")
    plt.legend()

    # plot accuracy
    plt.subplot(212)
    plt.title("Classification Accuracy")
    plt.plot(history.history["categorical_accuracy"], color="blue", label="train")
    plt.plot(history.history["val_categorical_accuracy"], color="orange", label="val")
    plt.legend()

    plt.tight_layout(pad=1.0)

    # save plot to file
    plt.savefig(out_name + "_plot.png")
    plt.close()


# --- MAIN FUNCTION ---


if __name__ == "__main__":

    # Initialize arguments parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, help="The path to the .csv file")
    parser.add_argument("-d", "--destination", type=str, help="The path to model file")
    parser.add_argument("-b", "--batch_size", type=str, help="The batch_size of the generators and model")
    parser.add_argument("-e", "--epochs", type=str, help="The number of epochs trained")
    parser.add_argument("-se", "--steps_per_epoch", type=str, help="The steps_per_epoch of the model")
    parser.add_argument("-vs", "--validation_steps", type=str, help="The validation_steps of the model")
    args = parser.parse_args()

    # Initialize in/out variables
    source = "data/fer2013.csv"
    destination = "models/model"
    batch_size = 64
    epochs = 1
    steps_per_epoch = 100
    validation_steps = 50

    if args.source is not None:
        source = args.source
    if args.destination is not None:
        destination = args.destination
    if args.batch_size is not None:
        batch_size = int(args.batch_size)
    if args.epochs is not None:
        epochs = int(args.epochs)
    if args.steps_per_epoch is not None:
        steps_per_epoch = int(args.steps_per_epoch)
    if args.validation_steps is not None:
        validation_steps = int(args.validation_steps)

    # Train / Evaluate / Save
    print(">>> LET'S TRAIN A NEW MODEL")
    model, train_ds, valid_ds = load_prepare_data(source)
    history = train_model(
            model,
            destination,
            train_ds,
            valid_ds,
            epochs,
            batch_size,
            steps_per_epoch,
            validation_steps
    )
    summarize_diagnostics(history, "rocauc")
    print(">>> MODEL TRAINING COMPLETE")
