#! /usr/bin/env python3
# coding: utf-8

import os
from flask import Flask, request, jsonify

import cv2
import tflite_runtime.interpreter as tflite
import numpy as np

app = Flask(__name__)

# --- Load TF Model ---
# print("Load Classification Model")
# model = load_model("models/model1extra.h5")
# model.load_weights("models/model1extra.epoch119-categorical_accuracy0.62.hdf5")

# --- Load TF-Lite model using an interpreter
interpreter = tflite.Interpreter(model_path="models/model1extra.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

emotion_names = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Sad",
    4: "Surprise",
    5: "Neutral",
}


@app.route("/")
def index():
    return f"Hello world !<br>The 'Facial Expression Recognotion' server is up."


@app.route("/predict", methods=["POST"])
def predict():

    try:
        face = np.fromfile(request.files["media"], np.uint8)
        print("IMAGE")

        # convert numpy array to image
        img = cv2.imdecode(face, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (48, 48))
        print(img.shape)
        print(gray.shape)

    except Exception:

        try:
            gray = np.array(request.get_json().split(" "))
            print("STRING")
        except Exception:
            pass

    # Process string to array
    gray = gray.reshape(48, 48, 1).astype("float32")
    face2 = gray / 255.0

    # Apply model
    interpreter.set_tensor(input_index, [face2])
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    label = emotion_names[preds[0].argmax()]

    # Return values
    preds_labels = [x for x in zip(emotion_names.values(), *preds)]

    return jsonify(
        f"The predicted label is: {label}\n\nwith the following probabilities: {preds_labels}\n"
    )


print("Server ready")

if __name__ == "__main__":
    current_port = int(os.environ.get("PORT") or 5000)
    app.run(debug=True, host="0.0.0.0", port=current_port)
