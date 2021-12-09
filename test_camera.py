from keras.models import load_model
import cv2
import numpy as np
from mtcnn import MTCNN

# --- OpenCV settings ---
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# --- Initialize MTCNN faces detector ---
detector = MTCNN(min_face_size=200)


# --- Load my model ---
model = load_model("models/model2extra.h5")
model.load_weights("models/model2extra.epoch84-categorical_accuracy0.64.hdf5")
emotion_names = {
    0: "Angry",
    1: "Fear",
    2: "Happy",
    3: "Sad",
    4: "Surprise",
    5: "Neutral",
}
# model = load_model("models/model1.h5")
# model.load_weights("models/model1.epoch96-categorical_accuracy0.61.hdf5")
# emotion_names = {
#     0: "Angry",
#     1: "Disgust",
#     2: "Fear",
#     3: "Happy",
#     4: "Sad",
#     5: "Surprise",
#     6: "Neutral",
# }

sample_width, sample_height = 48, 48


# --- Load image from file ---
# image = Image.open("data/test.jpg")
# image = image.convert("RGB")
# pixels = np.asarray(image)


# --- Define emotion functions ---

font = cv2.FONT_HERSHEY_SIMPLEX
rect_color = (0, 0, 255)
text_color = rect_color


def draw_emotion(results, gray):

    faces = []
    emotions = []

    for r in results:
        v = r["box"]
        x, y, w, h = v[0], v[1], v[2], v[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), rect_color, 2)
        face = gray[y: y + h, x: x + w]

        try:
            face = cv2.resize(face, (48, 48))
            faces.append(face)
            face2 = np.array([face / 255.0])  # Feature Scaling in the same time

            # Use my model to predict emotion
            preds = model.predict(face2)
            result = emotion_names[preds[0].argmax()]
            emotions.append(result)

        except Exception as e:
            print(f"Exception: {e}")

    return faces, emotions


# --- Main loop --- (would be better with separate threads)

i = 0
gray = None
results = []

while True:

    ret, img = camera.read()
    i += 1

    # Search faces with MTCNN
    if i % 6 == 0:  # search faces every 6 loops
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = detector.detect_faces(img)
        i = 0

    # Draw last detected faces with OpenCV
    faces, emotions = draw_emotion(results, gray)

    # Display last detected faces' labels and preview
    if len(faces) > 0:

        for j, face in enumerate(faces):

            if j > 3:
                continue

            # Print gray faces
            factor = 2
            img2 = cv2.merge((faces[j], faces[j], faces[j]))
            resized = cv2.resize(
                img2, (48 * factor, 48 * factor), interpolation=cv2.INTER_AREA
            )

            pad_x = 10
            pad_y = 10 + (48 * factor * j)
            img[
                pad_y + 0: pad_y + (48 * factor), pad_x + 0: pad_x + (48 * factor)
            ] = resized

            # Print emotions labels
            cv2.putText(
                img,
                emotions[j],
                ((48 * factor) + pad_x + 10, int(pad_y + (resized.shape[0] / 2))),
                font,
                0.5,
                text_color,
                1,
                cv2.LINE_AA,
            )

    cv2.imshow("Live", img)
    if cv2.waitKey(5) != -1:
        break

# --- Release resources ---
camera.release()
cv2.destroyAllWindows()
