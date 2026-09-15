import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = "../models/final_gesture_model.keras"
IMG_SIZE = (128, 128)
CLASS_NAMES = [f"Gesture_{i}" for i in range(14)]

model = tf.keras.models.load_model(MODEL_PATH)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(rgb, IMG_SIZE)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0]
    class_id = int(np.argmax(prediction))
    confidence = float(np.max(prediction))

    text = f"{CLASS_NAMES[class_id]} ({confidence * 100:.2f}%)"
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
