import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading

model = load_model("gender_classification_model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

gender_labels = {0: "Man", 1: "Woman"}

cap = cv2.VideoCapture(0)

frame_skip = 3 
frame_count = 0

gender_predictions = []
faces_coords = []

def predict_gender(faces):
    global gender_predictions
    gender_predictions = []
    
    for face in faces:
        face_resized = cv2.resize(face, (128, 128))
        face_resized = img_to_array(face_resized) / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        prediction = model.predict(face_resized)[0][0]
        gender_predictions.append("Man" if prediction < 0.5 else "Woman")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % frame_skip == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_coords = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))

        faces = [frame[y:y+h, x:x+w] for (x, y, w, h) in faces_coords]

        # Run gender prediction in a separate thread
        threading.Thread(target=predict_gender, args=(faces,)).start()

    for (box, gender) in zip(faces_coords, gender_predictions):
        x, y, w, h = box
        color = (0, 255, 0) if gender == "Man" else (255, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, gender, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()