import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox

# Load the pre-trained model
model = tf.keras.models.load_model("C:\\Users\\Sreerag\\Data Science\\Internship\\facial_recognition\\face_recognition_model.keras")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define emotion labels
emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprised']

# Function to predict emotions in a video frame
def predict_emotion(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray_frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=-1)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi)
        emotion_index = np.argmax(prediction[0])
        emotion = emotion_labels[emotion_index]

        # Draw rectangle around the face and display the emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    return frame

# Function to start video capture
def start_video_capture():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = predict_emotion(frame)
        cv2.imshow('Live Emotion Recognition', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to ask for permission to use the camera
def ask_permission():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    if messagebox.askyesno("Permission", "Do you allow access to the live video?"):
        start_video_capture()
    root.destroy()

# Main entry point
if __name__ == "__main__":
    ask_permission()
