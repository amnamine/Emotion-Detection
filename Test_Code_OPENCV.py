import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load pre-trained model and Haarcascade face detector
emotion_model = load_model('face.h5')  # Replace 'model.h5' with your model file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels (adjust based on your trained model's labels)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (Haarcascade works on grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw bounding box around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Preprocess the face for emotion detection
        face = gray_frame[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype('float') / 255.0  # Normalize pixel values
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)  # Expand dimensions to match model input

        # Predict emotion
        emotion_prediction = emotion_model.predict(face)[0]
        max_index = np.argmax(emotion_prediction)
        emotion = emotion_labels[max_index]

        # Display emotion label
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Display the video feed
    cv2.imshow('Real-Time Emotion Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
