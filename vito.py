import cv2
import sqlite3
import numpy as np
from keras.models import load_model

# Load pre-trained face recognition model
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load Haarcascade for face detection
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load the gesture recognition model
gesture_model = load_model("keras_Model.h5", compile=False)
class_names = [name.strip() for name in open("labels.txt", "r").readlines()]

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin = 30

knownTagColor = (100, 180, 0)
unknownTagColor = (0, 0, 255)
nametagHeight = 50

knownFaceRectangleColor = knownTagColor
unknownFaceRectangleColor = unknownTagColor
faceRectangleBorderSize = 2

# Open a connection to the first webcam
camera = cv2.VideoCapture(0)

# Start looping
while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each face found
    for (x, y, w, h) in faces:
        # Recognize the face
        customer_uid, Confidence = faceRecognizer.predict(gray[y:y + h, x:x + w])
        print(Confidence)
        
        # Connect to SQLite database

        # Set initial variables
        customer_name = "Unknown"
        faceRectangleColor = unknownFaceRectangleColor
        tagColor = unknownTagColor

        if 60 < Confidence < 85:  # Adjust confidence threshold as necessary
            try:
                conn = sqlite3.connect('customer_faces_data.db')
                c = conn.cursor()
                # print("Successfully connected to the database")
            except sqlite3.Error as e:
                print("SQLite error:", e)
                continue
            c.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (customer_uid,))
            row = c.fetchone()
            if row:
                customer_name = row[0].split(" ")[0]
                faceRectangleColor = knownFaceRectangleColor
                tagColor = knownTagColor

        # Create rectangle around the face
        cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), faceRectangleColor, faceRectangleBorderSize)

        # Display name tag
        cv2.rectangle(frame, (x - 22, y - nametagHeight), (x + w + 22, y - 22), tagColor, -1)
        cv2.putText(frame, f"{customer_name}", (x, y - fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)

        # Detect gesture
        gesture_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        gesture_image = np.asarray(gesture_image, dtype=np.float32).reshape(1, 224, 224, 3)
        gesture_image = (gesture_image / 127.5) - 1

        prediction = gesture_model.predict(gesture_image)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Debugging prints
        # print(f"Predicted gesture: {class_name}, Confidence: {confidence_score}")

        if "OK_SIGN" in class_name and confidence_score > 0.8:
            # Update the confirmation status in the database
            try:
                c.execute("UPDATE customers SET confirm = 1 WHERE customer_uid = ?", (customer_uid,))
                conn.commit()
                # print("Customer confirmed with OK gesture")
            except sqlite3.Error as e:
                print("SQLite error:", e)
            finally:
                conn.close()

    # Display the resulting frame
    cv2.imshow('Detecting Faces...', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()