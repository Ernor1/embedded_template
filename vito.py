import cv2
import sqlite3
import numpy as np
from keras.models import load_model

# Load pre-trained face recognition model
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load Haarcascade for face detection
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load sign detection model
signModel = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# Constants for display
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin = 30

nametagColor = (100, 180, 0)
nametagHeight = 50
faceRectangleBorderColor = nametagColor
faceRectangleBorderSize = 2

# Initialize face recognition counters
recognition_count = {}
REQUIRED_RECOGNITION_COUNT = 5

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
        customer_uid, confidence = faceRecognizer.predict(gray[y:y + h, x:x + w])
        
        # Connect to SQLite database
        try:
            conn = sqlite3.connect('customer_faces_data.db')
            c = conn.cursor()
        except sqlite3.Error as e:
            print("SQLite error:", e)

        c.execute("SELECT customer_name FROM customers WHERE customer_uid LIKE ?", (f"{customer_uid}%",))
        row = c.fetchone()
        if row:
            customer_name = row[0].split(" ")[0]
        else:
            customer_name = "Unknown"

        if 45 < confidence < 100:
            # Create rectangle around the face
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), faceRectangleBorderColor, faceRectangleBorderSize)

            # Display name tag
            cv2.rectangle(frame, (x - 22, y - nametagHeight), (x + w + 22, y - 22), nametagColor, -1)
            cv2.putText(frame, f"{customer_name}: {round(confidence, 2)}%", (x, y - fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)

            # Update recognition count
            if customer_uid not in recognition_count:
                recognition_count[customer_uid] = 0
            recognition_count[customer_uid] += 1

            # Check if the face has been recognized enough times
            if recognition_count[customer_uid] >= REQUIRED_RECOGNITION_COUNT:
                cv2.putText(frame, "Please make the 'OK' sign", (50, 50), fontFace, 1, (0, 0, 255), 2)
                
                # Sign detection logic
                while True:
                    ret, sign_image = camera.read()
                    if not ret:
                        break
                    
                    # Resize and normalize the sign image
                    sign_image_resized = cv2.resize(sign_image, (224, 224), interpolation=cv2.INTER_AREA)
                    sign_image_array = np.asarray(sign_image_resized, dtype=np.float32).reshape(1, 224, 224, 3)
                    sign_image_normalized = (sign_image_array / 127.5) - 1

                    # Predict the sign
                    prediction = signModel.predict(sign_image_normalized)
                    index = np.argmax(prediction)
                    class_name = class_names[index].strip()
                    confidence_score = prediction[0][index]

                    # Display the sign prediction and confidence score
                    cv2.putText(sign_image, f"Sign: {class_name[2:]} ({confidence_score*100:.2f}%)", (10, 30), fontFace, 1, (0, 255, 0), 2)
                    cv2.imshow("Sign Detection", sign_image)

                    # Check if the sign is "OK" and the confidence is above threshold
                    if class_name.lower() == "class 1" and confidence_score > 0.75:  # Adjust confidence threshold as needed
                        c.execute("INSERT INTO cart (customer_uid, customer_name) VALUES (?, ?)", (customer_uid, customer_name))
                        conn.commit()
                        print(f"{customer_name} added to database")
                        break
                    
                    # Exit sign detection loop on 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                cv2.destroyWindow("Sign Detection")

    # Display the resulting frame
    cv2.imshow('Detecting Faces...', frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
camera.release()
cv2.destroyAllWindows()

# Close the database connection
conn.close()
