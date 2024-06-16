import cv2

import sqlite3

# Load pre-trained face recognition model
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load Haarcascade for face detection
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2
fontBottomMargin = 30

nametagColor = (255, 0, 0)
nametagHeight = 50

faceRectangleBorderColor = nametagColor
faceRectangleBorderSize = 2

def run_face_detection():
    # Open a connection to the first webcam
    camera = cv2.VideoCapture(0)


    if not camera.isOpened():
        print("Error: Could not open video stream.")
        return

    # Start looping
    while True:
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # For each face found
        for (x, y, w, h) in faces:
            # Recognize the face
            ID, Confidence = faceRecognizer.predict(gray[y:y + h, x:x + w])
                    # Connect to SQLite database
            try:
                conn = sqlite3.connect('customer_faces_data.db')
                c = conn.cursor()
            #print("Successfully connected to the database")
            except sqlite3.Error as e:
                print("SQLite error:", e)

            # Confidence normalization to a 0-100 scale
            customer_name = "Unknown"
        

            if 50<Confidence <75 :
                try:
                    c.execute("SELECT customer_name FROM customers WHERE customer_uid = ?", (ID,))
                    print(ID)
                    row = c.fetchone()
                    if row:
                        print(row[0].split(" ")[0])
                        customer_name = row[0].split(" ")[0]
                    else:
                        customer_name = "Unknown"
                except sqlite3.Error as e:
                    print("SqliteError query error:", e)
                finally:
                    c.close()
                    conn.close()
                # Create rectangle around the face
                cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), faceRectangleBorderColor, faceRectangleBorderSize)
                print("confidence: ", Confidence)

                # Display name tag
                cv2.rectangle(frame, (x - 22, y - nametagHeight), (x + w + 22, y - 22), nametagColor, -1)
                cv2.putText(frame, f"{customer_name}: {round(Confidence, 2)}%", (x, y - fontBottomMargin), fontFace, fontScale, fontColor, fontWeight)
            else:
                cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (x, y - 10), fontFace, fontScale, (0, 0, 255), fontWeight)
                print("confidence: ", Confidence)
        # Display the resulting frame
        cv2.imshow('Detecting Faces...', frame)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        run_face_detection()