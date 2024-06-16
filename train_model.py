import cv2
import os
import numpy as np
from PIL import Image


def getImagesAndLabels(path):
    detector = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    faceSamples = []
    Ids = []
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            filename = os.path.basename(imagePath)
            Id = int(filename.split("_")[0].split(".")[1])
            faces = detector.detectMultiScale(imageNp)
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])
                Ids.append(Id)
                print(Id)
        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")
    return faceSamples, Ids

def Train():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Ids = getImagesAndLabels('./dataset')
    print(Ids)
    recognizer.train(faces, np.array(Ids))
    recognizer.save("models/trained_lbph_face_recognizer_model.yml")
    print("Done Training")

if __name__ == "__main__":
        Train()
