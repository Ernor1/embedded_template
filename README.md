
# Face and Sign Recognition System

## Overview
This application detects and recognizes customers' faces, ensuring that the face is identified correctly at least 5 times. After successful recognition, it prompts the customer to make an "OK" sign for further verification. Once verified, the customer’s information is added to a database.

## Features
- **Face Detection and Recognition**: Uses a pre-trained LBPH face recognizer and Haar Cascade classifier for face detection.
- **Sign Detection**: Uses a Keras model to recognize hand signs, specifically the "OK" sign.
- **Database Integration**: Stores customer information in an SQLite database after successful verification.

## Requirements
- Python 3.11.2 !important
- OpenCV
- Keras (with TensorFlow backend)
- NumPy
- SQLite3

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/your-repository/face-sign-recognition.git
    cd face-sign-recognition
    ```

2. **Install Dependencies**
    ```bash
    pip install opencv-python opencv-contrib-python tensorflow==2.12.0 keras==2.12.0 numpy
    ```

**N.B:** If you already installed TensorFlow and Keras and you are having trouble loading the Teachable Machine model, uninstall Keras and TensorFlow and install the indicated versions:

```bash
pip uninstall keras tensorflow
```

```bash
pip install tensorflow==2.12.0 keras==2.12.0 numpy
```

3. **Prepare the Models and Database**
    - **Face Recognition Model**: Ensure you have the pre-trained `trained_lbph_face_recognizer_model.yml` file in the `models/` directory.
    - **Haar Cascade Classifier**: Ensure you have `haarcascade_frontalface_default.xml` in the `models/` directory.
    - **Sign Detection Model**: Ensure you have the Keras model file `keras_Model.h5` and `labels.txt` in the root directory.
    - **SQLite Database**: Ensure you have `customer_faces_data.db` with a `customers` table, and optionally a `cart` table.

## Usage

1. **Run the Application**
    ```bash
    python face_sign_recognition.py
    ```

2. **Functionality**
    - The application will open the webcam and start detecting faces.
    - Each detected face will be recognized, and if it matches a known customer, the name and confidence score will be displayed.
    - After recognizing a face 5 times, the application will prompt the customer to make the "OK" sign.
    - If the "OK" sign is detected with sufficient confidence, the customer's information will be added to the database.

3. **Exiting the Application**
    - Press 'q' to quit the face detection loop.
    - Press 'q' again during sign detection to exit.

## File Structure

```
face-sign-recognition/
│
├── models/
│   ├── trained_lbph_face_recognizer_model.yml
│   ├── haarcascade_frontalface_default.xml
│
├── customer_faces_data.db
├── keras_Model.h5
├── labels.txt
├── face_sign_recognition.py
└── README.md
```

## Database Schema

- **customers** table:
  ```sql
  CREATE TABLE customers (
      customer_uid INTEGER PRIMARY KEY,
      customer_name TEXT
  );
  ```
- **cart** table (optional):
  ```sql
  CREATE TABLE cart (
      customer_uid INTEGER,
      customer_name TEXT,
      FOREIGN KEY (customer_uid) REFERENCES customers (customer_uid)
  );
  ```

## Notes
- Adjust the confidence thresholds in the code based on your model's accuracy and performance.
- Ensure the `labels.txt` file includes the "OK" sign and matches the output format expected by the Keras model.
- This application requires a webcam to function correctly.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

By following the instructions provided in this README, you should be able to set up and run the face and sign recognition system effectively. If you encounter any issues or have questions, please refer to the project's GitHub repository for further information and support.
