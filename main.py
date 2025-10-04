import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle # Used for saving and loading encodings

# --- 1. Load or Compute and Save Encodings ---

ENCODINGS_FILE = 'encodings.pkl'
KNOWN_FACES_DIR = 'known_faces'

print("Loading and checking encodings...")

# Get the list of names of people from the directory
known_face_names_from_dir = [os.path.splitext(f)[0] for f in os.listdir(KNOWN_FACES_DIR)]

# Check if the encodings file exists and is up-to-date
encodings_are_valid = False
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        saved_data = pickle.load(f)
        # Check if the names in the saved file match the names from the directory
        if sorted(saved_data['names']) == sorted(known_face_names_from_dir):
            classNames = saved_data['names']
            encodeListKnown = saved_data['encodings']
            encodings_are_valid = True
            print("Encodings loaded from file. Ready to go!")

if not encodings_are_valid:
    print("Encodings file not found or out of date. Re-encoding faces...")
    classNames = []
    encodeListKnown = []
    images = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        img = cv2.imread(path)
        images.append(img)
        classNames.append(os.path.splitext(filename)[0])
        
        # Encode the face
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(rgb_img)[0]
            encodeListKnown.append(encode)
        except IndexError as e:
            print(f"Warning: No face found in {filename}, skipping.")

    # Save the new encodings to the file
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump({'names': classNames, 'encodings': encodeListKnown}, f)
    print("New encodings saved. Ready to go!")

# --- 2. Liveness Detection Helper Functions ---
def eye_aspect_ratio(eye):
    """Calculates the eye aspect ratio to determine if an eye is open."""
    eye = np.array(eye, dtype=np.float32)
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def is_blinking(landmarks):
    """Checks if a person is blinking based on eye aspect ratio."""
    EAR_THRESHOLD = 0.21
    leftEAR = eye_aspect_ratio(landmarks['left_eye'])
    rightEAR = eye_aspect_ratio(landmarks['right_eye'])
    ear = (leftEAR + rightEAR) / 2.0
    return ear < EAR_THRESHOLD

# --- 3. Attendance Logic ---
todays_attendees = set()

def markAttendance(name):
    """Marks attendance in a CSV file if not already marked today."""
    if name not in todays_attendees:
        with open('attendance.csv', 'a+') as f:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
            todays_attendees.add(name)
            print(f"✅ Attendance marked for {name}")

if not os.path.exists('attendance.csv'):
    with open('attendance.csv', 'w') as f:
        f.writelines('Name,Timestamp')

# --- 4. Main Webcam Loop ---
print("Starting webcam...")
cap = cv2.VideoCapture(0)
blink_detected_for_person = {}

while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame from webcam. Exiting...")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    landmarksCurFrame = face_recognition.face_landmarks(imgS, facesCurFrame)

    for encodeFace, faceLoc, landmarks in zip(encodesCurFrame, facesCurFrame, landmarksCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.5)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            
            if is_blinking(landmarks):
                blink_detected_for_person[name] = True

            if blink_detected_for_person.get(name, False):
                markAttendance(name)
                color = (0, 255, 0)
                label = name
            else:
                color = (31, 67, 153)
                label = f"{name} (Close Eyes for 2 sec)"
        else:
            color = (0, 0, 255)
            label = "UNKNOWN"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
        cv2.putText(img, label, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    cv2.imshow('Webcam Attendance', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()