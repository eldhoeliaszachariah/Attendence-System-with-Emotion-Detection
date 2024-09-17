#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from keras.models import model_from_json
import pandas as pd

# Load the emotion detection model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(model_json)
emotion_model.load_weights("facialemotionmodel.h5")

# Load the attendance prediction model
json_file_attendence = open("attendencelast2.json", "r")
model_json_attendence = json_file_attendence.read()
json_file_attendence.close()
attendence_model = model_from_json(model_json_attendence)
attendence_model.load_weights("attendencelast2.h5")

# Load and encode face images from the training dataset
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to encode faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Attendance data list
attendance_data = []
marked_students = set()  # Set to keep track of students already marked present

# Function to mark attendance
def markAttendance(name, emotion):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    attendance_data.append([name, 'Present', emotion, dtString])
    marked_students.add(name)

# Load known face encodings
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Time range for attendance
start_time = datetime.strptime("09:30", "%H:%M").time()
end_time = datetime.strptime("10:00", "%H:%M").time()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Labels for the attendance classification model (assuming 5 classes)
attendence_labels = {0: 'Dileep (Student 1)', 1: 'Jayaram (Student 2)', 2: 'Mammotty (Student 3)', 3: 'Mohanlal (Student 4)', 4: 'Prithiraj (Student 5)'}

while True:
    current_time = datetime.now().time()

    # Check if the current time is within the attendance time range
    if not (start_time <= current_time <= end_time):
        print("Attendance system is only operational between 9:30 am and 10:00 am.")
        break

    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # Validate the face region before resizing
            if y1 >= 0 and y2 <= img.shape[0] and x1 >= 0 and x2 <= img.shape[1]:
                face = img[y1:y2, x1:x2]
                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, (48, 48))
                face_resized = face_resized.reshape(1, 48, 48, 1) / 255.0

                # Predict emotion
                emotion_prediction = emotion_model.predict(face_resized)
                emotion_label = emotion_labels[np.argmax(emotion_prediction)]

                # Predict class using attendance model
                attendance_prediction = attendence_model.predict(face_resized)
                predicted_class = attendence_labels[np.argmax(attendance_prediction)]

                # Draw rectangle and text around the recognized face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{name} ", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, emotion_label, (x1 + 6, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                # Mark attendance with the detected emotion only if not already marked
                if name not in marked_students:
                    markAttendance(name, emotion_label)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After 10:00 am, mark absent for students not detected
if current_time > end_time:
    for name in classNames:
        name = name.upper()
        if name not in marked_students:
            attendance_data.append([name, 'Absent', 'None', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])

cap.release()
cv2.destroyAllWindows()

# Save the attendance data to an Excel file
df = pd.DataFrame(attendance_data, columns=['Student Class', 'Attendance', 'Emotion', 'Time'])
df.to_excel('attendance_data2.xlsx', index=False)


# In[ ]:




