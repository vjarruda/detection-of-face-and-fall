import cv2
import face_recognition
import mediapipe as mp
import numpy as np
import os
import pickle

known_face_encodings = []
known_face_names = []

for file in os.listdir():
    if file.endswith(".pkl"):
        with open(file, "rb") as f:
            encoding = pickle.load(f)
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(file)[0])

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_capture = cv2.VideoCapture(0)

fall_detected = False
fall_counter = 0
fall_threshold = 5 

print("Reconhecimento facial e detecção de quedas iniciados. Pressione 'q' para sair.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    results = pose.process(rgb_frame)

    queda_confirmada = False 

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

        shoulder_to_hip_diff = np.abs(left_shoulder.y - left_hip.y)
        knee_to_ankle_diff = np.abs(left_knee.y - left_ankle.y)

        if shoulder_to_hip_diff > 0.15 and knee_to_ankle_diff > 0.15:
            fall_counter += 1
        else:
            fall_counter = 0

        if fall_counter >= fall_threshold:
            fall_message = "Sem Queda Detectada"
        else:
            fall_message = "Queda detectada"
            queda_confirmada = True

        cv2.putText(frame, fall_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255) if queda_confirmada else (0, 255, 0), 2)
        
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconhecido"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        if queda_confirmada and name != "Desconhecido":
            print(f"{name} teve uma queda detectada!")

    cv2.imshow("Reconhecimento Facial e Detecção de Quedas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
