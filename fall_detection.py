import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

video_capture = cv2.VideoCapture(0)

fall_detected = False
fall_counter = 0
fall_threshold = 5 

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    #conversao mediapipe
    rgb_frame = frame[:, :, ::-1]

    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]

        shoulder_to_hip_angle = np.abs(left_shoulder.y - left_hip.y)

        knee_to_ankle_angle = np.abs(left_knee.y - left_ankle.y)

        if shoulder_to_hip_angle > 0.15 and knee_to_ankle_angle > 0.15:
            fall_counter += 1
            fall_detected = True
        else:
            fall_counter = 0 
            fall_detected = False

        if fall_counter >= fall_threshold:
            cv2.putText(frame, "Sem queda detectada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Queda detectada!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Sistema de Detecção de Quedas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
