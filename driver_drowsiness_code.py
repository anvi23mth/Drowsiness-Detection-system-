"""
Driver Drowsiness Detection using CNN Eye State Classifier
- Uses MediaPipe to detect eyes
- CNN predicts eye state (open/closed)
- If eyes closed too long or blink rate drops, triggers alarm
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import threading
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- Config ----------------
MODEL_PATH = "eye_state_cnn.h5"   # pre-trained CNN model
IMG_SIZE = 34                     # input size expected by CNN (34x34 typical for eye datasets)

LONG_FRAME_THRESHOLD = 40         # closed-eye frames threshold
BLINKS_PER_MIN_THRESHOLD = 8      # blinks/min threshold
MIN_BEEP_INTERVAL = 5             # seconds between alarms
# ----------------------------------------

# Load CNN model
model = load_model(MODEL_PATH)

# MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

def crop_eye(frame, landmarks, eye_idx, margin=5):
    h, w = frame.shape[:2]
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx])
    x, y, w_eye, h_eye = cv2.boundingRect(pts)
    x = max(0, x - margin)
    y = max(0, y - margin)
    return frame[y:y+h_eye+margin, x:x+w_eye+margin]

def preprocess_eye(eye_img):
    eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    eye_resized = cv2.resize(eye_gray, (IMG_SIZE, IMG_SIZE))
    eye_norm = eye_resized.astype("float32") / 255.0
    return np.expand_dims(eye_norm, axis=(0, -1))  # shape (1, IMG_SIZE, IMG_SIZE, 1)

def play_alarm():
    os.system("say 'You are drowsy'")

def main():
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15

    closed_frame_counter = 0
    blink_timestamps = deque()
    last_alarm_time = 0

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # Crop both eyes
            left_eye_img = crop_eye(frame, lm, LEFT_EYE_IDX)
            right_eye_img = crop_eye(frame, lm, RIGHT_EYE_IDX)

            state_left, state_right = 0, 0  # 0=open, 1=closed

            if left_eye_img.size > 0:
                pred = model.predict(preprocess_eye(left_eye_img), verbose=0)
                state_left = int(pred[0][0] > 0.5)
            if right_eye_img.size > 0:
                pred = model.predict(preprocess_eye(right_eye_img), verbose=0)
                state_right = int(pred[0][0] > 0.5)

            both_closed = (state_left == 1 and state_right == 1)

            if both_closed:
                closed_frame_counter += 1
            else:
                if 1 <= closed_frame_counter <= 25:
                    blink_timestamps.append(time.time())
                closed_frame_counter = 0

            # Blink count (last 60s)
            now = time.time()
            while blink_timestamps and (now - blink_timestamps[0] > 60):
                blink_timestamps.popleft()
            blinks_per_min = len(blink_timestamps)

            # Display info
            cv2.putText(frame, f"Blinks/min: {blinks_per_min}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if both_closed:
                cv2.putText(frame, "Eyes CLOSED", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            else:
                cv2.putText(frame, "Eyes OPEN", (10,60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # Drowsiness logic
            if closed_frame_counter >= LONG_FRAME_THRESHOLD or blinks_per_min < BLINKS_PER_MIN_THRESHOLD:
                cv2.putText(frame, "YOU ARE DROWSY!", (10,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                if now - last_alarm_time > MIN_BEEP_INTERVAL:
                    last_alarm_time = now
                    threading.Thread(target=play_alarm, daemon=True).start()

        cv2.imshow("Drowsiness Detection (CNN)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
