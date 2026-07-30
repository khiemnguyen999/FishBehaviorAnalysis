# -*- coding: utf-8 -*-
"""This code is for analyzing fish swimming in body length/second.
"" We suggest run this code in Google Colab
"""
#Install the Ultralytics package
!pip install ultralytics

#Place the detection model, named "model.zip" in your folder in Google drive
# Note: model.zip is the pre-trained model used for fish detection. 
#Therefore, it is directly used to detect fish in the input videos for subsequent behavioral analysis.

from google.colab import drive
drive.mount('/content/drive')

!rm -rf /content/dataset
!unzip -q /content/drive/MyDrive/New/model.zip -d /content/dataset

!pip install opencv-python-headless deep_sort_realtime

import cv2
import math
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from google.colab.patches import cv2_imshow

# =============================
#  1. Config
# =============================
model = YOLO('/content/dataset/content/Deepfish_train/yolo2/weights/best.pt')  # Call the model after unzip
tracker = DeepSort(
    max_age=30,
    n_init=10
)

# Confidence threshold for filtering low-confidence bounding boxes.
CONF_THRESH = 0.3

fish_positions = {}  # {id: [(x, y, body_length), ...]}
fish_states = {}     # {id: {'velocity': v, 'direction': θ, 'state': 'slow'}}

STILL_SPEED = 0.03   # can be modified for suitable value
SLOW_SPEED  = 0.13   # can be modified for suitable value
FAST_SPEED  = 0.25   # can be modified for suitable value
TURN_ANGLE  = 45     # degree

# =============================
#  2. Calculate the velocity and direction by Body lenght
# =============================
def calculate_velocity_and_angle(pos_list, fps):
    if len(pos_list) < 2:
        return 0, 0
    x1, y1, bl1 = pos_list[-2]
    x2, y2, bl2 = pos_list[-1]
    dx = x2 - x1
    dy = y2 - y1
    distance = math.sqrt(dx**2 + dy**2)
    # Body Length averagely between 2 frames
    body_length = (bl1 + bl2) / 2
    if body_length < 1:
        body_length = 1
    # Body Length per second
    velocity = (distance / body_length) * fps
    angle = math.degrees(math.atan2(dy, dx))
    return velocity, angle

def classify_state(v, d_angle):
    if v < STILL_SPEED:
        return "Standing"
    elif abs(d_angle) >= TURN_ANGLE:
        return "Turning"
    elif v < SLOW_SPEED:
        return "Slow"
    elif v < FAST_SPEED:
        return "Medium"
    else:
        return "Fast"

# =============================
#  3. Apply CLAHE
# =============================
def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

# =======================================================================================
#  Create Video
# =======================================================================================
# Give an available video first
video_path = "/content/drive/MyDrive/<Name of Folder>/<Name of Video>.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Output empty video for later processing
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/<Name of Video>_output.mp4', fourcc, fps,
                      (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
print("Start to process video...")

import pandas as pd
result_v_a =[]

#Approach 1
result_v_a = []  #
body_lengths = []#

result_v_a_s =[]
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Step 1:  Apply CLAHE
    enhanced_frame = apply_clahe(frame)

    # Step 2: Apply YOLO
    results = model(enhanced_frame, verbose=False, conf=CONF_THRESH)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf)
        if conf < CONF_THRESH:
            continue
        detections.append([[x1, y1, x2 - x1, y2 - y1], conf, 'fish'])

    # Step 3: Update DeepSORT
    tracks = tracker.update_tracks(detections, frame=enhanced_frame)

    # Step 4: Processing state
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x, y, w, h = track.to_ltwh()
        cx, cy = int(x + w / 2), int(y + h / 2)

        body_length = float(max(w, h))
        body_lengths.append(body_length)  #

        fish_positions.setdefault(track_id, []).append((cx, cy, body_length)) # Save body length

        velocity, angle = calculate_velocity_and_angle(fish_positions[track_id], fps)

        result_v_a.append([track_id,velocity,angle])

        d_angle = angle - fish_states.get(track_id, {'direction': angle})['direction']
        state = classify_state(velocity, d_angle)

        fish_states[track_id] = {'velocity': velocity, 'direction': angle, 'state': state}

        result_v_a_s.append([track_id, velocity, angle, state])

        # Màu ngẫu nhiên theo ID (fix lỗi TypeError)
        try:
            np.random.seed(int(track_id))
        except:
            np.random.seed(sum(map(ord, str(track_id))))
        color = tuple(int(x) for x in np.random.randint(0, 255, 3))

        # Add information 
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(frame, f"ID:{track_id} {state} {velocity:.2f} BL/s",
                    (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
         # Draw trajectory
        if len(fish_positions[track_id]) > 1:
            for i in range(1, len(fish_positions[track_id])):
                p1 = fish_positions[track_id][i-1]
                p2 = fish_positions[track_id][i]
                cv2.line(
                    frame,
                    (int(p1[0]), int(p1[1])),
                    (int(p2[0]), int(p2[1])),
                    color,
                    2
                )
            if len(fish_positions[track_id]) > 100:
                fish_positions[track_id] = fish_positions[track_id][-100:]
    if frame_count % 50 == 0:
        cv2_imshow(frame)
    out.write(frame)
cap.release()
out.release()
# Save CSV
df = pd.DataFrame(
    result_v_a_s,
    columns=["track_id", "velocity", "angle", "state"]
)
df.to_csv("Velocity_Angle_State.csv", index=False)
