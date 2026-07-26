# -*- coding: utf-8 -*-
"""This code is designed to run in Google Colab.
"""
#Install the Ultralytics package
!pip install ultralytics

# Place the detection model, named "model.zip" in your folder in Google drive
# Note: model.zip is the pre-trained model used for fish detection. 
#Therefore, it is directly used to detect fish in the input videos for subsequent behavioral analysis.

# Access Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Unzip detection model
!rm -rf /content/dataset
!unzip -q /content/drive/MyDrive/New/model.zip -d /content/dataset

# Install DeepSort
!pip install opencv-python-headless deep_sort_realtime

# =============================
#  ANALYZE FISH MOVEMENT
# =============================

import cv2
import math
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from google.colab.patches import cv2_imshow


# Initial configuaration

model = YOLO('/content/dataset/content/Deepfish_train/yolo2/weights/best.pt')  #  Call the model after unzip
tracker = DeepSort(
    max_age=30,
    n_init=10
)

# Confidence threshold for filtering low-confidence bounding boxes.
CONF_THRESH = 0.3

fish_positions = {}  # {id: [(x, y), (x, y), ...]}
fish_states = {}     # {id: {'velocity': v, 'direction': θ, 'state': 'slow'}}

SLOW_SPEED = 45.0   #pixel/s
FAST_SPEED = 90.0   #pixel/s
STILL_SPEED = 8     #pixel/s
TURN_ANGLE = 45     #degree

#Function for calculating velocity and direction.
def calculate_velocity_and_angle(pos_list, fps):
    if len(pos_list) < 2:
        return 0, 0
    (x1, y1), (x2, y2) = pos_list[-2], pos_list[-1]
    dx, dy = x2 - x1, y2 - y1
    distance = math.sqrt(dx**2 + dy**2)
    velocity = distance * fps  # pixel/giây
    angle = math.degrees(math.atan2(dy, dx))
    return velocity, angle
#Function for classify state of swimming
def classify_state(v, d_angle):
    if v < STILL_SPEED:
        return "Standing"
    elif abs(d_angle) >= TURN_ANGLE:
        return "Turning"
    elif v < SLOW_SPEED:
        return "Slow"
    elif v >= FAST_SPEED:
        return "Fast"
    else:
        return "Medium"

# Apply CLAHE
def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

# Create empty video for later processing
video_path = "/content/drive/MyDrive/<Name of Folder>/<Name of Video>.mp4"

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Output Video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/content/<Name of file>_output.mp4', fourcc, fps,
                      (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
print("Start to process video...")


# Apply CLAHE to enhance image

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/drive/MyDrive/<Name of Folder>/<Name of Video>.jpeg')

print(img.shape)
enhanced_img = apply_clahe(img)
plt.figure(figsize=(10,5))

plt.imshow(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
cv2.imwrite('image_clahe.jpg', enhanced_img)


# =======================================================================================
#  PROCESSING VIDEO TO VISUALIZE FISH SWIMMING and OUTPUT CSV File
# =======================================================================================

import pandas as pd
result_v_a = []
result_v_a_s =[]

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Step1: Enhance image by CLAHE
    enhanced_frame = apply_clahe(frame)

    # Step2:Apply YOLO
    results = model(enhanced_frame, verbose=False, conf=CONF_THRESH)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf)
        if conf < CONF_THRESH:
            continue
        detections.append([[x1, y1, x2 - x1, y2 - y1], conf, 'fish'])

    # Step 3: Apply DeepSORT
    tracks = tracker.update_tracks(detections, frame=enhanced_frame)

    # Step 4: Processing swimming states
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x, y, w, h = track.to_ltwh()
        cx, cy = int(x + w / 2), int(y + h / 2)

        fish_positions.setdefault(track_id, []).append((cx, cy))
        velocity, angle = calculate_velocity_and_angle(fish_positions[track_id], fps)

        result_v_a.append([track_id,velocity,angle])

        d_angle = angle - fish_states.get(track_id, {'direction': angle})['direction']
        state = classify_state(velocity, d_angle)

        fish_states[track_id] = {'velocity': velocity, 'direction': angle, 'state': state}

        # Generate random color  for each trackID
        try:
            np.random.seed(int(track_id))
        except:
            np.random.seed(sum(map(ord, str(track_id))))
        color = tuple(int(x) for x in np.random.randint(0, 255, 3))

        # Add information
        cv2.circle(frame, (cx, cy), 4, color, -1)
        cv2.putText(frame, f"ID:{track_id} {state} {velocity:.1f}px/s",
                    (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
         # Draw Trajectory
        if len(fish_positions[track_id]) > 1:
            for i in range(1, len(fish_positions[track_id])):
                cv2.line(frame, fish_positions[track_id][i - 1],
                         fish_positions[track_id][i], color, 2)

            # Keep 100 points
            if len(fish_positions[track_id]) > 100:
                fish_positions[track_id] = fish_positions[track_id][-100:]
    # Show 30 frams to avoid lag
    if frame_count % 50 == 0:
        cv2_imshow(frame)

    out.write(frame)

    df = pd.DataFrame(result_v_a,columns=["track_id", "velocity", "angle"])

    df.to_csv("Velocity_angle.csv", index= False)
cap.release()
out.release()

print(" Video is saved !")





# =============================
#  DRAW A DIRECTION OF SWIMMING
# =============================



result_v_a_s =[]
FPS = fps                 # fps video
WINDOW_SEC = 1            # 1 second
WINDOW = max(1, int(fps * WINDOW_SEC))
MIN_DISPLACEMENT = 10     # filter noise

import numpy as np
def compute_trend_vectors(positions, window):
    """
    positions: list[(x,y)]
    window: Numnber of frame in a window
    return: list[(cx, cy, dx, dy)]
    """
    vectors = []

    if len(positions) < window + 1:
        return vectors

    for i in range(0, len(positions) - window, window):
        p_start = np.array(positions[i])
        p_end   = np.array(positions[i + window])

        dx, dy = p_end - p_start
        dist = np.linalg.norm([dx, dy])

        if dist < MIN_DISPLACEMENT:
            continue

        cx, cy = (p_start + p_end) / 2
        vectors.append((cx, cy, dx, dy))

    return vectors
# Process frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Step 1: Apply CLAHE
    enhanced_frame = apply_clahe(frame)

    # Step 2: Apply CLAHE
    results = model(enhanced_frame, verbose=False, conf=CONF_THRESH)[0]
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf)
        if conf < CONF_THRESH:
            continue
        detections.append([[x1, y1, x2 - x1, y2 - y1], conf, 'fish'])

    # Step 3: Apply DeepSORT
    tracks = tracker.update_tracks(detections, frame=enhanced_frame)

    # Step 4: Processing state of swimming
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        x, y, w, h = track.to_ltwh()
        cx, cy = int(x + w / 2), int(y + h / 2)

        fish_positions.setdefault(track_id, []).append((cx, cy))
        velocity, angle = calculate_velocity_and_angle(fish_positions[track_id], fps)

        prev_angle = fish_states.get(track_id, {'direction': angle})['direction']
        d_angle = angle - prev_angle
        state = classify_state(velocity, d_angle)

        fish_states[track_id] = {
            'velocity': velocity,
            'direction': angle,
            'state': state
        }

        result_v_a_s.append([track_id, velocity, angle, state])

        # Track ID color
        try:
            np.random.seed(int(track_id))
        except:
            np.random.seed(sum(map(ord, str(track_id))))
        color = tuple(int(c) for c in np.random.randint(0, 255, 3))

        # --- Draw bbox + info ---
        cv2.rectangle(frame, (int(x), int(y)),
                      (int(x + w), int(y + h)), color, 2)

        cv2.circle(frame, (cx, cy), 4, color, -1)

        cv2.putText(frame,
                    f"ID:{track_id} {state} {velocity:.1f}px/s",
                    (int(x), int(y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1)

 
        #  Draw vector of swimming direction

        trend_vectors = compute_trend_vectors(
            fish_positions[track_id], WINDOW
        )

        for (vcx, vcy, dx, dy) in trend_vectors:
            cv2.arrowedLine(
                frame,
                (int(vcx), int(vcy)),
                (int(vcx + dx), int(vcy + dy)),
                color,
                2,
                tipLength=0.3
            )

        # Keep 100 points
        if len(fish_positions[track_id]) > 100:
            fish_positions[track_id] = fish_positions[track_id][-100:]

    # --- Visualize frame---
    if frame_count % 30 == 0:
        cv2_imshow(frame)

    out.write(frame)


# =============================
#   Save CSV
# =============================
df = pd.DataFrame(
    result_v_a_s,
    columns=["track_id", "velocity", "angle", "state"]
)

df.to_csv("Velocity_Angle_State.csv", index=False)

cap.release()
out.release()

print("Done.")


# =======================================================================================
#  Plot a rose diagram to visualize the frequency distribution of turning angles
# =======================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("/content/Velocity_Angle_State.csv")


angles_deg = df["angle"].dropna().values

angles_deg = (angles_deg + 180) % 360 - 180

angles_rad = np.deg2rad(angles_deg)

N_BINS = 16   # 16 hướng (mỗi hướng 22.5 độ)
bins = np.linspace(-np.pi, np.pi, N_BINS + 1)


hist, bin_edges = np.histogram(angles_rad, bins=bins)

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2


hist = hist / hist.sum()

plt.figure(figsize=(7, 7))
ax = plt.subplot(111, polar=True)

bars = ax.bar(
    bin_centers,
    hist,
    width=(2 * np.pi / N_BINS),
    bottom=0.0,
    align="center",
    edgecolor="black"
)

ax.set_theta_zero_location("E")
ax.set_theta_direction(-1)


ticks_deg = [0, 45, 90, 135, 180, 225, 270, 315]
tick_labels = ["0°","45°","90°","135°","180°","-45°","-90°","-135°"]
ax.set_thetagrids(ticks_deg, labels=tick_labels)
ax.set_yticklabels([])
ax.grid(True)
ax.set_title("Root zones", fontsize=12, pad=50)
plt.show()


# =======================================================================================
#  Generate the behavioral signature file.
# =======================================================================================

import pandas as pd
import numpy as np
from scipy.stats import variation


df = pd.read_csv("Velocity_Angle_State.csv")

df = df[df["velocity"] > 0]        # loại frame đầu
df = df.dropna(subset=["angle"])

mean_speed = df["velocity"].mean()
speed_cv = variation(df["velocity"])   # std / mean

# turning rate = % frame có đổi hướng mạnh
TURN_THRESHOLD = 45  # độ

angles = df["angle"].values
d_angles = np.abs(np.diff(angles))
turning_rate = np.mean(d_angles > TURN_THRESHOLD)

angles_rad = np.deg2rad(angles)

R = np.sqrt(
    (np.mean(np.cos(angles_rad)))**2 +
    (np.mean(np.sin(angles_rad)))**2
)
# R ∈ [0,1] : Closer to 1 → strong turning

immobility_ratio = np.mean(df["state"] == "Standing")
fast_ratio = np.mean(df["state"] == "Fast")

signature = pd.DataFrame([{
    "mean_speed": mean_speed,
    "speed_cv": speed_cv,
    "turning_rate": turning_rate,
    "directionality_R": R,
    "immobility_ratio": immobility_ratio,
    "fast_ratio": fast_ratio
}])

print(signature)

# Save
signature.to_csv("behavioral_signature.csv", index=False)


# =======================================================================================
#  CREATE MARKOV GRAPH
# =======================================================================================


import pandas as pd
import numpy as np
df = pd.read_csv("/content/Velocity_Angle_State.csv")
states = ["Standing", "Fast","Turning","Medium","Slow"]
state_to_idx = {s: i for i, s in enumerate(states)}
transition_counts = np.zeros((len(states),len(states)))
for track_id, group in df.groupby("track_id"):
    seq = group["state"].values
    for i in range(len(seq) - 1):
        from_state = seq[i]
        to_state = seq[i + 1]
        transition_counts[
            state_to_idx[from_state],
            state_to_idx[to_state]
        ] += 1


transition_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)


transition_df = pd.DataFrame(
    transition_matrix,
    index=states,
    columns=states
)

print(transition_df)

state_duration = df.groupby("state").size()
print(state_duration)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.heatmap(
    transition_df,
    annot=True,
    fmt=".2f",
    cmap="YlOrRd",
    cbar=True
)

plt.title("<Habitat>",fontsize=12, pad=30)
plt.xlabel("Next State")
plt.ylabel("Current State")
plt.tight_layout()
plt.show()

import networkx as nx
import matplotlib.pyplot as plt

# Tạo graph
G = nx.DiGraph()

for state in transition_df.index:
    G.add_node(state)

threshold = 0.05
for from_state in transition_df.index:
    for to_state in transition_df.columns:
        if from_state == to_state:
            continue

        prob = transition_df.loc[from_state, to_state]
        if prob >= threshold:
            G.add_edge(from_state, to_state, weight=prob)

pos = nx.circular_layout(G)

# Draw node + edge
edges = G.edges(data=True)
weights = [d["weight"] * 8 for (_, _, d) in edges]

plt.figure(figsize=(5, 5))
nx.draw(
    G, pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=11,
    width=weights,
    edge_color="gray",
    arrowsize=20,

)

# PROBABILITY (string labels)
edge_labels = {
    (u, v): f"{d['weight']:.2f}"
    for u, v, d in edges
}

# DRAW and Re-Detect Text objects
edge_texts = nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=10,
    label_pos=0.7
)

plt.title("Root zones",fontsize=12, pad=30)
plt.show()