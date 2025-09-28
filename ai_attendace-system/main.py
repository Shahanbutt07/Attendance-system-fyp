import cv2
import numpy as np
import json
from deepface import DeepFace
from ultralytics import YOLO
import datetime
import time

# ==== CONFIG ====
YOLO_MODEL_PATH = "/Users/a1/Desktop/shahan_fyp_backend/model.pt"
EMBEDDINGS_PATH = "/Users/a1/Desktop/shahan_fyp_backend/embeddings_facenet.json"

# Manual student list (no CSV)
students = [
    {"Roll_No": "F2021266031", "Name": "Shahan", "Department": "CS", "Status": "Absent"},
    {"Roll_No": "F2021266600", "Name": "Haroon", "Department": "CS", "Status": "Absent"},
]

# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

# Load embeddings
with open(EMBEDDINGS_PATH, "r") as f:
    stored_embeddings = json.load(f)
stored_embeddings = {name: np.array(vec).astype("float32")[:128] for name, vec in stored_embeddings.items()}

# Normalize stored embeddings
for k in stored_embeddings:
    stored_embeddings[k] = stored_embeddings[k] / np.linalg.norm(stored_embeddings[k])

# Threshold for recognition
THRESHOLD = 1.1

# ✅ Track detection start times
detection_times = {s["Name"]: None for s in students}
CONFIRMATION_TIME = 5  # seconds

# ✅ Track first detection timestamp
first_detect_time = None
AUTO_STOP_TIME = 5  # seconds after first known face detection

# Initialize webcam
cap = cv2.VideoCapture(0)
print("✅ Webcam started. Press 'q' to quit automatically.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detected_names = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        try:
            rep = DeepFace.represent(face_crop, model_name="Facenet",
                                     enforce_detection=False,
                                     detector_backend="skip")[0]["embedding"]
            rep = np.array(rep).astype("float32")[:128]
            rep = rep / np.linalg.norm(rep)

            best_match = "Unknown"
            best_dist = float("inf")

            for name, emb in stored_embeddings.items():
                dist = np.linalg.norm(rep - emb)
                print(f"Distance to {name}: {dist:.4f}")
                if dist < best_dist:
                    best_dist = dist
                    best_match = name

            if best_dist < THRESHOLD:
                detected_names.append(best_match)
                cv2.putText(frame, best_match, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # ✅ Start auto-stop timer when first known face appears
                if first_detect_time is None:
                    first_detect_time = time.time()

            else:
                cv2.putText(frame, "Unknown", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        except Exception as e:
            print("Recognition error:", e)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # ✅ Start/Stop detection timers
    current_time = time.time()
    for student in students:
        name = student["Name"]
        if name in detected_names:
            if detection_times[name] is None:
                detection_times[name] = current_time
            elif current_time - detection_times[name] >= CONFIRMATION_TIME:
                student["Status"] = "Present"
        else:
            detection_times[name] = None

    # ✅ Show countdown if first face detected
    if first_detect_time:
        remaining = AUTO_STOP_TIME - (current_time - first_detect_time)
        if remaining > 0:
            cv2.putText(frame, f"Auto stopping in {remaining:.1f}s",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            print("⏹️ Auto-stopping after 7 seconds of detection.")
            break

    cv2.imshow("Real-Time Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Print final attendance
print("\n📝 Attendance Report:")
today = datetime.datetime.now().strftime("%Y-%m-%d")
for s in students:
    print(f"{today} - {s['Name']}: {s['Status']}")
