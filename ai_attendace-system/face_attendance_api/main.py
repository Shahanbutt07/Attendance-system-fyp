from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn, cv2, numpy as np, json, datetime
from deepface import DeepFace
from ultralytics import YOLO

# ==== CONFIG ====
YOLO_MODEL_PATH = "model.pt"
EMBEDDINGS_PATH = "embeddings_facenet.json"

# Load YOLO model
model = YOLO(YOLO_MODEL_PATH)

# Load embeddings
with open(EMBEDDINGS_PATH, "r") as f:
    stored_embeddings = json.load(f)

# Normalize embeddings
stored_embeddings = {
    name: np.array(vec).astype("float32")[:128] / np.linalg.norm(np.array(vec).astype("float32")[:128])
    for name, vec in stored_embeddings.items()
}

students = [
    {"Roll_No": f"R-{i+1:03}", "Name": name, "Department": "CS", "Status": "Absent"}
    for i, name in enumerate(stored_embeddings.keys())
]

THRESHOLD = 0.6  # stricter value to avoid false matches

app = FastAPI()

@app.post("/mark_attendance")
async def mark_attendance(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    for s in students:
        s["Status"] = "Absent"

    detected_names = []
    already_detected = set()  # ✅ track which names are already used

    results = model(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        rep = DeepFace.represent(face_crop, model_name="Facenet",
                                 enforce_detection=False,
                                 detector_backend="skip")[0]["embedding"]
        rep = np.array(rep).astype("float32")[:128]
        rep = rep / np.linalg.norm(rep)

        best_match = "Unknown"
        best_dist = float("inf")

        for name, emb in stored_embeddings.items():
            dist = np.linalg.norm(rep - emb)
            if dist < best_dist:
                best_dist = dist
                best_match = name

        # ✅ Strict: only allow match if not already detected
        if best_dist < THRESHOLD and best_match not in already_detected:
            detected_names.append(best_match)
            already_detected.add(best_match)
        else:
            detected_names.append("Unknown")

    for student in students:
        if student["Name"] in detected_names:
            student["Status"] = "Present"

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return JSONResponse({
        "date": today,
        "detected": detected_names,
        "attendance": students
    })

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
