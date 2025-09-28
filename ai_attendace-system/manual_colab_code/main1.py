from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
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
stored_embeddings = {name: np.array(vec).astype("float32")[:128] for name, vec in stored_embeddings.items()}
for k in stored_embeddings:
    stored_embeddings[k] = stored_embeddings[k] / np.linalg.norm(stored_embeddings[k])

# Students list
students = [
    {"Roll_No": "F2021266031", "Name": "Shahan", "Department": "CS", "Status": "Absent"},
    {"Roll_No": "F2021266600", "Name": "Haroon", "Department": "CS", "Status": "Absent"},
]

THRESHOLD = 1.1

app = FastAPI()

# ✅ Show HTML upload page at "/"
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>Face Attendance</title></head>
        <body style="font-family: Arial; text-align: center; margin-top: 50px;">
            <h2>Upload an Image for Attendance</h2>
            <form action="/mark_attendance" enctype="multipart/form-data" method="post">
                <input name="file" type="file" accept="image/*">
                <button type="submit">Upload & Detect</button>
            </form>
        </body>
    </html>
    """

@app.post("/mark_attendance")
async def mark_attendance(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Reset students status every call
    for s in students:
        s["Status"] = "Absent"

    detected_names = []

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

        if best_dist < THRESHOLD:
            detected_names.append(best_match)

    # Update attendance
    for student in students:
        if student["Name"] in detected_names:
            student["Status"] = "Present"

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    return JSONResponse({"date": today, "attendance": students})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
