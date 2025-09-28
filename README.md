# AI-Based Face Recognition Attendance System 🎓

An **AI-powered attendance management system** that uses **YOLOv8** for face detection and **DeepFace (FaceNet)** for face recognition.  
The system automates attendance by detecting and recognizing multiple students in real-time or from uploaded classroom images.  

Built with **FastAPI** for seamless backend integration with mobile or web apps.

---

## 🚀 Features
- Detect multiple student faces in **real-time** using YOLOv8.
- Recognize students using **FaceNet embeddings (128-d)** via DeepFace.
- Mark attendance automatically as **Present** or **Absent**.
- Supports:
  - **Live webcam feed**
  - **Uploaded class images** (teacher view)
- Calculates **attendance percentage** automatically.
- Unregistered students are automatically marked **Absent**.
- Generates attendance reports in CSV format with the following fields:
  - Name
  - Roll Number
  - Department
  - Date
  - Attendance Status
  - Attendance Percentage

---

## 🛠 Tech Stack
- **Python** (Backend Development)
- **YOLOv8** – Face Detection
- **DeepFace (FaceNet)** – Face Recognition
- **FastAPI** – API Development
- **OpenCV** – Image Processing
- **Google Colab** – Model Training and Prototyping

---

## 📊 Workflow
1. **Face Detection:**  
   YOLOv8 detects all faces in a live webcam feed or uploaded class photo.
2. **Face Recognition:**  
   Detected faces are compared against pre-stored FaceNet embeddings.
3. **Attendance Marking:**  
   - Matched faces → **Present**
   - Unmatched faces → **Absent**
4. **Report Generation:**  
   Attendance data is saved and percentages are auto-calculated.

---

## 📝 Example Attendance CSV Output
| Name           | Roll_No     | Department | Date       | Status   | Attendance % |
|----------------|------------|------------|------------|----------|--------------|
| Shahan         | F2021266031| CS         | 2025-09-27 | Present  | 95%          |
| Haroon         | F2021266600| CS         | 2025-09-27 | Absent   | 87%          |
| Wassam Shah    | F2021266030| CS         | 2025-09-27 | Absent   | 70%          |

---

## 📌 Future Improvements
- Web-based dashboard for teachers and admins.
- SQL database integration for scalable data storage.
- Facial expression recognition (e.g., detecting smiling students 😁).
- Real-time analytics dashboard.

---

## 📜 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 👤 Author
**Muhammad Shahan Butt**  
[GitHub Profile](https://github.com/yourusername) | [LinkedIn](https://linkedin.com/in/yourlinkedin)
