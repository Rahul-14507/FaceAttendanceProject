# 🧠 Face Recognition Attendance System (with Blink-Based Liveness Detection)

A simple yet powerful **AI-based Attendance System** that uses **face recognition** and **blink detection (liveness check)** to mark attendance automatically.  
Built using `OpenCV`, `face_recognition`, and `NumPy`.

---

## 🚀 Features

- 👤 Detects and recognizes faces in real-time via webcam  
- 👁️ Detects blinking for liveness (prevents spoofing using photos)  
- 🗃️ Saves and reuses face encodings with `pickle` for faster startup  
- 🕒 Automatically logs attendance with timestamps into `attendance.csv`  
- 🧾 Modular and easy-to-customize Python code  

---

## 🧩 Project Structure

* **FaceRecognition-Attendance/**
    * `main.py` - Main script for running face recognition and attendance
    * `camera.py` - Test script to find correct camera index
    * **`known_faces/`** - Folder containing known face images (one per person)
        * `Rahul.jpg`
        * `Alice.jpg`
        * ...
    * `encodings.pkl` - Auto-generated file storing saved encodings
    * `attendance.csv` - Auto-generated attendance log
    * `requirements.txt` - Python dependencies
    * `README.md` - Project documentation

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>


