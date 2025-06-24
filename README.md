# 🧠 IBM Hackathon YOLOv8 Object Detection Project

This project was developed as part of the IBM Hackathon and demonstrates real-time object detection using **YOLOv8** and a lightweight web interface using **Flask**.

## 📌 Features

- 🔍 Object detection using YOLOv8 (`ultralytics`)
- 🎥 Real-time webcam/video feed support
- 🌐 Web UI for uploading and analyzing videos
- 🖼️ Visual output and screenshots of detections
- 🧪 Modular design for YOLO inference and preprocessing

---

## 🚀 Tech Stack

- Python 3
- YOLOv8 (`ultralytics`)
- Flask (for web hosting)
- HTML/CSS (for UI)
- OpenCV, NumPy

---

## 📁 Project Structure

```plaintext
IBM_Hackathon/
│
├── VideoFetching.py            # Handles video stream
├── Webhost.py                  # Flask app to host web UI
├── templates/
│   └── index.html              # Web interface template
├── Temp/
│   ├── ObjectDetection.py      # Core detection logic
│   ├── yolo.py / yolo_v1.py    # YOLO-specific modules
│   └── yolov8n.pt              # YOLO model weights (ignored in .gitignore)
├── utils/
│   └── coco.txt                # COCO labels
├── Outputs/                    # Contains screenshots, outputs (ignored in .gitignore)
└── requirements.txt            # Python package dependencies






