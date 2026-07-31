# Live AI Proctoring System

An exam proctoring platform that uses computer vision to analyze webcam feeds in real time and score the probability of cheating behavior. Built with a Python FastAPI inference service, a Spring Boot web backend, and a browser-based frontend.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Risk Scoring Engine](#risk-scoring-engine)
- [Detection Modules](#detection-modules)
- [Known Limitations](#known-limitations)

---

## Overview

This system captures a student's webcam feed every 2 seconds during an exam and sends each frame to a Python AI service for analysis. The service returns a risk score (0–100) along with detailed signals — such as head direction, eye gaze, phone detection, and multiple face detection — which are displayed live in the browser dashboard.

The goal is not to accuse students of cheating based on a single frame, but to accumulate a signal over time using streak-based penalties with grace periods, smoothed by a weighted exponential moving average.

---

## System Architecture

```
Browser (HTML + JS)
      │
      │  POST /upload (multipart JPEG every 2s)
      ▼
Spring Boot (port 8080)
      │
      │  Proxies to → POST /analyze
      ▼
FastAPI AI Service (port 8000)
      │
      ├── YOLO v8n          — person + phone detection
      ├── OpenCV DNN SSD    — face count (reliable cross-platform)
      └── MediaPipe FaceMesh — head pose + iris gaze (optional)
```

Spring Boot serves the HTML frontend and acts as a reverse proxy, forwarding frame uploads to the Python service and returning the JSON response to the browser.

---

## Features

| Feature | Method |
|---|---|
| Person detection | YOLOv8n (COCO class 0) |
| Phone detection | YOLOv8n (COCO class 67) |
| Face detection & count | OpenCV DNN SSD (res10_300x300) |
| Head direction (yaw/pitch) | MediaPipe FaceMesh + solvePnP |
| Eye gaze (iris position) | MediaPipe iris landmarks 468/473 |
| Camera blocked detection | Frame brightness threshold |
| Multiple persons / faces | YOLO + DNN combined |
| Streak-based penalties | Grace period + capped accumulation |
| Risk smoothing | Weighted EMA over last 5 frames |
| Event logging | Per-session deque, logged every N frames |
| Session reset | POST /reset clears all state |

---

## Tech Stack

**AI Service**
- Python 3.10+
- FastAPI + Uvicorn
- Ultralytics YOLOv8
- OpenCV (cv2) with DNN module
- MediaPipe (optional — for head pose and gaze)
- NumPy

**Web Backend**
- Java 17+
- Spring Boot 3.x
- Acts as reverse proxy for `/upload` → FastAPI `/analyze`

**Frontend**
- Vanilla HTML + JavaScript
- `getUserMedia` for webcam access
- Canvas API for frame capture
- No external JS dependencies

---

## Project Structure

```
ai-services/
├── main.py                          # FastAPI app (this is proctor_api.py)
├── deploy.prototxt                  # OpenCV DNN face detector config (auto-downloaded)
├── res10_300x300_ssd_iter_140000.caffemodel  # Face model weights (auto-downloaded)
├── face_landmarker.task             # MediaPipe tasks model (auto-downloaded if needed)
├── yolov8n.pt                       # YOLO weights (auto-downloaded by ultralytics)
└── .venv/

spring-boot-app/
├── src/main/resources/templates/
│   └── proctoring.html              # Frontend dashboard
└── src/main/java/.../
    └── ProctorController.java       # Proxy controller for /upload and /reset-session
```

---

## Setup & Installation

### 1. Python AI Service

```bash
cd ai-services
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install fastapi uvicorn ultralytics opencv-python numpy
```

MediaPipe is optional but recommended for head pose and gaze detection:

```bash
# Recommended (last version with mp.solutions API — most stable)
pip install mediapipe==0.10.3

# Or use the latest version (new tasks API — model auto-downloads on first run)
pip install mediapipe
```

The OpenCV DNN face detector models (`deploy.prototxt` and the `.caffemodel` file) are downloaded automatically on first startup. No manual download required.

### 2. Spring Boot Backend

Ensure you have Java 17+ and Maven or Gradle installed. Add a proxy controller that forwards `POST /upload` to `http://localhost:8000/analyze` and `POST /reset-session` to `http://localhost:8000/reset`.

Example Spring Boot proxy (RestTemplate):

```java
@PostMapping("/upload")
public ResponseEntity<String> upload(@RequestParam("file") MultipartFile file) {
    // forward multipart to FastAPI /analyze and return JSON response
}

@PostMapping("/reset-session")
public ResponseEntity<String> reset() {
    // forward to FastAPI /reset
}
```

---

## Running the System

### Start the AI service

```bash
cd ai-services
.venv\Scripts\activate      # Windows
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

On first run you will see the face detector models downloading — this happens once only.

### Start Spring Boot

```bash
./mvnw spring-boot:run
# or
./gradlew bootRun
```

Open `http://localhost:8080` in the browser, click **Start Monitoring**, and allow camera access.

---

## API Reference

### `POST /analyze`

Accepts a JPEG image as multipart form data. Returns a JSON object with all detection results and the current risk score.

**Request:** `multipart/form-data` with field `file` (JPEG image)

**Response:**

```json
{
  "risk_score": 42,
  "risk_reasons": [
    { "event": "HEAD_TURN", "detail": "Head looking_left (yaw=-28.3° pitch=2.1°)", "delta": 12 }
  ],
  "person_count": 1,
  "phone_detected": false,
  "face_count": 1,
  "face_missing": false,
  "face_missing_streak": 0,
  "head_direction": { "direction": "looking_left", "yaw": -28.3, "pitch": 2.1 },
  "eye_gaze": { "gaze": "center", "suspicious": false, "ratio": 0.51 },
  "look_away_streak": 2,
  "frame_number": 47,
  "session_elapsed_sec": 94,
  "camera_blocked": false,
  "mediapipe_available": true,
  "recent_events": [ ... ]
}
```

### `GET /events`

Returns the full session event log and total frame count.

### `POST /reset`

Resets all session state (streaks, event log, frame counter, risk history).

### `GET /health`

Returns service status, frames processed, and active detection modes.

---

## Risk Scoring Engine

Risk is scored per frame on a 0–100 scale and smoothed across frames. It is designed for exam context — a single brief glance should not trigger a high score.

### Risk weights

| Signal | Base Score | Notes |
|---|---|---|
| Camera blocked | +80 | Immediate, camera covered |
| Multiple persons | +70 | Additional person in frame |
| Phone detected | +65 | Mobile phone visible |
| Multiple faces | +60 | Second face detected |
| No person (after grace) | +35 | Student stepped away |
| No face (after grace) | +20 | Face not visible |
| Face hidden (person present) | +25 | Head down / obscured |
| Head turn | +5 to +35 | Proportional to angle |
| Look-away streak | +15/frame capped at 35 | Sustained gaze away |
| Face missing streak | +10/frame capped at 40 | Sustained face absence |
| Suspicious eye gaze | +15 | Iris offset, head forward |

### Grace periods

Streak penalties only activate after a minimum number of consecutive frames, preventing transient posture changes from triggering false alarms:

| Condition | Grace period |
|---|---|
| Face missing | 4 frames (~8 seconds) |
| Looking away | 3 frames (~6 seconds) |
| No person | 3 frames (~6 seconds) |

### Smoothing

Risk is smoothed using a weighted average of the last 5 frames, with the most recent frame contributing 40% of the final score. This eliminates single-frame spikes caused by motion blur or detection noise.

---

## Detection Modules

### Person & Phone — YOLOv8n

Uses Ultralytics YOLOv8 nano model on COCO classes: `0` (person) and `67` (cell phone). Confidence threshold: 0.40.

### Face Detection — OpenCV DNN SSD

Uses the ResNet-10 SSD face detector included in OpenCV's DNN module. This model is chosen for its reliability across platforms without MediaPipe version constraints. Two small model files are downloaded automatically on first startup. Confidence threshold: 0.55.

### Head Pose — MediaPipe FaceMesh + solvePnP

When a face is confirmed by the DNN detector, MediaPipe FaceMesh extracts 478 facial landmarks. Six canonical points (nose tip, chin, eye corners, mouth corners) are matched against a 3D face model using `cv2.solvePnP` to recover yaw and pitch angles. Labeled as forward, looking\_left, looking\_right, looking\_up, or looking\_down.

### Eye Gaze — Iris Landmarks

When `refine_landmarks=True` is available (MediaPipe solutions API), iris center landmarks at indices 468 and 473 are used to compute how far the iris sits within the eye boundary. A ratio below 0.35 or above 0.65 is flagged as suspicious gaze.

### Camera Blocked Detection

If the mean pixel brightness of a frame falls below 12 (on a 0–255 scale), the frame is classified as dark and `CAMERA_BLOCKED` is raised immediately.

---

## Known Limitations

- Frame rate is fixed at one frame every 2 seconds. Fast actions between captures are not detected.
- YOLO phone detection may miss phones held below the camera's field of view.
- Eye gaze detection requires MediaPipe solutions API with `refine_landmarks=True`. The newer tasks API does not reliably expose iris landmarks, so gaze falls back to unavailable.
- Head pose estimation assumes a roughly frontal face. Extreme angles (>60°) may produce unstable PnP solutions.
- The system is designed as a supporting signal for human review, not as a standalone judgment tool.
