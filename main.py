from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import time

# ==========================================
# FACE DETECTION SETUP
# ==========================================

try:
    import mediapipe as mp

    mp_face = getattr(mp, "solutions", None)

    if mp_face is not None and hasattr(mp_face, "face_detection"):
        face_detector = mp_face.face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )
        face_detector_available = True
    else:
        face_detector = None
        face_detector_available = False

except Exception:
    face_detector = None
    face_detector_available = False

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==========================================
# FASTAPI
# ==========================================

app = FastAPI()

# ==========================================
# YOLO MODEL
# ==========================================

yolo_model = YOLO("yolov8n.pt")

COCO_CLASSES = yolo_model.names

# ==========================================
# STATE
# ==========================================

state = {
    "risk_score": 0,
    "face_missing_counter": 0,
    "last_head_direction": "CENTER",

    "phone_active": False,
    "book_active": False,
    "laptop_active": False,
    "keyboard_active": False,
    "mouse_active": False,

    "events": []
}

# ==========================================
# EVENT LOGGER
# ==========================================

def add_event(event_name, risk_delta):

    state["events"].append({
        "timestamp": time.strftime("%H:%M:%S"),
        "event": event_name,
        "risk_delta": risk_delta
    })

    if len(state["events"]) > 20:
        state["events"] = state["events"][-20:]

# ==========================================
# ANALYZE API
# ==========================================

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    try:

        contents = await file.read()

        image_array = np.frombuffer(
            contents,
            np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return {
                "risk_score": 0,
                "person_count": 0,
                "phone_detected": False,
                "book_detected": False,
                "laptop_detected": False,
                "keyboard_detected": False,
                "mouse_detected": False,
                "face_detected": False,
                "head_direction": "UNKNOWN",
                "events": []
            }

        # ==================================
        # YOLO DETECTION
        # ==================================

        results = yolo_model(frame)

        person_count = 0

        phone_detected = False
        book_detected = False
        laptop_detected = False
        keyboard_detected = False
        mouse_detected = False

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])
                class_name = COCO_CLASSES[cls]

                if cls == 0:
                    person_count += 1

                if class_name == "cell phone":
                    phone_detected = True

                elif class_name == "book":
                    book_detected = True

                elif class_name == "laptop":
                    laptop_detected = True

                elif class_name == "keyboard":
                    keyboard_detected = True

                elif class_name == "mouse":
                    mouse_detected = True

        # ==================================
        # EVENTS FOR OBJECTS
        # ==================================

        if phone_detected and not state["phone_active"]:
            add_event("PHONE_DETECTED", 50)
            state["phone_active"] = True

        if not phone_detected:
            state["phone_active"] = False

        if book_detected and not state["book_active"]:
            add_event("BOOK_DETECTED", 20)
            state["book_active"] = True

        if not book_detected:
            state["book_active"] = False

        if laptop_detected and not state["laptop_active"]:
            add_event("LAPTOP_DETECTED", 30)
            state["laptop_active"] = True

        if not laptop_detected:
            state["laptop_active"] = False

        if keyboard_detected and not state["keyboard_active"]:
            add_event("KEYBOARD_DETECTED", 15)
            state["keyboard_active"] = True

        if not keyboard_detected:
            state["keyboard_active"] = False

        if mouse_detected and not state["mouse_active"]:
            add_event("MOUSE_DETECTED", 15)
            state["mouse_active"] = True

        if not mouse_detected:
            state["mouse_active"] = False

        # ==================================
        # FACE DETECTION
        # ==================================

        face_detected = False
        head_direction = "CENTER"

        height, width, _ = frame.shape

        if face_detector_available and face_detector is not None:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            face_results = face_detector.process(rgb)

            if face_results.detections:

                face_detected = True
                state["face_missing_counter"] = 0

                detection = face_results.detections[0]

                bbox = detection.location_data.relative_bounding_box

                center_x = (
                    bbox.xmin +
                    bbox.width / 2
                ) * width

                if center_x < width * 0.35:
                    head_direction = "LEFT"

                elif center_x > width * 0.65:
                    head_direction = "RIGHT"

                else:
                    head_direction = "CENTER"

                if head_direction != state["last_head_direction"]:

                    add_event(
                        f"HEAD_{head_direction}",
                        5
                    )

                    state["last_head_direction"] = head_direction

            else:

                state["face_missing_counter"] += 1

        else:

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5
            )

            if len(faces) > 0:

                face_detected = True
                state["face_missing_counter"] = 0

                x, y, w_face, h_face = faces[0]

                center_x = x + (w_face / 2)

                if center_x < width * 0.35:
                    head_direction = "LEFT"

                elif center_x > width * 0.65:
                    head_direction = "RIGHT"

                else:
                    head_direction = "CENTER"

                if head_direction != state["last_head_direction"]:

                    add_event(
                        f"HEAD_{head_direction}",
                        5
                    )

                    state["last_head_direction"] = head_direction

            else:

                state["face_missing_counter"] += 1

        # ==================================
        # FACE MISSING EVENT
        # ==================================

        if state["face_missing_counter"] >= 3:

            add_event(
                "FACE_MISSING",
                30
            )

            state["face_missing_counter"] = 0

        # ==================================
        # RISK ENGINE
        # ==================================

        risk = state["risk_score"]

        # decay
        risk = max(0, risk - 2)

        if person_count > 1:
            risk += 40

        if phone_detected:
            risk += 20

        if book_detected:
            risk += 10

        if laptop_detected:
            risk += 15

        if keyboard_detected:
            risk += 5

        if mouse_detected:
            risk += 5

        if not face_detected:
            risk += 20

        if head_direction in ["LEFT", "RIGHT"]:
            risk += 10

        state["risk_score"] = min(100, risk)

        # ==================================
        # RESPONSE
        # ==================================

        return {
            "risk_score": state["risk_score"],

            "person_count": person_count,

            "phone_detected": phone_detected,
            "book_detected": book_detected,
            "laptop_detected": laptop_detected,
            "keyboard_detected": keyboard_detected,
            "mouse_detected": mouse_detected,

            "face_detected": face_detected,
            "head_direction": head_direction,

            "events": state["events"]
        }

    except Exception as e:

        print("ERROR:", str(e))

        return {
            "risk_score": 0,
            "person_count": 0,

            "phone_detected": False,
            "book_detected": False,
            "laptop_detected": False,
            "keyboard_detected": False,
            "mouse_detected": False,

            "face_detected": False,
            "head_direction": "UNKNOWN",

            "events": [],

            "error": str(e)
        }