import streamlit as st
import cv2
import tempfile
import os
import re
import easyocr
from ultralytics import YOLO
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="Hit-and-Run Detection", layout="wide")

st.title("🚨 Hit-and-Run Detection System")
st.write("Upload a transport-bus video. The system will automatically detect the collision, fleeing vehicle, and number plate.")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    vehicle_model = YOLO("yolo11n.pt")
    plate_model = YOLO("best.pt")
    reader = easyocr.Reader(["en"], gpu=False)
    return vehicle_model, plate_model, reader


vehicle_model, plate_model, reader = load_models()

# -----------------------------
# VIDEO UPLOAD
# -----------------------------
video = st.file_uploader(
    "Upload accident video",
    type=["mp4", "avi", "mov", "mkv"]
)

if video is not None:

    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_video.write(video.read())
    temp_video.close()

    st.video(video)

    if st.button("🔍 Analyze Video"):

        cap = cv2.VideoCapture(temp_video.name)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        st.info("Processing video... Please wait.")

        progress = st.progress(0)

        # Track history
        tracks = {}

        collision_events = []

        frame_no = 0

        # -----------------------------
        # STEP 1: VEHICLE TRACKING
        # -----------------------------
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            results = vehicle_model.track(
                frame,
                tracker="bytetrack.yaml",
                persist=True,
                conf=0.4,
                verbose=False
            )

            boxes = results[0].boxes

            if boxes.id is not None:

                ids = boxes.id.cpu().numpy()
                classes = boxes.cls.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()

                current_objects = []

                for obj_id, cls, box in zip(ids, classes, xyxy):

                    obj_id = int(obj_id)
                    cls = int(cls)

                    x1, y1, x2, y2 = box

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # COCO:
                    # 0 = person
                    # 2 = car
                    # 3 = motorcycle
                    # 5 = bus
                    # 7 = truck

                    if cls in [0, 2, 3, 5, 7]:

                        previous = tracks.get(obj_id)

                        speed = 0

                        if previous is not None:
                            px, py, pf = previous
                            speed = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

                        tracks[obj_id] = (cx, cy, frame_no)

                        current_objects.append({
                            "id": obj_id,
                            "class": cls,
                            "cx": cx,
                            "cy": cy,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "speed": speed
                        })

                # -----------------------------
                # COLLISION CANDIDATE
                # -----------------------------
                for i in range(len(current_objects)):

                    for j in range(i + 1, len(current_objects)):

                        a = current_objects[i]
                        b = current_objects[j]

                        distance = (
                            (a["cx"] - b["cx"]) ** 2 +
                            (a["cy"] - b["cy"]) ** 2
                        ) ** 0.5

                        # Vehicle/person or vehicle/vehicle
                        if distance < 100:

                            collision_events.append({
                                "frame": frame_no,
                                "a": a,
                                "b": b
                            })

            frame_no += 1

            if total_frames > 0:
                progress.progress(
                    min(frame_no / total_frames, 1.0)
                )

        cap.release()

        # -----------------------------
        # STEP 2: FIND COLLISION
        # -----------------------------
        if len(collision_events) == 0:

            st.warning(
                "No possible collision was detected in this video."
            )

        else:

            collision = collision_events[0]

            collision_frame = collision["frame"]

            obj_a = collision["a"]
            obj_b = collision["b"]

            # Prefer vehicle as offender
            vehicle_classes = [2, 3, 5, 7]

            if obj_a["class"] in vehicle_classes:
                offender_id = obj_a["id"]
                offender_box = obj_a
            elif obj_b["class"] in vehicle_classes:
                offender_id = obj_b["id"]
                offender_box = obj_b
            else:
                offender_id = obj_a["id"]
                offender_box = obj_a

            timestamp = collision_frame / fps

            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)

            time_string = f"{minutes:02d}:{seconds:02d}"

            # -----------------------------
            # STEP 3: FIND PLATE
            # -----------------------------
            cap = cv2.VideoCapture(temp_video.name)

            plate_results = []

            # Look around collision frame
            start_frame = max(0, collision_frame - 30)
            end_frame = collision_frame + 60

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            current = start_frame

            while current <= end_frame:

                ret, frame = cap.read()

                if not ret:
                    break

                # Plate detection
                results = plate_model.predict(
                    frame,
                    imgsz=640,
                    conf=0.15,
                    verbose=False
                )

                for result in results:

                    if result.boxes is None:
                        continue

                    for box in result.boxes:

                        confidence = float(
                            box.conf[0]
                        )

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].cpu().numpy()
                        )

                        crop = frame[
                            max(0, y1):max(y1 + 1, y2),
                            max(0, x1):max(x1 + 1, x2)
                        ]

                        if crop.size == 0:
                            continue

                        # OCR
                        ocr_results = reader.readtext(
                            crop,
                            detail=1
                        )

                        for _, text, ocr_conf in ocr_results:

                            cleaned = re.sub(
                                r"[^A-Z0-9]",
                                "",
                                text.upper()
                            )

                            if len(cleaned) >= 5:

                                final_conf = (
                                    float(ocr_conf) *
                                    confidence
                                )

                                plate_results.append(
                                    (
                                        cleaned,
                                        final_conf,
                                        current
                                    )
                                )

                current += 1

            cap.release()

            # -----------------------------
            # STEP 4: BEST PLATE RESULT
            # -----------------------------
            if len(plate_results) > 0:

                plate_results.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                plate_number = plate_results[0][0]
                plate_confidence = plate_results[0][1]

                # Limit to 100%
                plate_confidence = min(
                    plate_confidence,
                    1.0
                )

                st.success("🚨 HIT-AND-RUN DETECTED")

                st.markdown("---")

                st.subheader("Incident Details")

                col1, col2 = st.columns(2)

                with col1:
                    st.write(
                        f"**Offender Track ID:** {offender_id}"
                    )

                    st.write(
                        f"**Registration Number:** {plate_number}"
                    )

                with col2:
                    st.write(
                        f"**OCR Confidence:** "
                        f"{plate_confidence * 100:.1f}%"
                    )

                    st.write(
                        f"**Timestamp:** {time_string}"
                    )

                st.markdown("---")

                st.error(
                    f"""
🚨 HIT-AND-RUN ALERT

Vehicle Track ID : {offender_id}
Registration     : {plate_number}
OCR Confidence   : {plate_confidence * 100:.1f}%
Timestamp        : {time_string}
"""
                )

            else:

                st.warning(
                    "Hit-and-run candidate detected, "
                    "but the number plate could not be read clearly."
                )

        os.unlink(temp_video.name)
