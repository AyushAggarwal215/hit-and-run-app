import streamlit as st
import cv2
import tempfile
import os
import re
import requests
import easyocr

from datetime import datetime, timezone
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BACKEND_URL = "https://urban-net-sih26124.onrender.com/api/edge/events"

BUS_ID = "BUS_17"
CAMERA_ID = "CAM_FRONT"


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="Hit-and-Run Detection",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Hit-and-Run Detection System")

st.write(
    "Upload a transport-bus video. The system will automatically "
    "detect vehicles, identify a possible hit-and-run, detect the "
    "number plate and send the result to the backend."
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    vehicle_model = YOLO("yolo11n.pt")

    plate_model = YOLO("best.pt")

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    return vehicle_model, plate_model, reader


vehicle_model, plate_model, reader = load_models()


# ============================================================
# VIDEO UPLOAD
# ============================================================

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

        # ====================================================
        # OPEN VIDEO
        # ====================================================

        cap = cv2.VideoCapture(temp_video.name)

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 30

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        st.info("Processing video... Please wait.")

        progress = st.progress(0)


        # ====================================================
        # TRACK HISTORY
        # ====================================================

        tracks = {}

        collision_events = []

        frame_no = 0


        # ====================================================
        # STEP 1 — VEHICLE / PERSON TRACKING
        # ====================================================

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


                for obj_id, cls, box in zip(
                    ids,
                    classes,
                    xyxy
                ):

                    obj_id = int(obj_id)

                    cls = int(cls)

                    x1, y1, x2, y2 = box

                    cx = (x1 + x2) / 2

                    cy = (y1 + y2) / 2


                    # ----------------------------------------
                    # Calculate movement speed in pixels/frame
                    # ----------------------------------------

                    previous = tracks.get(obj_id)

                    speed = 0

                    if previous is not None:

                        px, py, pf = previous

                        speed = (
                            (cx - px) ** 2 +
                            (cy - py) ** 2
                        ) ** 0.5


                    tracks[obj_id] = (
                        cx,
                        cy,
                        frame_no
                    )


                    # COCO classes:
                    #
                    # 0 = person
                    # 2 = car
                    # 3 = motorcycle
                    # 5 = bus
                    # 7 = truck

                    if cls in [0, 2, 3, 5, 7]:

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


                # ====================================================
                # COLLISION CANDIDATE DETECTION
                # ====================================================

                for i in range(
                    len(current_objects)
                ):

                    for j in range(
                        i + 1,
                        len(current_objects)
                    ):

                        a = current_objects[i]

                        b = current_objects[j]


                        distance = (
                            (a["cx"] - b["cx"]) ** 2 +
                            (a["cy"] - b["cy"]) ** 2
                        ) ** 0.5


                        # Possible collision

                        if distance < 100:

                            collision_events.append({

                                "frame": frame_no,

                                "a": a,

                                "b": b

                            })


            frame_no += 1


            if total_frames > 0:

                progress.progress(
                    min(
                        frame_no / total_frames,
                        1.0
                    )
                )


        cap.release()


        # ====================================================
        # STEP 2 — CHECK COLLISION
        # ====================================================

        if len(collision_events) == 0:

            st.warning(
                "No possible collision was detected "
                "in this video."
            )

            os.unlink(temp_video.name)

        else:

            collision = collision_events[0]

            collision_frame = collision["frame"]

            obj_a = collision["a"]

            obj_b = collision["b"]


            # ====================================================
            # STEP 3 — IDENTIFY OFFENDING VEHICLE
            # ====================================================

            vehicle_classes = [
                2, 3, 5, 7
            ]


            if obj_a["class"] in vehicle_classes:

                offender_id = obj_a["id"]

                offender_box = obj_a


            elif obj_b["class"] in vehicle_classes:

                offender_id = obj_b["id"]

                offender_box = obj_b


            else:

                offender_id = obj_a["id"]

                offender_box = obj_a


            # ====================================================
            # VIDEO TIMESTAMP
            # ====================================================

            timestamp_seconds = (
                collision_frame / fps
            )

            minutes = int(
                timestamp_seconds // 60
            )

            seconds = int(
                timestamp_seconds % 60
            )

            video_timestamp = (
                f"{minutes:02d}:{seconds:02d}"
            )


            # ====================================================
            # STEP 4 — NUMBER PLATE DETECTION
            # ====================================================

            cap = cv2.VideoCapture(
                temp_video.name
            )


            plate_results = []


            # Search frames around collision

            start_frame = max(
                0,
                collision_frame - 30
            )

            end_frame = (
                collision_frame + 60
            )


            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                start_frame
            )


            current = start_frame


            while current <= end_frame:

                ret, frame = cap.read()

                if not ret:
                    break


                # --------------------------------------------
                # Plate detector
                # --------------------------------------------

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

                        plate_detection_conf = float(
                            box.conf[0]
                        )


                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0].cpu().numpy()
                        )


                        # ------------------------------------
                        # Crop plate
                        # ------------------------------------

                        crop = frame[
                            max(0, y1):
                            max(y1 + 1, y2),

                            max(0, x1):
                            max(x1 + 1, x2)
                        ]


                        if crop.size == 0:
                            continue


                        # ------------------------------------
                        # Resize plate for OCR
                        # ------------------------------------

                        crop = cv2.resize(
                            crop,
                            None,
                            fx=3,
                            fy=3,
                            interpolation=cv2.INTER_CUBIC
                        )


                        # ------------------------------------
                        # OCR
                        # ------------------------------------

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

                                combined_confidence = (
                                    float(ocr_conf) *
                                    plate_detection_conf
                                )


                                plate_results.append({

                                    "plate": cleaned,

                                    "confidence":
                                        combined_confidence,

                                    "frame": current

                                })


                current += 1


            cap.release()


            # ====================================================
            # STEP 5 — SELECT BEST PLATE
            # ====================================================

            if len(plate_results) > 0:

                plate_results.sort(
                    key=lambda x: x["confidence"],
                    reverse=True
                )


                best_plate = plate_results[0]


                plate_number = best_plate["plate"]

                plate_confidence = min(
                    best_plate["confidence"],
                    1.0
                )


                # ====================================================
                # VEHICLE TYPE
                # ====================================================

                vehicle_class = offender_box["class"]


                vehicle_type_map = {

                    2: "car",

                    3: "motorcycle",

                    5: "bus",

                    7: "truck"

                }


                vehicle_type = vehicle_type_map.get(
                    vehicle_class,
                    "vehicle"
                )


                # ====================================================
                # CREATE BACKEND JSON
                # ====================================================

                payload = {

                    "eventType":
                        "HIT_AND_RUN",


                    "busId":
                        BUS_ID,


                    "cameraId":
                        CAMERA_ID,


                    "timestamp":
                        datetime.now(
                            timezone.utc
                        ).isoformat(),


                    "location": {

                        "latitude": 26,

                        "longitude": 77,

                        "address": None

                    },


                    "detection": {

                        "confidence":
                            float(
                                plate_confidence
                            ),

                        "severity":
                            "CRITICAL"

                    },


                    "model": {

                        "name":
                            "hit-and-run-yolo",

                        "version":
                            "1.0"

                    },


                    "evidence": {

                        "imageUrl":
                            None

                    },


                    "metadata": {

                        "offendingVehicleReg":
                            plate_number,

                        "offendingVehicleDetails":
                            vehicle_type

                    }

                }


                # ====================================================
                # SEND RESULT TO BACKEND
                # ====================================================

                try:

                    response = requests.post(

                        BACKEND_URL,

                        json=payload,

                        timeout=15

                    )


                    backend_success = (
                        response.status_code
                        in [200, 201]
                    )


                except Exception as e:

                    backend_success = False

                    backend_error = str(e)


                # ====================================================
                # STREAMLIT RESULT
                # ====================================================

                st.success(
                    "🚨 HIT-AND-RUN DETECTED"
                )


                st.markdown("---")


                st.subheader(
                    "Incident Details"
                )


                col1, col2 = st.columns(2)


                with col1:

                    st.write(
                        f"**Offender Track ID:** "
                        f"{offender_id}"
                    )

                    st.write(
                        f"**Registration Number:** "
                        f"{plate_number}"
                    )

                    st.write(
                        f"**Vehicle Type:** "
                        f"{vehicle_type}"
                    )


                with col2:

                    st.write(
                        f"**OCR Confidence:** "
                        f"{plate_confidence * 100:.1f}%"
                    )

                    st.write(
                        f"**Video Timestamp:** "
                        f"{video_timestamp}"
                    )


                st.markdown("---")


                # ====================================================
                # SHOW JSON
                # ====================================================

                st.subheader(
                    "📡 Backend Payload"
                )

                st.json(payload)


                # ====================================================
                # BACKEND STATUS
                # ====================================================

                if backend_success:

                    st.success(
                        "✅ Result successfully sent "
                        "to the website backend."
                    )

                else:

                    st.error(
                        "❌ Detection completed, "
                        "but the result could not be "
                        "sent to the backend."
                    )

                    if not backend_success:

                        try:

                            st.write(
                                "Backend response:",
                                response.text
                            )

                        except:

                            pass


                # ====================================================
                # FINAL ALERT
                # ====================================================

                st.markdown("---")

                st.error(
                    f"""
🚨 HIT-AND-RUN ALERT

Vehicle Track ID : {offender_id}
Registration     : {plate_number}
Vehicle Type     : {vehicle_type}
OCR Confidence   : {plate_confidence * 100:.1f}%
Video Timestamp  : {video_timestamp}
"""
                )


            else:

                st.warning(
                    "Hit-and-run candidate detected, "
                    "but the number plate could not "
                    "be read clearly."
                )


            os.unlink(temp_video.name)
