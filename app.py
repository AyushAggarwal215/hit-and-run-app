import streamlit as st
import numpy as np
from PIL import Image
import easyocr
from ultralytics import YOLO

st.set_page_config(
    page_title="Hit-and-Run Detection",
    page_icon="🚨"
)

st.title("🚨 Hit-and-Run Detection")
st.write("License Plate Detection + OCR")


@st.cache_resource
def load_model():
    return YOLO("best.pt")


@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)


model = load_model()
reader = load_ocr()


uploaded_file = st.file_uploader(
    "Upload a vehicle image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_array = np.array(image)

    results = model.predict(
        image_array,
        imgsz=640,
        conf=0.15,
        verbose=False
    )

    detections = []

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].cpu().numpy()
        )

        plate = image_array[y1:y2, x1:x2]

        if plate.size == 0:
            continue

        ocr_results = reader.readtext(
            plate,
            detail=1
        )

        for _, text, confidence in ocr_results:

            text = text.upper().replace(" ", "")

            if text:
                detections.append(
                    (text, confidence)
                )


    if detections:

        st.success("License plate detected!")

        for text, confidence in detections:

            st.subheader(
                f"🚗 Registration: {text}"
            )

            st.write(
                f"OCR Confidence: {confidence:.2%}"
            )

    else:

        st.warning(
            "No readable license plate detected."
        )
