
import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from io import BytesIO

st.set_page_config(
    page_title="AI Object Detector",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Object Detection")
st.write(
    "Upload an image to detect multiple objects "
    "with bounding boxes and confidence scores."
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")

# ---------------- SETTINGS ----------------
with st.sidebar:
    st.header("Detection Settings")

    confidence = st.slider(
        "Minimum confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.caption(
        "Higher confidence thresholds usually produce "
        "fewer detections."
    )

    st.write("Model: YOLO26 Nano")

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    try:
        image_bytes = uploaded_file.getvalue()

        if len(image_bytes) > 10 * 1024 * 1024:
            st.error("Image must be smaller than 10 MB.")
            st.stop()

        image = Image.open(BytesIO(image_bytes))
        image.verify()

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Please upload a valid image.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Results")

        detect_button = st.button(
            "🔍 Detect Objects",
            type="primary",
            use_container_width=True
        )

        if detect_button:
            try:
                with st.spinner("Detecting objects..."):
                    model = load_model()

                    result = model.predict(
                        source=image,
                        conf=confidence,
                        verbose=False
                    )[0]

                    # YOLO returns an annotated BGR image.
                    annotated_bgr = result.plot()

                    # Convert BGR to RGB for Streamlit.
                    annotated_rgb = annotated_bgr[:, :, ::-1].copy()

                    detection_rows = []

                    if result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls.item())
                            score = float(box.conf.item())

                            x1, y1, x2, y2 = (
                                box.xyxy[0].tolist()
                            )

                            detection_rows.append({
                                "Object": result.names[class_id],
                                "Confidence (%)": round(
                                    score * 100, 2
                                ),
                                "X1": round(x1, 1),
                                "Y1": round(y1, 1),
                                "X2": round(x2, 1),
                                "Y2": round(y2, 1)
                            })

                    st.session_state["detection_image"] = (
                        annotated_rgb
                    )
                    st.session_state["detection_rows"] = (
                        detection_rows
                    )
                    st.session_state["detection_file"] = (
                        uploaded_file.name
                    )

            except Exception as error:
                st.error("Object detection failed.")

                with st.expander("Technical details"):
                    st.code(str(error))

    # ---------------- DISPLAY RESULTS ----------------
    if (
        st.session_state.get("detection_file")
        == uploaded_file.name
        and "detection_image" in st.session_state
    ):
        st.divider()
        st.subheader("📦 Detected Objects")

        annotated = st.session_state["detection_image"]
        rows = st.session_state["detection_rows"]

        st.image(
            annotated,
            caption="Detected objects with bounding boxes",
            use_container_width=True
        )

        total_objects = len(rows)

        metric1, metric2 = st.columns(2)

        metric1.metric("Total Objects", total_objects)

        if rows:
            metric2.metric(
                "Object Categories",
                len(set(row["Object"] for row in rows))
            )

            df = pd.DataFrame(rows)

            st.subheader("Detection Details")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "⬇️ Download Detection Results",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="detection_results.csv",
                mime="text/csv"
            )

            output = BytesIO()
            Image.fromarray(annotated).save(
                output,
                format="PNG"
            )

            st.download_button(
                "⬇️ Download Annotated Image",
                data=output.getvalue(),
                file_name="detected_objects.png",
                mime="image/png"
            )

        else:
            st.info(
                "No objects were detected above the selected "
                "confidence threshold. Try lowering it."
            )

else:
    st.info("Upload an image to begin.")
