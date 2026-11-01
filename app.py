
import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
from transformers import pipeline
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Image Intelligence",
    page_icon="🧠",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 38px;
    font-weight: 700;
    text-align: center;
}
.subtitle {
    text-align: center;
    color: #808495;
    margin-bottom: 25px;
}
[data-testid="stMetric"] {
    background: rgba(128,128,128,0.08);
    padding: 16px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">🧠 AI Image Intelligence</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'Upload an image and discover what AI recognizes.'
    '</div>',
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")

    top_k = st.slider(
        "Number of predictions",
        min_value=1,
        max_value=5,
        value=5
    )

    st.divider()

    st.subheader("About the model")
    st.write("Architecture: Vision Transformer (ViT)")
    st.write("Task: Image classification")
    st.write("Model: Google ViT Base Patch16 224")

    st.caption(
        "Predictions are based on the pretrained "
        "model's learned image categories."
    )

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_classifier():
    return pipeline(
        task="image-classification",
        model="google/vit-base-patch16-224"
    )

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📂 Upload an image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Choose a JPG, JPEG, PNG, or WEBP image."
)

if uploaded_file is not None:

    # Validate image before classification
    try:
        image_bytes = uploaded_file.getvalue()

        if len(image_bytes) > 10 * 1024 * 1024:
            st.error("Please upload an image smaller than 10 MB.")
            st.stop()

        image = Image.open(BytesIO(image_bytes))
        image.verify()

        image = Image.open(BytesIO(image_bytes)).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Invalid or corrupted image. Please upload another.")
        st.stop()

    # ---------------- IMAGE PREVIEW ----------------
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("📷 Image Preview")
        st.image(
            image,
            caption=uploaded_file.name,
            use_container_width=True
        )

        st.caption(f"Image dimensions: {image.width} × {image.height}")

    with right:
        st.subheader("🔍 Classification")

        if st.button(
            "🚀 Classify Image",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner("Loading AI model and analyzing image..."):
                    classifier = load_classifier()
                    predictions = classifier(
                        image,
                        top_k=top_k
                    )

                st.session_state.last_predictions = predictions
                st.session_state.last_filename = uploaded_file.name

                st.session_state.history.append({
                    "Image": uploaded_file.name,
                    "Prediction": predictions[0]["label"],
                    "Confidence (%)": round(
                        predictions[0]["score"] * 100, 2
                    )
                })

            except Exception as error:
                st.error("Classification failed.")
                with st.expander("Technical details"):
                    st.code(str(error))

    # ---------------- DISPLAY RESULTS ----------------
    predictions = st.session_state.get("last_predictions")
    last_filename = st.session_state.get("last_filename")

    if predictions and last_filename == uploaded_file.name:
        st.divider()
        st.subheader("📊 Prediction Results")

        best = predictions[0]
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Best Prediction",
                best["label"]
            )

        with col2:
            st.metric(
                "Model Confidence",
                f'{best["score"] * 100:.2f}%'
            )

        st.markdown("#### Ranked predictions")

        for index, prediction in enumerate(predictions, start=1):
            label = prediction["label"]
            score = float(prediction["score"])

            st.write(f"**{index}. {label}**")
            st.progress(score)
            st.caption(f"Confidence: {score * 100:.2f}%")

        # ---------------- DOWNLOAD RESULTS ----------------
        result_df = pd.DataFrame([
            {
                "Rank": i,
                "Label": p["label"],
                "Confidence (%)": round(p["score"] * 100, 2)
            }
            for i, p in enumerate(predictions, start=1)
        ])

        csv_data = result_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Predictions (CSV)",
            data=csv_data,
            file_name="image_classification_results.csv",
            mime="text/csv"
        )

else:
    st.info("Upload an image to get started.")

# ---------------- PREDICTION HISTORY ----------------
if st.session_state.history:
    st.divider()
    st.subheader("🕘 Session Prediction History")

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)

    history_csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download History (CSV)",
        data=history_csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    if st.button("Clear History"):
        st.session_state.history = []
        st.rerun()
