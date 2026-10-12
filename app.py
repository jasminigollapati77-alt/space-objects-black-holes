
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO
from pathlib import Path
import io

MODEL_PATH = (
    "/content/drive/MyDrive/blackhole_project/"
    str(Path(__file__).resolve().parent / "best.pt")
)

st.set_page_config(
    page_title="Space Objects Around Black Holes",
    page_icon="🌌",
    layout="wide"
)

st.title("🌌 Space Objects Around Black Holes")
st.caption("Classification and automatic bright-region analysis")

@st.cache_resource
def load_model():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return YOLO(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Upload an astronomical image",
    type=["png", "jpg", "jpeg"]
)

st.sidebar.header("Bright-region settings")
percentile = st.sidebar.slider("Brightness threshold percentile", 90, 99, 97)
min_area = st.sidebar.slider("Minimum region area (pixels)", 5, 500, 25)
max_regions = st.sidebar.slider("Maximum boxes", 1, 50, 15)
padding = st.sidebar.slider("Crop padding (pixels)", 0, 100, 32)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    rgb = np.array(image)

    with st.spinner("Classifying the full image..."):
        result = model.predict(
            source=image, imgsz=224, device="cpu", verbose=False
        )[0]

    st.subheader("Whole-image classification")
    if result.probs is not None:
        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        st.write(f"**Predicted class:** {result.names[class_id]}")
        st.write(f"**Confidence:** {confidence:.2%}")

    st.subheader("Bright regions with predicted labels")

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = np.percentile(smooth, percentile)
    mask = (smooth >= threshold).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    regions = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            regions.append((int(area), int(x), int(y), int(w), int(h)))

    regions.sort(reverse=True)
    regions = regions[:max_regions]

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    rows = []

    for number, (area, x, y, w, h) in enumerate(regions, 1):
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.width, x + w + padding)
        y2 = min(image.height, y + h + padding)
        crop = image.crop((x1, y1, x2, y2))

        try:
            crop_result = model.predict(
                source=crop, imgsz=224, device="cpu", verbose=False
            )[0]
            crop_class = int(crop_result.probs.top1)
            name = crop_result.names[crop_class]
            score = float(crop_result.probs.top1conf)
        except Exception:
            name = "Prediction unavailable"
            score = 0.0

        label = f"{name} ({score:.0%})"
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=3)
        draw.text(
            (x, max(0, y - 20)),
            label,
            fill="yellow",
            stroke_width=2,
            stroke_fill="black"
        )

        rows.append({
            "Region": number,
            "Predicted class": name,
            "Confidence": f"{score:.2%}",
            "X": x, "Y": y, "Width": w, "Height": h,
            "Area (pixels)": area
        })

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original image", width="stretch")
    with col2:
        st.image(annotated, caption="Labeled candidate regions", width="stretch")

    st.write(f"**Candidate regions found:** {len(regions)}")

    if rows:
        st.subheader("Region prediction details")
        st.dataframe(rows, use_container_width=True)

    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    st.download_button(
        "Download labeled image",
        data=buffer.getvalue(),
        file_name="labeled_astronomical_image.png",
        mime="image/png"
    )

    st.warning(
        "Each box is a bright image region. Its name is a classification "
        "prediction for a crop, not a verified identification of an "
        "individual astronomical object."
    )
