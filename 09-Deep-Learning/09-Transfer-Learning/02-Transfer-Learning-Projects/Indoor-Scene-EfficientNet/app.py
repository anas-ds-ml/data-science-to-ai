import streamlit as st
import numpy as np
import json

from PIL import Image
from tensorflow import keras
from huggingface_hub import hf_hub_download


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Indoor Scene Classifier",
    page_icon="🏠",
    layout="centered"
)


# =========================================================
# HUGGING FACE SETTINGS
# =========================================================

MODEL_REPO = "MuhammadAnasDS/indoor-scene-efficientnet"

MODEL_FILE = "indoor_scene_efficientnet_final.keras"

IMAGE_SIZE = (224, 224)


# =========================================================
# TITLE
# =========================================================

st.title("🏠 Indoor Scene Classifier")

st.write(
    "Upload an image and the EfficientNetB0 model "
    "will classify it into one of 67 indoor scene categories."
)


# =========================================================
# LOAD MODEL FROM HUGGING FACE
# =========================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        repo_type="model"
    )

    model = keras.models.load_model(model_path)

    return model


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = load_model()

except Exception as e:

    st.error("❌ Could not load the model from Hugging Face.")

    st.exception(e)

    st.stop()


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "airport_inside",
    "artstudio",
    "auditorium",
    "bakery",
    "bar",
    "bathroom",
    "bedroom",
    "bookstore",
    "bowling",
    "buffet",
    "casino",
    "children_room",
    "church_inside",
    "classroom",
    "cloister",
    "closet",
    "clothingstore",
    "computerroom",
    "concert_hall",
    "conference_center",
    "conference_room",
    "corridor",
    "deli",
    "dentaloffice",
    "dining_room",
    "door",
    "elevator",
    "fastfood_restaurant",
    "florist",
    "gameroom",
    "garage",
    "greenhouse",
    "grocerystore",
    "gym",
    "hairsalon",
    "hospitalroom",
    "inside_bus",
    "inside_subway",
    "jewelleryshop",
    "kindergarden",
    "kitchen",
    "laboratorywet",
    "laundromat",
    "library",
    "livingroom",
    "lobby",
    "locker_room",
    "mall",
    "meeting_room",
    "movietheater",
    "museum",
    "nursery",
    "office",
    "operating_room",
    "pantry",
    "poolinside",
    "prisoncell",
    "restaurant",
    "restaurant_kitchen",
    "shoeshop",
    "staircase",
    "studiomusic",
    "subway",
    "toilet",
    "trainstation",
    "tv_studio",
    "videostore",
    "waitingroom",
    "warehouse",
    "winecellar"
]


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an indoor scene image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    try:

        # Open image
        image = Image.open(uploaded_file).convert("RGB")

        # Display image
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        # -------------------------------------------------
        # PREPROCESS IMAGE
        # -------------------------------------------------

        resized_image = image.resize(IMAGE_SIZE)

        image_array = np.array(resized_image)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Get probabilities
        probabilities = predictions[0]

        # Find highest probability
        predicted_index = np.argmax(probabilities)

        predicted_class = class_names[predicted_index]

        confidence = float(
            probabilities[predicted_index]
        )

        # -------------------------------------------------
        # DISPLAY RESULT
        # -------------------------------------------------

        st.subheader("Prediction")

        st.success(
            f"🏠 {predicted_class}"
        )

        st.write(
            f"Confidence: **{confidence * 100:.2f}%**"
        )

        st.progress(confidence)

        # -------------------------------------------------
        # TOP 5 PREDICTIONS
        # -------------------------------------------------

        st.subheader("Top 5 Predictions")

        top_5_indices = np.argsort(
            probabilities
        )[-5:][::-1]

        for index in top_5_indices:

            class_name = class_names[index]

            probability = float(
                probabilities[index]
            )

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )

    except Exception as e:

        st.error(
            "❌ Unable to process this image."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "EfficientNetB0 • Transfer Learning • "
    "67-Class Indoor Scene Classification"
)