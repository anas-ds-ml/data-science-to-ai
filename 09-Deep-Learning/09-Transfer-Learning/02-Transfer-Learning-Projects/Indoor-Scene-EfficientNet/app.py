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

CLASS_NAMES_FILE = "class_names.json"

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

    model = keras.models.load_model(
        model_path
    )

    return model


# =========================================================
# LOAD CLASS NAMES FROM HUGGING FACE
# =========================================================

@st.cache_data
def load_class_names():

    class_names_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=CLASS_NAMES_FILE,
        repo_type="model"
    )

    with open(
        class_names_path,
        "r"
    ) as f:

        class_names = json.load(f)

    return class_names


# =========================================================
# LOAD MODEL + CLASS NAMES
# =========================================================

try:

    model = load_model()

    class_names = load_class_names()

except Exception as e:

    st.error(
        "❌ Could not load the model or class names "
        "from Hugging Face."
    )

    st.exception(e)

    st.stop()


# =========================================================
# VERIFY CLASS COUNT
# =========================================================

if len(class_names) != 67:

    st.error(
        f"❌ Expected 67 classes, "
        f"but found {len(class_names)} classes "
        "in class_names.json."
    )

    st.stop()


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload an indoor scene image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    try:

        # -------------------------------------------------
        # OPEN IMAGE
        # -------------------------------------------------

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # -------------------------------------------------
        # DISPLAY IMAGE
        # -------------------------------------------------

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )


        # -------------------------------------------------
        # PREPROCESS IMAGE
        # -------------------------------------------------

        resized_image = image.resize(
            IMAGE_SIZE
        )

        image_array = np.array(
            resized_image,
            dtype=np.float32
        )

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


        # -------------------------------------------------
        # GET PROBABILITIES
        # -------------------------------------------------

        probabilities = predictions[0]


        # -------------------------------------------------
        # GET PREDICTED CLASS
        # -------------------------------------------------

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = class_names[
            predicted_index
        ]


        # -------------------------------------------------
        # GET CONFIDENCE
        # -------------------------------------------------

        confidence = float(
            probabilities[predicted_index]
        )


        # =================================================
        # DISPLAY PREDICTION
        # =================================================

        st.subheader("Prediction")

        st.success(
            f"🏠 {predicted_class}"
        )

        st.write(
            f"Confidence: "
            f"**{confidence * 100:.2f}%**"
        )

        st.progress(
            confidence
        )


        # =================================================
        # TOP 5 PREDICTIONS
        # =================================================

        st.subheader(
            "Top 5 Predictions"
        )

        top_5_indices = np.argsort(
            probabilities
        )[-5:][::-1]


        for index in top_5_indices:

            index = int(index)

            class_name = class_names[
                index
            ]

            probability = float(
                probabilities[index]
            )

            st.write(
                f"**{class_name}** — "
                f"{probability * 100:.2f}%"
            )


    # =====================================================
    # IMAGE PROCESSING ERROR
    # =====================================================

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