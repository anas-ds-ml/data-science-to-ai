import streamlit as st
import tensorflow as tf
import numpy as np
import json

from PIL import Image
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Background */
    .stApp {
        background-color: #0b0b0b;
        color: white;
    }

    /* Main container */
    .block-container {
        max-width: 800px;
        padding-top: 45px;
        padding-bottom: 50px;
    }

    /* Title */
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #999999;
        margin-bottom: 35px;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #111111;
        border: 1px solid #292929;
        border-radius: 14px;
        padding: 12px;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        height: 48px;

        background-color: #8b5cf6;
        color: white;

        border: none;
        border-radius: 10px;

        font-size: 16px;
        font-weight: 600;

        transition: 0.2s ease;
    }

    .stButton > button:hover {
        background-color: #7c3aed;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #666666;
        font-size: 13px;
        margin-top: 35px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HUGGING FACE CONFIG
# ============================================================

REPO_ID = "MuhammadAnasDS/flower-classification-cnn"

MODEL_FILENAME = "flower_classification_cnn.keras"

CLASS_NAMES_FILENAME = "class_names.json"

IMG_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model"
    )

    model = tf.keras.models.load_model(model_path)

    return model


# ============================================================
# LOAD CLASS NAMES
# ============================================================

@st.cache_data
def load_class_names():

    class_names_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=CLASS_NAMES_FILENAME,
        repo_type="model"
    )

    with open(class_names_path, "r") as file:
        class_names = json.load(file)

    return class_names


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image, model, class_names):

    image_array = preprocess_image(image)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = predictions[predicted_index]

    return predicted_class, confidence


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🌸 Flower Classification</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a flower image and let the CNN classify it.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner("Loading model..."):

        model = load_model()

        class_names = load_class_names()

except Exception as e:

    st.error("Unable to load the model from Hugging Face.")

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a flower image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file:

    image = Image.open(uploaded_file)

    st.markdown("<br>", unsafe_allow_html=True)

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🔍 Predict Flower",
        width="stretch"
    ):

        with st.spinner("Analyzing image..."):

            predicted_class, confidence = predict_image(
                image,
                model,
                class_names
            )

        # ====================================================
        # RESULT CARD
        # ====================================================

        st.html(
            f"""
            <div style="
                margin-top: 20px;
                padding: 25px;
                text-align: center;

                background-color: #111111;

                border: 1px solid #292929;

                border-radius: 16px;
            ">

                <div style="
                    color: #888888;
                    font-size: 14px;
                    margin-bottom: 8px;
                ">
                    PREDICTED FLOWER
                </div>

                <div style="
                    font-size: 32px;
                    font-weight: 700;
                    color: #c084fc;
                    margin-bottom: 8px;
                ">
                    {predicted_class}
                </div>

                <div style="
                    color: #aaaaaa;
                    font-size: 16px;
                ">
                    Confidence: {confidence * 100:.2f}%
                </div>

            </div>
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Built with TensorFlow • Keras • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)