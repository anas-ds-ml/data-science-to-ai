import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
from huggingface_hub import hf_hub_download


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_REPO = "MuhammadAnasDS/cat_dog_cnn"
MODEL_FILE = "cat_dog_cnn.keras"

IMAGE_SIZE = (128, 128)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🐱 Cat vs Dog Classifier")

st.write(
    "Upload an image and let the CNN classify it as a Cat or Dog."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        repo_type="model"
    )

    model = keras.models.load_model(model_path)

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = load_model()

except Exception as e:

    st.error("❌ Could not load the model.")

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a Cat or Dog image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------------------------
    # Display uploaded image
    # --------------------------------------------------------

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )


    # --------------------------------------------------------
    # Resize image
    # --------------------------------------------------------

    resized_image = image.resize(
        IMAGE_SIZE
    )


    # --------------------------------------------------------
    # Convert image to NumPy
    # --------------------------------------------------------

    image_array = np.array(
        resized_image
    )


    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )


    prediction_value = float(
        prediction[0][0]
    )


    # --------------------------------------------------------
    # Determine class
    # --------------------------------------------------------

    if prediction_value >= 0.5:

        predicted_class = "Dog 🐶"

        confidence = prediction_value

    else:

        predicted_class = "Cat 🐱"

        confidence = 1 - prediction_value


    confidence_percentage = confidence * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Prediction")

    st.success(predicted_class)

    st.write(
        f"Confidence: **{confidence_percentage:.2f}%**"
    )


    # Confidence bar
    st.progress(
        confidence
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CNN Binary Image Classification • TensorFlow / Keras"
)