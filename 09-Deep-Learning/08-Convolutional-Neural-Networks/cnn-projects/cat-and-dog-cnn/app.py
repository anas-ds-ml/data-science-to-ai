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

# Minimum confidence required to classify as Cat or Dog
CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🐱🐶 Cat vs Dog Classifier")

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

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

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
        resized_image,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # Normalize image
    # Same preprocessing used during training
    # --------------------------------------------------------

    image_array = image_array / 255.0


    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

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


    # ========================================================
    # CALCULATE CLASS PROBABILITIES
    # ========================================================

    dog_probability = prediction_value

    cat_probability = 1 - prediction_value


    # ========================================================
    # DETERMINE CLASS
    # ========================================================

    if dog_probability >= CONFIDENCE_THRESHOLD:

        predicted_class = "Dog 🐶"

        confidence = dog_probability

        is_unknown = False


    elif cat_probability >= CONFIDENCE_THRESHOLD:

        predicted_class = "Cat 🐱"

        confidence = cat_probability

        is_unknown = False


    else:

        predicted_class = "No Cat or Dog detected ❓"

        confidence = max(
            cat_probability,
            dog_probability
        )

        is_unknown = True


    confidence_percentage = confidence * 100


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Prediction")


    if is_unknown:

        st.warning(
            "❓ No clear Cat or Dog detected."
        )

        st.write(
            "The model is not confident enough to classify "
            "this image as a Cat or Dog."
        )

    else:

        st.success(
            predicted_class
        )

        st.write(
            f"Confidence: **{confidence_percentage:.2f}%**"
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader("Prediction Probabilities")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "🐱 Cat",
            f"{cat_probability * 100:.2f}%"
        )


    with col2:

        st.metric(
            "🐶 Dog",
            f"{dog_probability * 100:.2f}%"
        )


    # ========================================================
    # PROBABILITY BARS
    # ========================================================

    st.write("🐱 Cat Probability")

    st.progress(
        float(cat_probability)
    )


    st.write("🐶 Dog Probability")

    st.progress(
        float(dog_probability)
    )


    # ========================================================
    # THRESHOLD INFORMATION
    # ========================================================

    with st.expander("ℹ️ How does this work?"):

        st.write(
            f"""
            The application uses a confidence threshold of
            **{CONFIDENCE_THRESHOLD * 100:.0f}%**.

            If the model is at least **70% confident** that the
            image is a Dog, it displays **Dog 🐶**.

            If the model is at least **70% confident** that the
            image is a Cat, it displays **Cat 🐱**.

            If neither Cat nor Dog reaches the threshold, the
            application displays:

            **❓ No Cat or Dog detected**
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "CNN Binary Image Classification • TensorFlow / Keras"
)
