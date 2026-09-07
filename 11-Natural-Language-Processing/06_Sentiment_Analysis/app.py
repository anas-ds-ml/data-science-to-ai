import streamlit as st
import joblib
import re
import string
import nltk

from bs4 import BeautifulSoup
from better_profanity import profanity
from huggingface_hub import hf_hub_download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IBM Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# =========================================================
# NLTK
# =========================================================

@st.cache_resource
def setup_nltk():

    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


setup_nltk()


# =========================================================
# LOAD MODEL FROM HUGGING FACE
# =========================================================

MODEL_REPO = "MuhammadAnasDS/sentiment-analysis-model"


@st.cache_resource
def load_model():

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="sentiment_model.pkl"
    )

    vectorizer_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="vectorizer.pkl"
    )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


# =========================================================
# LOAD MODEL SAFELY
# =========================================================

try:

    model, vectorizer = load_model()

    model_loaded = True

except Exception as e:

    model_loaded = False
    model = None
    vectorizer = None

    st.error(
        "⚠️ Unable to load the sentiment analysis model."
    )

    st.info(
        "Please make sure the model files are available "
        "in the Hugging Face Model Repository."
    )


# =========================================================
# PROFANITY
# =========================================================

profanity.load_censor_words()


# =========================================================
# STOPWORDS
# =========================================================

stop_words = set(
    stopwords.words("english")
)

# Keep sentiment-important words
stop_words.discard("not")
stop_words.discard("no")
stop_words.discard("never")

lemmatizer = WordNetLemmatizer()


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280 !important;
        font-size: 17px;
        margin-bottom: 35px;
    }

    textarea {
        border-radius: 12px !important;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 50px;
        font-size: 17px;
        font-weight: 600;
    }

    .results-title {
        font-size: 30px;
        font-weight: 700;
        margin-top: 35px;
        margin-bottom: 25px;
    }

    .result-card {
        border: 1px solid #e5e7eb;
        border-radius: 15px;
        padding: 25px;
        background-color: #ffffff !important;
        min-height: 150px;
        color: #111827 !important;
    }

    .result-label {
        color: #6b7280 !important;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .result-value {
        font-size: 32px;
        font-weight: 700;
        color: #111827 !important;
    }

    .confidence-value {
        font-size: 36px;
        font-weight: 700;
        color: #111827 !important;
    }

    .score-header {
        display: grid;
        grid-template-columns: 1fr 2fr 1fr;
        background-color: #f8fafc !important;
        padding: 14px;
        border: 1px solid #e5e7eb;
        font-weight: 600;
        color: #374151 !important;
    }

    .score-row {
        display: grid;
        grid-template-columns: 1fr 2fr 1fr;
        padding: 15px;
        border-left: 1px solid #e5e7eb;
        border-right: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    .clear-button {
        text-align: center;
        padding: 15px;
        margin-top: 25px;
        border: 1px solid #d1d5db;
        border-radius: 12px;
        color: #6b7280 !important;
        font-weight: 500;
    }

    .info-box {
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #f8fafc;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #374151;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">'
    '🎬 IBM Movie Sentiment Analyzer'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze movie reviews and predict whether the sentiment '
    'is Positive or Negative using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL STATUS
# =========================================================

if model_loaded:

    st.success(
        "✅ Sentiment Analysis Model Loaded Successfully"
    )


# =========================================================
# MOVIE REVIEW INPUT
# =========================================================

review = st.text_area(
    "Enter your movie review",
    height=220,
    placeholder=(
        "Example: This movie was absolutely amazing! "
        "The acting was excellent and the story was fantastic."
    )
)


# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🔍 Analyze Review"):

    # -----------------------------------------------------
    # MODEL CHECK
    # -----------------------------------------------------

    if not model_loaded:

        st.error(
            "The model could not be loaded. "
            "Please check the Hugging Face model repository."
        )

    # -----------------------------------------------------
    # EMPTY REVIEW CHECK
    # -----------------------------------------------------

    elif not review.strip():

        st.warning(
            "Please enter a movie review."
        )

    else:

        # =================================================
        # PROFANITY CHECK
        # =================================================

        if profanity.contains_profanity(review):

            st.error(
                "⚠️ Your review contains abusive or offensive "
                "language. Please use respectful language and "
                "try again."
            )

        else:

            # =================================================
            # PREPROCESSING
            # =================================================

            review_clean = review.lower()

            # Remove HTML
            review_clean = BeautifulSoup(
                review_clean,
                "html.parser"
            ).get_text()

            # Remove URLs
            review_clean = re.sub(
                r"http\S+|www\S+",
                "",
                review_clean
            )

            # Remove punctuation
            review_clean = review_clean.translate(
                str.maketrans(
                    "",
                    "",
                    string.punctuation
                )
            )

            # Remove numbers
            review_clean = re.sub(
                r"\d+",
                "",
                review_clean
            )

            # Remove extra spaces
            review_clean = " ".join(
                review_clean.split()
            )

            # Tokenization
            words = review_clean.split()

            # Remove stopwords
            words = [
                word
                for word in words
                if word not in stop_words
            ]

            # Lemmatization
            words = [
                lemmatizer.lemmatize(word)
                for word in words
            ]

            review_clean = " ".join(words)


            # =================================================
            # VECTORIZATION
            # =================================================

            try:

                review_vector = vectorizer.transform(
                    [review_clean]
                )

            except Exception as e:

                st.error(
                    "❌ Error while transforming the review."
                )

                st.exception(e)

                st.stop()


            # =================================================
            # PREDICTION
            # =================================================

            try:

                prediction = model.predict(
                    review_vector
                )[0]

            except Exception as e:

                st.error(
                    "❌ Error while making prediction."
                )

                st.exception(e)

                st.stop()


            # =================================================
            # PROBABILITIES
            # =================================================

            try:

                probabilities = model.predict_proba(
                    review_vector
                )[0]

                classes = model.classes_

                probability_dict = {
                    str(cls).lower(): probability
                    for cls, probability in zip(
                        classes,
                        probabilities
                    )
                }

            except Exception as e:

                st.error(
                    "❌ This model does not support "
                    "probability prediction."
                )

                st.exception(e)

                st.stop()


            # =================================================
            # POSITIVE / NEGATIVE PROBABILITIES
            # =================================================

            positive_probability = 0
            negative_probability = 0

            for label, probability in probability_dict.items():

                if label in [
                    "positive",
                    "1",
                    "true"
                ]:

                    positive_probability = probability

                elif label in [
                    "negative",
                    "0",
                    "false"
                ]:

                    negative_probability = probability


            # =================================================
            # RESULT
            # =================================================

            prediction_text = str(
                prediction
            ).lower()

            if prediction_text in [
                "positive",
                "1",
                "true"
            ]:

                result = "POSITIVE"
                emoji = "😊"
                confidence = positive_probability * 100

            else:

                result = "NEGATIVE"
                emoji = "😞"
                confidence = negative_probability * 100


            # =================================================
            # RESULTS TITLE
            # =================================================

            st.markdown(
                '<div class="results-title">'
                'Analysis Results'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # RESULT COLUMNS
            # =================================================

            col1, col2 = st.columns(2)


            # =================================================
            # PREDICTION CARD
            # =================================================

            with col1:

                st.markdown(
                    '<div class="result-card">'
                    '<div class="result-label">'
                    'Predicted Sentiment'
                    '</div>'
                    '<div class="result-value">'
                    f'{emoji} {result}'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )


            # =================================================
            # CONFIDENCE CARD
            # =================================================

            with col2:

                st.markdown(
                    '<div class="result-card">'
                    '<div class="result-label">'
                    'Confidence Score'
                    '</div>'
                    '<div class="confidence-value">'
                    f'{confidence:.2f}%'
                    '</div>'
                    '</div>',
                    unsafe_allow_html=True
                )


            # =================================================
            # DETAILED SCORES
            # =================================================

            st.markdown(
                '<div class="results-title" '
                'style="font-size:26px;">'
                'Detailed Scores'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # SCORE HEADER
            # =================================================

            st.markdown(
                '<div class="score-header">'
                '<div>Type</div>'
                '<div>Sentiment</div>'
                '<div>Confidence</div>'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # NEGATIVE SCORE
            # =================================================

            st.markdown(
                '<div class="score-row">'
                '<div>😞</div>'
                '<div>NEGATIVE</div>'
                f'<div>{negative_probability * 100:.2f}%</div>'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # POSITIVE SCORE
            # =================================================

            st.markdown(
                '<div class="score-row">'
                '<div>😊</div>'
                '<div>POSITIVE</div>'
                f'<div>{positive_probability * 100:.2f}%</div>'
                '</div>',
                unsafe_allow_html=True
            )


            # =================================================
            # REVIEW PROCESSED
            # =================================================

            with st.expander("🔎 View Processed Review"):

                st.write(review_clean)


            # =================================================
            # CLEAR MESSAGE
            # =================================================

            st.markdown(
                '<div class="clear-button">'
                '🔄 Enter another movie review '
                'to analyze its sentiment'
                '</div>',
                unsafe_allow_html=True
            )