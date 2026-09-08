import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Multilingual Translator",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = "facebook/nllb-200-distilled-600M"


# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {
    "🇬🇧 English": "eng_Latn",
    "🇵🇰 Urdu": "urd_Arab",
    "🇸🇦 Arabic": "arb_Arab",
    "🇪🇸 Spanish": "spa_Latn",
    "🇫🇷 French": "fra_Latn",
    "🇩🇪 German": "deu_Latn",
    "🇮🇹 Italian": "ita_Latn",
    "🇵🇹 Portuguese": "por_Latn",
    "🇷🇺 Russian": "rus_Cyrl",
    "🇨🇳 Chinese (Simplified)": "zho_Hans",
    "🇯🇵 Japanese": "jpn_Jpan",
    "🇰🇷 Korean": "kor_Hang",
    "🇮🇳 Hindi": "hin_Deva",
    "🇧🇩 Bengali": "ben_Beng",
    "🇹🇷 Turkish": "tur_Latn",
    "🇮🇷 Persian": "pes_Arab",
    "🇳🇱 Dutch": "nld_Latn",
    "🇵🇱 Polish": "pol_Latn",
    "🇸🇪 Swedish": "swe_Latn",
    "🇬🇷 Greek": "ell_Grek"
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


# ============================================================
# TRANSLATION FUNCTION
# ============================================================

def translate_text(
    text,
    source_language,
    target_language
):

    tokenizer.src_lang = source_language

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    target_token_id = tokenizer.convert_tokens_to_ids(
        target_language
    )

    with torch.no_grad():

        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=target_token_id,
            max_new_tokens=256,
            num_beams=5,
            early_stopping=True
        )

    translated_text = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return translated_text


# ============================================================
# HEADER
# ============================================================

st.title("🌍 AI Multilingual Translator")

st.markdown(
    """
    ### 🤖 Transformer-Based Language Translation

    Translate text between **20 different languages**
    using the pretrained **NLLB-200 Transformer model**.
    """
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

with st.spinner("🤖 Loading NLLB-200 Transformer model..."):

    tokenizer, model = load_model()


st.success("✅ Translation model is ready!")


# ============================================================
# LANGUAGE SELECTION
# ============================================================

col1, col2 = st.columns(2)


with col1:

    source_language = st.selectbox(
        "🌐 Translate From",
        list(LANGUAGES.keys())
    )


with col2:

    target_language = st.selectbox(
        "🌍 Translate To",
        list(LANGUAGES.keys()),
        index=1
    )


# ============================================================
# TEXT INPUT
# ============================================================

st.subheader("✍️ Enter Your Text")

text = st.text_area(
    "Input Text",
    height=180,
    placeholder="Type or paste your text here..."
)


# ============================================================
# TRANSLATE BUTTON
# ============================================================

translate_button = st.button(
    "🚀 Translate",
    use_container_width=True
)


# ============================================================
# TRANSLATION
# ============================================================

if translate_button:

    if not text.strip():

        st.warning(
            "⚠️ Please enter some text before translating."
        )

    elif source_language == target_language:

        st.warning(
            "⚠️ Please select different source and target languages."
        )

    else:

        source_code = LANGUAGES[source_language]

        target_code = LANGUAGES[target_language]

        with st.spinner("🔄 Translating..."):

            try:

                result = translate_text(
                    text,
                    source_code,
                    target_code
                )

                st.subheader("📝 Translation")

                st.success("✅ Translation completed!")

                st.text_area(
                    "Translated Text",
                    result,
                    height=180
                )

            except Exception as e:

                st.error(
                    f"❌ Translation failed: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    ### 🧠 Model Information

    **Model:** NLLB-200 Distilled 600M  
    **Architecture:** Sequence-to-Sequence Transformer  
    **Languages:** 20 selected languages  
    **Framework:** PyTorch + Hugging Face Transformers  
    **Interface:** Streamlit  

    > This application uses a pretrained Transformer model
    > and does not train a Transformer from scratch.
    """
)