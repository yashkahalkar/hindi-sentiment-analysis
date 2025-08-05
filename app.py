# app.py (Final, Polished Version)

import streamlit as st
import google.generativeai as genai
import whisper
import os
import joblib
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
import tempfile # <<< IMPROVEMENT: Import tempfile for robust audio handling

# --- Page Configuration ---
# <<< IMPROVEMENT: Made title more specific to the app's function >>>
st.set_page_config(layout="wide", page_title="Hindi Emotion Analysis")

# --- API Configuration ---
# Load environment variables (this part is correct)
try:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"]
except (FileNotFoundError, KeyError):
    st.error("API Key not found. Please set GOOGLE_API_KEY in your environment or Streamlit secrets.")
    st.stop()
    
genai.configure(api_key=GOOGLE_API_KEY)
EMBEDDING_MODEL = 'models/embedding-001'

# --- Model Loading ---

@st.cache_resource
def load_model_from_hf():
    """ Downloads and caches the SVM model from Hugging Face Hub. """
    
    # --- CRITICAL: REPLACE WITH YOUR HUGGING FACE REPO ID ---
    repo_id = "your-hf-username/your-model-repo-name"  # Example: "yashkahalkar/hindi-emotion-svm"
    filename = "svm_emotion_model_tuned.pkl"
    # ---------------------------------------------------------
    
    try:
        with st.spinner(f"Downloading model '{filename}' from Hugging Face Hub... This may take a moment."):
            model_path = hf_hub_download(repo_id="yashkahalkar/hindi_sentiment_analysis", filename="svm_emotion_model.pkl")
        
        model_data = joblib.load(model_path)
        st.success("Model loaded successfully!")
        return model_data
    except Exception as e:
        st.error(f"Error loading model from Hugging Face Hub: {e}")
        st.error("Please ensure the repo_id and filename are correct and the repository is public.")
        return None

# Load the model data
model_data = load_model_from_hf()

# Unpack models and metadata, stopping the app if loading failed
if model_data:
    svm_model = model_data['model']
    id_to_label = model_data['id_to_label']
else:
    st.header("Model could not be loaded. The application cannot proceed.")
    st.stop()

@st.cache_resource
def load_whisper_model():
    """Loads and caches the Whisper model."""
    with st.spinner("Loading speech-to-text model..."):
        model = whisper.load_model("base")
    return model

whisper_model = load_whisper_model()

# --- Core Processing Functions ---

def get_gemini_embedding(text):
    """Generates text embedding using Gemini API."""
    try:
        return np.array(genai.embed_content(model=EMBEDDING_MODEL, content=text)['embedding'])
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

def predict_emotion(text):
    """Predicts emotion for a given text."""
    if not text or not text.strip():
        return None
    
    embedding = get_gemini_embedding(text)
    if embedding is not None:
        prediction_id = svm_model.predict(embedding.reshape(1, -1))[0]
        emotion = id_to_label[prediction_id]
        return emotion
    return None

def transcribe_audio(audio_file_path):
    """Transcribes Hindi audio file to text."""
    try:
        return whisper_model.transcribe(audio_file_path, language='hi', fp16=False)['text']
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# --- Streamlit UI ---

st.title("Hindi Emotion Analysis 🗣️")
st.markdown("Analyze the emotion of Hindi text or audio using Gemini, Whisper, and a custom SVM model.")

EMOJI_MAP = {
    'neutral': '😐', 'sadness': '🥲', 'anger': '😠',
    'surprise': '😲', 'joy': '😊', 'fear': '😨', 'disgust': '🤢'
}

tab1, tab2 = st.tabs(["✍️ **Text Input**", "🎤 **Audio Input**"])

with tab1:
    st.header("Analyze Hindi Text")
    hindi_text = st.text_area("Enter Hindi text below (जैसे: 'आज मैं बहुत खुश हूँ'):", height=150)
    
    if st.button("Analyze Emotion", key="text_button", type="primary"):
        if hindi_text and hindi_text.strip():
            with st.spinner("Analyzing..."):
                predicted_emotion = predict_emotion(hindi_text)
                if predicted_emotion:
                    emoji = EMOJI_MAP.get(predicted_emotion, '💬')
                    st.success(f"**Predicted Emotion: {predicted_emotion.capitalize()} {emoji}**")
                else:
                    st.warning("Could not analyze the text.")
        else:
            st.warning("Please enter some text.")

with tab2:
    st.header("Analyze Hindi Audio")
    uploaded_audio = st.file_uploader("Upload a Hindi audio file...", type=['wav', 'mp3', 'm4a'])
    
    if uploaded_audio:
        st.audio(uploaded_audio)
        if st.button("Analyze Emotion", key="audio_button", type="primary"):
            with st.spinner("Transcribing audio... (this can take a moment)"):
                # <<< IMPROVEMENT: Use a robust temporary file for safety >>>
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(uploaded_audio.getvalue())
                    temp_file_path = temp_file.name

            transcribed_text = transcribe_audio(temp_file_path)
            os.remove(temp_file_path) # Clean up the unique temporary file
            
            # <<< IMPROVEMENT: More robust check for transcription result >>>
            if transcribed_text and transcribed_text.strip():
                st.info(f"**Transcribed Text:** {transcribed_text}")
                with st.spinner("Analyzing emotion..."):
                    predicted_emotion = predict_emotion(transcribed_text)
                    if predicted_emotion:
                        emoji = EMOJI_MAP.get(predicted_emotion, '💬')
                        st.success(f"**Predicted Emotion: {predicted_emotion.capitalize()} {emoji}**")
                    else:
                        st.warning("Could not analyze the transcribed text.")
            else:
                st.error("Audio transcription failed. The audio might be silent or not in Hindi.")