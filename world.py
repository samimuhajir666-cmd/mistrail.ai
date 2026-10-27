from datetime import datetime
import os
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st
import whisper
import torch
import torchaudio

# Try importing SpeechBrain
try:
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    HAS_SPEECHBRAIN = True
except ImportError:
    HAS_SPEECHBRAIN = False

# Page Configuration
st.set_page_config(
    page_title="AI Audio Studio - Senior Task", page_icon="🎙️", layout="wide"
)

st.title("🎙️ Audio Cleaning & Whisper STT Comparison (SpeechBrain + Noisereduce)")
st.write("Senior Task: Compare Original vs AI-Enhanced/Cleaned Transcription")

os.makedirs("pipeline_output", exist_ok=True)

# Cache the whisper model so it doesn't reload every time
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Cache the SpeechBrain model so it loads only once
@st.cache_resource
def load_speechbrain_enhancer():
    if HAS_SPEECHBRAIN:
        try:
            return SpectralMaskEnhancement.from_hparams(
                source="speechbrain/metricgan-plus-voicebank"
            )
        except Exception as e:
            st.warning(f"Could not load SpeechBrain model: {e}")
            return None
    return None

model = load_whisper_model()
sb_enhancer = load_speechbrain_enhancer()

# Sidebar / UI Option to select cleaning engine
cleaning_method = st.radio(
    "Choose Noise Cleaning Engine:",
    ["🧹 Traditional NoiseReduce", "🧠 SpeechBrain AI (MetricGAN+)"]
)

# Input Section
input_method = st.radio(
    "Choose Input Method:", ["🔴 Live Record Audio", "📂 Upload Audio File"]
)

audio_file = None
if input_method == "🔴 Live Record Audio":
    audio_file = st.audio_input("Record your English paragraph with background noise:")
else:
    audio_file = st.file_uploader(
        "Upload noisy English audio file:", type=["wav", "mp3", "m4a"]
    )

if audio_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = f"pipeline_output/raw_{timestamp}.wav"
    cleaned_path = f"pipeline_output/cleaned_{timestamp}.wav"

    # Save raw audio
    with open(raw_path, "wb") as f:
        f.write(audio_file.getbuffer())

    st.markdown("### 🔊 Original Noisy Audio")
    st.audio(raw_path, format="audio/wav")

    if st.button("🚀 Run Cleaning & STT Pipeline"):
        
        # --- STAGE 1: TRANSCRIBE ORIGINAL (NOISY) AUDIO ---
        with st.spinner("🤖 Transcribing Original Noisy Audio (Before)..."):
            try:
                original_result = model.transcribe(raw_path, language="en", fp16=False)
                original_transcript = original_result["text"].strip()
            except Exception as e:
                st.error(f"Whisper Error on Original Audio: {e}")
                original_transcript = "Error generating transcript."

        # --- STAGE 2: AUDIO CLEANING (SpeechBrain or NoiseReduce) ---
        with st.spinner(f"🧹 Applying {cleaning_method}..."):
            try:
                if "SpeechBrain" in cleaning_method and HAS_SPEECHBRAIN and sb_enhancer is not None:
                    # Run SpeechBrain AI Enhancement
                    enhanced_speech = sb_enhancer.enhance_file(raw_path)
                    enhanced_np = enhanced_speech.squeeze(0).cpu().numpy()
                    sf.write(cleaned_path, enhanced_np, 16000)
                    st.success("✨ SpeechBrain AI Enhancement completed successfully!")
                else:
                    if "SpeechBrain" in cleaning_method and not HAS_SPEECHBRAIN:
                        st.warning("SpeechBrain library not installed. Falling back to NoiseReduce.")
                    
                    # Fallback / Traditional NoiseReduce
                    data, rate = sf.read(raw_path)
                    if data.ndim > 1:    
                        data = np.mean(data, axis=1)

                    cleaned_data = nr.reduce_noise(
                        y=data.astype(np.float32), sr=rate, prop_decrease=0.85
                    )
                    sf.write(cleaned_path, cleaned_data, rate)
                    st.success("✨ Traditional NoiseReduce completed successfully!")
                
                st.markdown("### 🎧 Cleaned Audio")
                st.audio(cleaned_path, format="audio/wav")
            except Exception as e:
                st.error(f"Cleaning Error: {e}")
                cleaned_path = raw_path

        # --- STAGE 3: TRANSCRIBE CLEANED AUDIO ---
        with st.spinner("🤖 Transcribing Cleaned Audio (After)..."):
            try:
                cleaned_result = model.transcribe(cleaned_path, language="en", fp16=False)
                cleaned_transcript = cleaned_result["text"].strip()
            except Exception as e:
                st.error(f"Whisper Error on Cleaned Audio: {e}")
                cleaned_transcript = "Error generating transcript."

        # --- STAGE 4: COMPARISON DISPLAY ---
        st.markdown("---")
        st.subheader("📊 Comparison of Transcriptions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ❌ Before (Original Noisy Audio)")
            st.text_area("Transcript from Noisy Audio:", value=original_transcript, height=200, key="original_text")
            
        with col2:
            st.markdown("#### ✅ After (Cleaned Audio)")
            st.text_area("Transcript from Cleaned Audio:", value=cleaned_transcript, height=200, key="cleaned_text")

        st.info("💡 **Analysis Tip:** Compare both texts above to see how much word accuracy SpeechBrain or NoiseReduce adds!")
