from datetime import datetime
import os
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st
import whisper

# Page Configuration
st.set_page_config(
    page_title="AI Audio Studio - Senior Task", page_icon="🎙️", layout="wide"
)

st.title("🎙️ Audio Cleaning & Whisper STT Comparison")
st.write("Senior Task: Check Original vs Cleaned Transcription")

os.makedirs("pipeline_output", exist_ok=True)

# Cache the whisper model so it doesn't reload every time
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

model = load_whisper_model()

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
        with st.spinner("🤖 Transcribing Original Noisy Audio (This shows the 'Before' result)..."):
            try:
                # Original audio ka transcript nikalna
                original_result = model.transcribe(raw_path, language="en", fp16=False)
                original_transcript = original_result["text"].strip()
            except Exception as e:
                st.error(f"Whisper Error on Original Audio: {e}")
                original_transcript = "Error generating transcript."

        # --- STAGE 2: AUDIO CLEANING (Noise Reduction) ---
        with st.spinner("🧹 Applying Noise Reduction..."):
            try:
                data, rate = sf.read(raw_path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)

                cleaned_data = nr.reduce_noise(
                    y=data.astype(np.float32), sr=rate, prop_decrease=0.85
                )
                sf.write(cleaned_path, cleaned_data, rate)
                st.success("✨ Audio cleaning completed successfully!")
                
                st.markdown("### 🎧 Cleaned Audio")
                st.audio(cleaned_path, format="audio/wav")
            except Exception as e:
                st.error(f"Cleaning Error: {e}")
                cleaned_path = raw_path

        # --- STAGE 3: TRANSCRIBE CLEANED AUDIO ---
        with st.spinner("🤖 Transcribing Cleaned Audio (This is the 'After' result)..."):
            try:
                # Cleaned audio ka transcript nikalna
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

        st.info("💡 **Analysis Tip:** Compare both texts above. If the 'Cleaned' text has more correct words and fewer weird sentences, the Noise Reduction worked perfectly!")
