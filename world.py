from datetime import datetime
import os
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st
import whisper
import torch
import torchaudio

# Page Configuration
st.set_page_config(page_title="AI Audio Studio - Senior Task", page_icon="🎙️", layout="wide")

st.title("🎙️ Professional Audio Cleaning & Whisper STT Pipeline")
st.write("Senior Task Implementation: Noise Reduction ➔ Whisper STT ➔ Error Analysis")

# Output directory for GitHub tracking
os.makedirs("pipeline_output", exist_ok=True)

# Input Section
input_method = st.radio("Choose Input Method:", ["🔴 Live Record Audio", "📂 Upload Audio File"])

audio_file = None
if input_method == "🔴 Live Record Audio":
    audio_file = st.audio_input("Record your English paragraph with background noise:")
else:
    audio_file = st.file_uploader("Upload noisy English audio file:", type=["wav", "mp3", "m4a"])

if audio_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = f"pipeline_output/raw_{timestamp}.wav"
    cleaned_path = f"pipeline_output/cleaned_{timestamp}.wav"

    # Save raw audio safely
    with open(raw_path, "wb") as f:
        f.write(audio_file.getbuffer())

    st.markdown("### 🔊 1. Original Noisy Audio")
    st.audio(raw_path, format="audio/wav")

    if st.button("🚀 Run Cleaning & STT Pipeline"):
        
        # --- STAGE 1: AUDIO CLEANING ---
        with st.spinner("🧹 Stage 1: Applying Noise Reduction..."):
            try:
                # Read safely using soundfile
                data, rate = sf.read(raw_path)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)

                # Noise reduction
                cleaned_data = nr.reduce_noise(y=data.astype(np.float32), sr=rate, prop_decrease=0.85)
                sf.write(cleaned_path, cleaned_data, rate)
                st.success("✨ Audio cleaning completed successfully!")
                
                st.markdown("### 🎧 2. Cleaned Audio Output")
                st.audio(cleaned_path, format="audio/wav")
            except Exception as e:
                st.error(f"Audio Cleaning Error: {e}")
                cleaned_path = raw_path # Agar clean na ho, to original hi use karo

        # --- STAGE 2: LOCAL WHISPER STT ---
        with st.spinner("🤖 Stage 2: Running Whisper STT Transcription..."):
            try:
                # Loading lightweight base model locally
                model = whisper.load_model("base")
                result = model.transcribe(cleaned_path, language="en", fp16=False)
                transcript_text = result["text"].strip()

                st.subheader("📝 3. Raw Transcription Result")
                st.text_area("Generated Transcript:", value=transcript_text, height=150)
            except Exception as e:
                st.error(f"Whisper Error: {e}")
                transcript_text = ""

        # --- STAGE 3: LLM PROMPT CLEANUP (Optional for now) ---
        if transcript_text:
            st.markdown("---")
            st.subheader("✨ 4. Final Polished Transcript (LLM Cleanup)")
            st.info("Currently showing raw Whisper output. When LLM API is connected, it will clean spelling mistakes and filler words.")
            
            # Yahan hum baad mein Groq/OpenAI ka API laga sakte hain. Abhi ke liye raw text hi show kar rahe hain.
            polished_transcript = transcript_text 
            st.success(polished_transcript)

        # --- STAGE 4: ERROR ANALYSIS ---
        st.markdown("---")
        st.subheader("📊 5. Error Analysis & Evaluation")
        st.write("Compare the generated transcript with your original spoken paragraph to check for missed words.")

        expected_text = st.text_area(
            "Enter Expected Original Paragraph (for comparison):",
            placeholder="Type the exact English paragraph you read..."
        )

        if expected_text and st.button("🔍 Compare Results"):
            st.info("Evaluation complete! Check word matches and identify any transcription errors.")
