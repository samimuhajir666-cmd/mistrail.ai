from datetime import datetime
import os
import noisereduce as nr
import numpy as np
import soundfile as sf
import streamlit as st
import torch
import torchaudio
import whisper

# Try importing SpeechBrain safely
try:
  from speechbrain.inference.enhancement import SpectralMaskEnhancement

  HAS_SPEECHBRAIN = True
except ImportError:
  HAS_SPEECHBRAIN = False

# Page Configuration
st.set_page_config(
    page_title="AI Audio Studio - Senior Task Pipeline",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ Dual-Layer Audio Cleaning & Whisper STT Pipeline")
st.write(
    "Senior Task: NoiseReduce + SpeechBrain AI Enhancement ➔ Local Whisper STT"
    " Comparison"
)

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


with st.spinner("⏳ Loading AI models into memory..."):
  model = load_whisper_model()
  sb_enhancer = load_speechbrain_enhancer()

# Input Section
input_method = st.radio(
    "Choose Input Method:", ["🔴 Live Record Audio", "📂 Upload Audio File"]
)

audio_file = None
if input_method == "🔴 Live Record Audio":
  audio_file = st.audio_input(
      "Record your English order-booking paragraph with background noise:"
  )
else:
  audio_file = st.file_uploader(
      "Upload noisy English audio file:", type=["wav", "mp3", "m4a"]
  )

if audio_file is not None:
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  raw_path = f"pipeline_output/raw_{timestamp}.wav"
  step1_path = f"pipeline_output/step1_cleaned_{timestamp}.wav"
  cleaned_path = f"pipeline_output/fully_polished_{timestamp}.wav"

  # Save raw audio safely
  with open(raw_path, "wb") as f:
    f.write(audio_file.getbuffer())

  st.markdown("### 🔊 Original Noisy Audio")
  st.audio(raw_path, format="audio/wav")

  if st.button("🚀 Run Dual-Layer AI Pipeline"):
    # --- STAGE 1: TRANSCRIBE ORIGINAL (BEFORE) ---
    with st.spinner("🤖 Transcribing Original Noisy Audio (Before Result)..."):
      try:
        original_result = model.transcribe(raw_path, language="en", fp16=False)
        original_transcript = original_result["text"].strip()
      except Exception as e:
        st.error(f"Whisper Error on Original Audio: {e}")
        original_transcript = "Error generating transcript."

    # --- STAGE 2: DUAL-LAYER CLEANING (NoiseReduce + SpeechBrain) ---
    with st.spinner(
        "🧹 Layer 1: Applying NoiseReduce for heavy background noise..."
    ):
      try:
        data, rate = sf.read(raw_path)
        if data.ndim > 1:
          data = np.mean(data, axis=1)

        step1_data = nr.reduce_noise(
            y=data.astype(np.float32), sr=rate, prop_decrease=0.85
        )
        sf.write(step1_path, step1_data, rate)
      except Exception as e:
        st.error(f"Layer 1 Error: {e}")
        step1_path = raw_path

    with st.spinner(
        "🧠 Layer 2: Running SpeechBrain AI for voice enhancement..."
    ):
      try:
        if HAS_SPEECHBRAIN and sb_enhancer is not None:
          enhanced_speech = sb_enhancer.enhance_file(step1_path)
          enhanced_np = enhanced_speech.squeeze(0).cpu().numpy()
          sf.write(cleaned_path, enhanced_np, 16000)
          st.success("✨ Dual-Layer AI Processing Completed Successfully!")
        else:
          cleaned_path = step1_path
          st.warning(
              "SpeechBrain not found/loaded. Using Layer-1 NoiseReduce output."
          )

        st.markdown("### 🎧 Fully Polished Audio (After Dual-Layer)")
        st.audio(cleaned_path, format="audio/wav")
      except Exception as e:
        st.error(f"Layer 2 Cleaning Error: {e}")
        cleaned_path = step1_path

    # --- STAGE 3: TRANSCRIBE CLEANED AUDIO (AFTER) ---
    with st.spinner(
        "🤖 Transcribing Cleaned Audio (After Dual-Layer Result)..."
    ):
      try:
        cleaned_result = model.transcribe(
            cleaned_path, language="en", fp16=False
        )
        cleaned_transcript = cleaned_result["text"].strip()
      except Exception as e:
        st.error(f"Whisper Error on Cleaned Audio: {e}")
        cleaned_transcript = "Error generating transcript."

    # --- STAGE 4: COMPARISON DISPLAY ---
    st.markdown("---")
    st.subheader("📊 Comparison of Transcriptions")

    col1, col2 = st.columns(2)

    with col1:
      st.markdown("#### ❌ Before (Raw Noisy Audio)")
      st.text_area(
          "Transcript from Noisy Audio:",
          value=original_transcript,
          height=200,
          key=f"orig_text_{timestamp}",
      )

    with col2:
      st.markdown("#### ✅ After (Dual-Layer Cleaned)")
      st.text_area(
          "Transcript from Cleaned Audio:",
          value=cleaned_transcript,
          height=200,
          key=f"clean_text_{timestamp}",
      )

    st.info(
        "💡 **Analysis Tip:** Compare both texts. The 'After' box should display"
        " cleaner text with accurate order-booking terms even with background"
        " noise present."
    )
