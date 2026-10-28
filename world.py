from datetime import datetime
import os
import librosa
import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
import soundfile as sf
import streamlit as st
import torch
import torchaudio
import whisper

# Try importing DeepFilterNet safely
try:
  from df.enhance import enhance, init_df
  from df.io import load_audio, save_audio

  HAS_DEEPFILTER = True
except ImportError:
  HAS_DEEPFILTER = False

# Try importing SpeechBrain safely
try:
  from speechbrain.inference.enhancement import SpectralMaskEnhancement

  HAS_SPEECHBRAIN = True
except ImportError:
  HAS_SPEECHBRAIN = False

# Page Configuration
st.set_page_config(
    page_title="AI Audio Studio",
    page_icon="🎙️",
    layout="wide",
)

# ====================================================
# SIDEBAR NAVIGATION (Dono ko alag rakhne ke liye)
# ====================================================
st.sidebar.title("🛠️ Navigation")
app_mode = st.sidebar.radio(
    "Choose Tool Mode:",
    [
        "🎙️ Enhancement & Whisper STT",
        "🗣️ Speaker Classification (Pyannote)",
    ],
)

# ====================================================
# MODE 1: Noise Reduction & Whisper STT Pipeline
# ====================================================
if app_mode == "🎙️ Enhancement & Whisper STT":
  st.title("🎙️ Dual-Layer AI Pipeline + Context-Aware Whisper STT")
  st.write(
      "DeepFilterNet (Noise Suppression) ➔ SpeechBrain (Enhancement) ➔ Smart"
      " Whisper STT"
  )

  os.makedirs("pipeline_output", exist_ok=True)

  @st.cache_resource
  def load_whisper_model():
    return whisper.load_model("base")

  @st.cache_resource
  def load_deepfilter_model():
    if HAS_DEEPFILTER:
      try:
        model, df_state, _ = init_df()
        return model, df_state
      except Exception as e:
        st.warning(f"Could not load DeepFilterNet model: {e}")
        return None, None
    return None, None

  @st.cache_resource
  def load_speechbrain_enhancer():
    if HAS_SPEECHBRAIN:
      try:
        return SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank"
        )
      except Exception as e:
        st.warning(f"Could not load SpeechBrain model: {e}")
        return None, None
    return None, None

  with st.spinner(
      "⏳ Loading AI models into memory (DeepFilter + SpeechBrain +"
      " Whisper)..."
  ):
    whisper_model = load_whisper_model()
    df_model, df_state = load_deepfilter_model()
    sb_enhancer = load_speechbrain_enhancer()

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
    step1_path = f"pipeline_output/layer1_deepfiltered_{timestamp}.wav"
    cleaned_path = f"pipeline_output/fully_polished_{timestamp}.wav"

    with open(raw_path, "wb") as f:
      f.write(audio_file.getbuffer())

    st.markdown("### 🔊 Original Noisy Audio")
    st.audio(raw_path, format="audio/wav")

    if st.button("🚀 Run Complete AI Pipeline"):
      # --- STAGE 1: TRANSCRIBE ORIGINAL (BEFORE) ---
      with st.spinner("🤖 Transcribing Original Noisy Audio (Before Result)..."):
        try:
          original_result = whisper_model.transcribe(
              raw_path, language="en", fp16=False
          )
          original_transcript = original_result["text"].strip()
        except Exception as e:
          st.error(f"Whisper Error on Original Audio: {e}")
          original_transcript = "Error generating transcript."

      # --- STAGE 2: LAYER 1 - DEEPFILTERNET ---
      with st.spinner(
          "🧹 Layer 1: Running DeepFilterNet for heavy noise suppression..."
      ):
        try:
          if HAS_DEEPFILTER and df_model is not None and df_state is not None:
            audio_tensor, _ = load_audio(raw_path, sr=df_state.sr())
            enhanced_df = enhance(df_model, df_state, audio_tensor)
            save_audio(step1_path, enhanced_df, df_state.sr())
          else:
            step1_path = raw_path
            st.warning("DeepFilterNet skipped, using raw audio.")
        except Exception as e:
          st.error(f"Layer 1 Error: {e}")
          step1_path = raw_path

      # --- STAGE 3: LAYER 2 - SPEECHBRAIN ---
      with st.spinner(
          "🧠 Layer 2: Running SpeechBrain for voice clarity & enhancement..."
      ):
        try:
          if HAS_SPEECHBRAIN and sb_enhancer is not None:
            enhanced_speech = sb_enhancer.enhance_file(step1_path)
            enhanced_np = enhanced_speech.squeeze(0).cpu().numpy()
            sf.write(cleaned_path, enhanced_np, 16000)
            st.success("✨ Dual-Layer AI Cleaning Completed Successfully!")
          else:
            cleaned_path = step1_path
            st.warning("SpeechBrain skipped, using Layer 1 output.")

          st.markdown("### 🎧 Fully Polished Audio (After Dual-Layer)")
          st.audio(cleaned_path, format="audio/wav")
        except Exception as e:
          st.error(f"Layer 2 Error: {e}")
          cleaned_path = step1_path

      # --- STAGE 4: TRANSCRIBE CLEANED AUDIO WITH PROMPT (AFTER) ---
      with st.spinner(
          "🤖 Transcribing Cleaned Audio with Order-Booking Context Prompt..."
      ):
        try:
          order_context_prompt = (
              "This is a food or item order-booking conversation. Please"
              " transcribe the customer's order, quantities, and items"
              " accurately, ignoring any background noise, static, or"
              " irrelevant chatter."
          )
          cleaned_result = whisper_model.transcribe(
              cleaned_path,
              language="en",
              fp16=False,
              initial_prompt=order_context_prompt,
          )
          cleaned_transcript = cleaned_result["text"].strip()
        except Exception as e:
          st.error(f"Whisper Error on Cleaned Audio: {e}")
          cleaned_transcript = "Error generating transcript."

      # --- STAGE 5: COMPARISON DISPLAY ---
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
        st.markdown("#### ✅ After (Dual-Layer + Smart Prompt)")
        st.text_area(
            "Transcript from Cleaned Audio:",
            value=cleaned_transcript,
            height=200,
            key=f"clean_text_{timestamp}",
        )

      st.info(
          "💡 **Analysis Tip:** The 'After' text now leverages both physical"
          " noise reduction layers and semantic prompting for optimal"
          " order-booking accuracy."
      )

# ====================================================
# MODE 2: Pure Pyannote Speaker Classification (Independent)
# ====================================================
elif app_mode == "🗣️ Speaker Classification (Pyannote)":
  st.title("🗣️ Multi-Speaker Classification (Pyannote Diarization)")
  st.write(
      "Upload audio to see WHO spoke WHEN. (Completely independent tool)"
  )

  # Token Input in Sidebar
  with st.sidebar:
    st.header("🔑 Pyannote Auth")
    HF_TOKEN = os.getenv("HF_TOKEN") or st.text_input(
        "Hugging Face Token:", type="password"
    )
    st.markdown("[Get Token Here](https://huggingface.co/settings/tokens)")

  if HF_TOKEN:

    @st.cache_resource
    def load_pyannote_pipeline(token):
      try:
        # Fixed to modern 'token' argument instead of deprecated 'use_auth_token'
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", token=token
        )
      except Exception as e:
        st.error(f"Pipeline load error: {e}")
        return None

    pipeline = load_pyannote_pipeline(HF_TOKEN)

    if pipeline is not None:
      uploaded_file = st.file_uploader(
          "Upload Audio File (.wav, .mp3)", type=["wav", "mp3"]
      )
      num_speakers = st.number_input(
          "Expected Speakers (0 = Auto-detect)",
          min_value=0,
          max_value=10,
          value=0,
      )

      if uploaded_file is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_upload_path = f"temp_upload_{timestamp}.wav"
        temp_wav_path = f"temp_audio_{timestamp}.wav"

        with open(temp_upload_path, "wb") as f:
          f.write(uploaded_file.getbuffer())

        st.markdown("### 🔊 Original Audio")
        st.audio(temp_upload_path, format="audio/wav")

        if st.button("Process & Classify Audio 🚀", type="primary"):
          with st.spinner(
              "AI Aawazein alag alag kar raha hai... Please wait..."
          ):
            try:
              y, sr = librosa.load(temp_upload_path, sr=16000)
              sf.write(temp_wav_path, y, sr)

              if num_speakers > 0:
                diarization = pipeline(
                    temp_wav_path, num_speakers=num_speakers
                )
              else:
                diarization = pipeline(temp_wav_path)

              st.success("Classification Complete! 🎉")
              st.markdown("---")
              st.subheader("📊 Speaker Timeline Results")

              results_data = []
              for turn, _, speaker in diarization.itertracks(yield_label=True):
                results_data.append({
                    "Start Time": f"{turn.start:.2f}s",
                    "End Time": f"{turn.end:.2f}s",
                    "Duration": f"{turn.end - turn.start:.2f}s",
                    "Speaker": speaker,
                })

              if results_data:
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True)
                st.info(
                    f"🗣️ Total {df['Speaker'].nunique()} speakers detected."
                )
              else:
                st.warning("No speech detected.")

            except Exception as e:
              st.error(f"Processing Error: {e}")
            finally:
              if os.path.exists(temp_upload_path):
                os.remove(temp_upload_path)
              if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
  else:
    st.info("👈 Please enter your Hugging Face Token in the sidebar.")
