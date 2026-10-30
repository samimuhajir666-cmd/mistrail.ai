import streamlit as st
import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datetime import datetime

# ====================================================
# SAFE IMPORT FUNASR
# ====================================================
try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    HAS_FUNASR = True
except ImportError as e:
    HAS_FUNASR = False
    IMPORT_ERROR = str(e)

st.set_page_config(page_title="FunASR Test", page_icon="🎙️", layout="wide")
st.title("🗣️ FunASR Multi-Speaker Diarization")

if not HAS_FUNASR:
    st.error(f"FunASR is not installed. Error: {IMPORT_ERROR}")
    st.info("Please ensure your requirements.txt only contains the tested minimal packages and Streamlit is using Python 3.11.")
    st.stop()

@st.cache_resource
def load_funasr_model():
    try:
        model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            spk_model="cam++",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        return model
    except Exception as e:
        st.error(f"Failed to load FunASR model: {e}")
        return None

with st.spinner("⏳ Loading FunASR Model (May take a few minutes first time)..."):
    funasr_model = load_funasr_model()

if funasr_model is None:
    st.stop()

uploaded_file = st.file_uploader("Upload Audio File (.wav, .mp3)", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_upload_path = f"temp_upload_{timestamp}.wav"
    temp_wav_path = f"temp_audio_{timestamp}.wav"
    
    with open(temp_upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.audio(temp_upload_path, format="audio/wav")
    
    if st.button("Process & Diarize Audio 🚀", type="primary"):
        with st.spinner("Step 1: Converting audio to 16kHz WAV..."):
            try:
                y, sr = librosa.load(temp_upload_path, sr=16000)
                sf.write(temp_wav_path, y, sr)
            except Exception as e:
                st.error(f"Audio Conversion Error: {e}")
                st.stop()

        with st.spinner("Step 2: Running FunASR Diarization..."):
            try:
                res = funasr_model.generate(
                    input=temp_wav_path,
                    cache={},
                    language="auto",
                    use_itn=True,
                    batch_size_s=60,
                    merge_vad=True,
                    merge_length_s=15,
                )
            except Exception as e:
                st.error(f"FunASR Processing Error: {e}")
                st.stop()

        st.success("Diarization Complete! 🎉")
        st.markdown("---")
        st.subheader("📊 Speaker-Attributed Transcript")
        
        results_data = []
        if res and len(res) > 0 and "sentence_info" in res[0]:
            for sent in res[0]["sentence_info"]:
                text = rich_transcription_postprocess(sent["text"])
                speaker = sent.get("spk", "Unknown")
                
                start_sec = sent["start"] / 1000.0
                end_sec = sent["end"] / 1000.0
                start_str = f"{int(start_sec//60):02d}:{int(start_sec%60):02d}"
                end_str = f"{int(end_sec//60):02d}:{int(end_sec%60):02d}"
                
                results_data.append({
                    "Time": f"{start_str} - {end_str}",
                    "Speaker": f"Speaker {speaker}",
                    "Text": text
                })

        if results_data:
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)
            for row in results_data:
                st.markdown(f"**[{row['Time']}] {row['Speaker']}:** {row['Text']}")
        else:
            st.warning("No speech segments found.")

        if os.path.exists(temp_upload_path): os.remove(temp_upload_path)
        if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
