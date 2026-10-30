from datetime import datetime
import os
import librosa
import numpy as np
import pandas as pd
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

# Try importing FunASR safely
try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    HAS_FUNASR = True
except ImportError:
    HAS_FUNASR = False

# Page Configuration
st.set_page_config(
    page_title="AI Audio Studio",
    page_icon="🎙️",
    layout="wide",
)

# ====================================================
# SIDEBAR NAVIGATION
# ====================================================
st.sidebar.title("🛠️ Navigation")
app_mode = st.sidebar.radio(
    "Choose Tool Mode:",
    [
        "🎙️ Enhancement & Whisper STT",
        "🗣️ Speaker Diarization (FunASR)",
    ],
)

# ====================================================
# MODE 1: Noise Reduction & Whisper STT Pipeline
# ====================================================
if app_mode == "🎙️ Enhancement & Whisper STT":
    st.title("🎙️ Dual-Layer AI Pipeline + Context-Aware Whisper STT")
    st.write("DeepFilterNet (Noise Suppression) ➔ SpeechBrain (Enhancement) ➔ Smart Whisper STT")

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

    with st.spinner("⏳ Loading AI models into memory..."):
        whisper_model = load_whisper_model()
        df_model, df_state = load_deepfilter_model()
        sb_enhancer = load_speechbrain_enhancer()

    input_method = st.radio("Choose Input Method:", ["🔴 Live Record Audio", "📂 Upload Audio File"])
    audio_file = None

    if input_method == "🔴 Live Record Audio":
        audio_file = st.audio_input("Record your English order-booking paragraph:")
    else:
        audio_file = st.file_uploader("Upload noisy English audio file:", type=["wav", "mp3", "m4a"])

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
            with st.spinner("🤖 Transcribing Original Noisy Audio..."):
                try:
                    original_result = whisper_model.transcribe(raw_path, language="en", fp16=False)
                    original_transcript = original_result["text"].strip()
                except Exception as e:
                    st.error(f"Whisper Error: {e}")
                    original_transcript = "Error"

            with st.spinner("🧹 Layer 1: Running DeepFilterNet..."):
                try:
                    if HAS_DEEPFILTER and df_model is not None and df_state is not None:
                        audio_tensor, _ = load_audio(raw_path, sr=df_state.sr())
                        enhanced_df = enhance(df_model, df_state, audio_tensor)
                        save_audio(step1_path, enhanced_df, df_state.sr())
                    else:
                        step1_path = raw_path
                except Exception as e:
                    st.error(f"Layer 1 Error: {e}")
                    step1_path = raw_path

            with st.spinner("🧠 Layer 2: Running SpeechBrain..."):
                try:
                    if HAS_SPEECHBRAIN and sb_enhancer is not None:
                        enhanced_speech = sb_enhancer.enhance_file(step1_path)
                        enhanced_np = enhanced_speech.squeeze(0).cpu().numpy()
                        sf.write(cleaned_path, enhanced_np, 16000)
                        st.success("✨ Dual-Layer AI Cleaning Completed!")
                    else:
                        cleaned_path = step1_path
                    
                    st.markdown("### 🎧 Fully Polished Audio")
                    st.audio(cleaned_path, format="audio/wav")
                except Exception as e:
                    st.error(f"Layer 2 Error: {e}")
                    cleaned_path = step1_path

            with st.spinner("🤖 Transcribing Cleaned Audio..."):
                try:
                    order_context_prompt = "This is a food order conversation. Transcribe accurately."
                    cleaned_result = whisper_model.transcribe(cleaned_path, language="en", fp16=False, initial_prompt=order_context_prompt)
                    cleaned_transcript = cleaned_result["text"].strip()
                except Exception as e:
                    st.error(f"Whisper Error: {e}")
                    cleaned_transcript = "Error"

            st.markdown("---")
            st.subheader("📊 Comparison of Transcriptions")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### ❌ Before")
                st.text_area("Noisy Transcript:", value=original_transcript, height=150)
            with col2:
                st.markdown("#### ✅ After")
                st.text_area("Cleaned Transcript:", value=cleaned_transcript, height=150)


# ====================================================
# MODE 2: FunASR (Speaker Diarization without Token)
# ====================================================
elif app_mode == "🗣️ Speaker Diarization (FunASR)":
    st.title("🗣️ Multi-Speaker Diarization (FunASR)")
    st.write("Identifies WHO spoke WHEN using Alibaba's FunASR. No Hugging Face token required.")

    if not HAS_FUNASR:
        st.error("FunASR is not installed. Please check requirements.txt and Python version (3.11).")
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

    with st.spinner("⏳ Loading FunASR Models (This may take a few minutes first time)..."):
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
            
        st.markdown("### 🔊 Original Audio")
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
                
                st.markdown("### 📝 Detailed View")
                for row in results_data:
                    st.markdown(f"**[{row['Time']}] {row['Speaker']}:** {row['Text']}")
            else:
                st.warning("No speech segments found. Check the audio or FunASR output.")

            if os.path.exists(temp_upload_path): os.remove(temp_upload_path)
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
