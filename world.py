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
# SIDEBAR NAVIGATION
# ====================================================
st.sidebar.title("🛠️ Navigation")
app_mode = st.sidebar.radio(
    "Choose Tool Mode:",
    [
        "🎙️ Enhancement & Whisper STT",
        "🗣️ Speaker Diarization + STT (Merged)",
    ],
)

# ====================================================
# MODE 1: Noise Reduction & Whisper STT Pipeline
# ====================================================
if app_mode == "🎙️ Enhancement & Whisper STT":
    st.title("🎙️ Dual-Layer AI Pipeline + Context-Aware Whisper STT")
    st.write(
        "DeepFilterNet (Noise Suppression) ➔ SpeechBrain (Enhancement) ➔ Smart Whisper STT"
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

    with st.spinner("⏳ Loading AI models into memory..."):
        whisper_model = load_whisper_model()
        df_model, df_state = load_deepfilter_model()
        sb_enhancer = load_speechbrain_enhancer()

    input_method = st.radio(
        "Choose Input Method:", ["🔴 Live Record Audio", "📂 Upload Audio File"]
    )
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
            # --- STAGE 1: TRANSCRIBE ORIGINAL ---
            with st.spinner("🤖 Transcribing Original Noisy Audio..."):
                try:
                    original_result = whisper_model.transcribe(raw_path, language="en", fp16=False)
                    original_transcript = original_result["text"].strip()
                except Exception as e:
                    st.error(f"Whisper Error: {e}")
                    original_transcript = "Error"

            # --- STAGE 2: LAYER 1 - DEEPFILTERNET ---
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

            # --- STAGE 3: LAYER 2 - SPEECHBRAIN ---
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

            # --- STAGE 4: TRANSCRIBE CLEANED AUDIO ---
            with st.spinner("🤖 Transcribing Cleaned Audio..."):
                try:
                    order_context_prompt = "This is a food order conversation. Transcribe accurately."
                    cleaned_result = whisper_model.transcribe(cleaned_path, language="en", fp16=False, initial_prompt=order_context_prompt)
                    cleaned_transcript = cleaned_result["text"].strip()
                except Exception as e:
                    st.error(f"Whisper Error: {e}")
                    cleaned_transcript = "Error"

            # --- STAGE 5: COMPARISON ---
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
# MODE 2: Merged Diarization + STT
# ====================================================
elif app_mode == "🗣️ Speaker Diarization + STT (Merged)":
    st.title("🗣️ Speaker-Attributed Transcript (Pyannote + Whisper)")
    st.write("Identifies WHO spoke WHEN and merges it with the transcribed TEXT.")

    # Token Logic
    HF_TOKEN = os.getenv("HF_TOKEN")
    if "HF_TOKEN" in st.secrets:
        HF_TOKEN = st.secrets["HF_TOKEN"]
    if not HF_TOKEN:
        HF_TOKEN = "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" # Fallback token

    os.environ["HF_TOKEN"] = HF_TOKEN

    @st.cache_resource
    def load_whisper_model():
        return whisper.load_model("base")

    @st.cache_resource
    def load_pyannote_pipeline():
        try:
            # Pyannote 3.x/4.x token handling
            return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=os.environ["HF_TOKEN"])
        except Exception as e:
            st.error(f"Pipeline Load Error: {e}")
            return None

    with st.spinner("⏳ Loading AI models (Whisper + Pyannote)..."):
        whisper_model = load_whisper_model()
        pipeline = load_pyannote_pipeline()

    if pipeline is None:
        st.error("Could not load Pyannote pipeline. Check token & accepted terms.")
        st.stop()

    uploaded_file = st.file_uploader("Upload Audio File (.wav, .mp3)", type=["wav", "mp3"])
    num_speakers = st.number_input("Expected Speakers (0 = Auto-detect)", min_value=0, max_value=10, value=0)
    
    if uploaded_file is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_upload_path = f"temp_upload_{timestamp}.wav"
        temp_wav_path = f"temp_audio_{timestamp}.wav"
        
        with open(temp_upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.markdown("### 🔊 Original Audio")
        st.audio(temp_upload_path, format="audio/wav")
        
        if st.button("Process & Merge Audio 🚀", type="primary"):
            with st.spinner("Step 1: Running Pyannote Diarization..."):
                try:
                    y, sr = librosa.load(temp_upload_path, sr=16000)
                    sf.write(temp_wav_path, y, sr)
                    
                    if num_speakers > 0:
                        diarization_output = pipeline(temp_wav_path, num_speakers=num_speakers)
                    else:
                        diarization_output = pipeline(temp_wav_path)
                        
                    # ==========================================
                    # THE FIX: Handle Pyannote 4.x DiarizeOutput
                    # ==========================================
                    if hasattr(diarization_output, "speaker_diarization"):
                        # Pyannote 4.x returns DiarizeOutput wrapper
                        annotation = diarization_output.speaker_diarization
                    else:
                        # Pyannote 3.x returns Annotation directly
                        annotation = diarization_output
                        
                    # Now safely extract segments
                    diarization_segments = []
                    for turn, _, speaker in annotation.itertracks(yield_label=True):
                        diarization_segments.append({
                            "start": turn.start,
                            "end": turn.end,
                            "speaker": speaker
                        })
                    diarization_segments = sorted(diarization_segments, key=lambda x: x["start"])
                    
                except Exception as e:
                    st.error(f"Pyannote Error: {e}")
                    st.stop()

            with st.spinner("Step 2: Running Whisper Transcription..."):
                try:
                    whisper_result = whisper_model.transcribe(temp_wav_path, language="en", fp16=False)
                    transcript_segments = whisper_result.get("segments", [])
                except Exception as e:
                    st.error(f"Whisper Error: {e}")
                    st.stop()

            with st.spinner("Step 3: Merging Diarization & STT Results..."):
                # Merge Logic
                fill_nearest = False
                merged_results = []
                
                for seg in transcript_segments:
                    seg_start = seg.get("start", 0.0)
                    seg_end = seg.get("end", 0.0)
                    speaker_overlap = {}
                    
                    for dia in diarization_segments:
                        intersection = min(dia["end"], seg_end) - max(dia["start"], seg_start)
                        if intersection <= 0:
                            continue
                        speaker = dia["speaker"]
                        speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + intersection
                        
                    if speaker_overlap:
                        assigned_speaker = max(speaker_overlap.items(), key=lambda x: x[1])[0]
                    elif fill_nearest and diarization_segments:
                        midpoint = (seg_start + seg_end) / 2
                        nearest = min(diarization_segments, key=lambda x: abs(((x["start"] + x["end"]) / 2) - midpoint))
                        assigned_speaker = nearest["speaker"]
                    else:
                        assigned_speaker = "UNKNOWN"
                        
                    start_str = f"{int(seg_start//60):02d}:{int(seg_start%60):02d}"
                    end_str = f"{int(seg_end//60):02d}:{int(seg_end%60):02d}"
                    
                    merged_results.append({
                        "Time": f"{start_str} - {end_str}",
                        "Speaker": assigned_speaker,
                        "Text": seg["text"].strip()
                    })

                # --- DISPLAY RESULTS ---
                st.success("Merging Complete! 🎉")
                st.markdown("---")
                st.subheader("📊 Speaker-Attributed Transcript")
                
                if merged_results:
                    df = pd.DataFrame(merged_results)
                    st.dataframe(df, use_container_width=True)
                    
                    st.markdown("### 📝 Detailed View")
                    for row in merged_results:
                        st.markdown(f"**[{row['Time']}] {row['Speaker']}:** {row['Text']}")
                else:
                    st.warning("No speech segments found to merge.")

            # Cleanup
            if os.path.exists(temp_upload_path): os.remove(temp_upload_path)
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
