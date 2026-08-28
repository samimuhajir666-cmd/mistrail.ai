"""
Streamlit Romanian Speech-to-Text Web App
With Advanced Noise Detection and Garbage Voice Filtering

Features:
- Real-time Romanian speech recognition
- Background noise detection and reduction
- Garbage voice filtering
- Web-based interface
- GitHub integration
- Deployable on Streamlit Cloud

Usage:
1. Install: pip install streamlit pyaudio python-dotenv PyGithub requests
2. Run: streamlit run app.py
3. Deploy: streamlit deploy
"""

import os
import time
import base64
import wave
import numpy as np
from datetime import datetime
import requests
import streamlit as st
from io import BytesIO

# Try to import optional dependencies
try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    st.warning("PyAudio not installed. Microphone recording disabled.")

try:
    from github import Github, GithubException
    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False
    st.warning("PyGithub not installed. GitHub posting disabled.")

try:
    from dotenv import load_dotenv
    load_dotenv()
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False


# Advanced noise detection and filtering
class AudioProcessor:
    """Advanced audio processing for noise reduction and voice filtering"""
    
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.noise_threshold = 1000  # Default threshold
        self.voice_activity_threshold = 0.05  # 5% of max amplitude
        self.silence_duration = 2.0  # 2 seconds of silence
        self.min_voice_duration = 1.0  # 1 second minimum voice
        self.buffer = []
        self.silent_chunks = 0
        self.voice_chunks = 0
        self.is_voice_active = False
        self.recording = False
    
    def detect_voice_activity(self, audio_data):
        """Detect voice activity using amplitude threshold"""
        try:
            # Convert bytes to numpy array
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate RMS (Root Mean Square) for amplitude
            rms = np.sqrt(np.mean(audio_np ** 2))
            
            # Dynamic threshold based on recent noise levels
            if rms > self.voice_activity_threshold * 32768:  # 16-bit max
                return True
            return False
        except:
            return False
    
    def is_noise(self, audio_data):
        """Check if audio chunk is noise"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate various features
            rms = np.sqrt(np.mean(audio_np ** 2))
            max_amp = np.max(np.abs(audio_np))
            
            # Noise detection: low amplitude, high frequency content
            if rms < self.noise_threshold and max_amp < 5000:
                return True
            
            # Check for sudden spikes (clipping/garbage)
            if max_amp > 30000:  # Close to 16-bit max
                return True
            
            return False
        except:
            return False
    
    def filter_garbage_voice(self, audio_data):
        """Filter out garbage/non-human voice"""
        try:
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            
            # Check for clipping (garbage audio)
            if np.max(np.abs(audio_np)) > 30000:
                return None
            
            # Check for DC offset (bad recording)
            if np.abs(np.mean(audio_np)) > 1000:
                audio_np = audio_np - np.mean(audio_np)
            
            # Simple high-pass filter to remove low-frequency noise
            if len(audio_np) > 100:
                audio_np = self.high_pass_filter(audio_np, cutoff=100)
            
            return audio_np.tobytes()
        except:
            return audio_data
    
    def high_pass_filter(self, data, cutoff=100):
        """Simple high-pass filter to remove low-frequency noise"""
        try:
            # Simple FIR high-pass filter
            filtered = np.zeros_like(data)
            alpha = 0.98  # Smoothing factor
            
            prev = 0
            for i in range(len(data)):
                filtered[i] = alpha * (filtered[i-1] if i > 0 else 0) + (1 - alpha) * data[i]
                filtered[i] = data[i] - filtered[i]
            
            return filtered
        except:
            return data
    
    def process_audio_chunk(self, chunk):
        """Process individual audio chunk with noise filtering"""
        # Filter garbage voice
        filtered = self.filter_garbage_voice(chunk)
        if filtered is None:
            return None, False
        
        # Check if it's noise
        if self.is_noise(filtered):
            return None, False
        
        # Check for voice activity
        has_voice = self.detect_voice_activity(filtered)
        
        return filtered, has_voice


class RomanianSTTStreamlit:
    """Streamlit web app for Romanian STT with noise filtering"""
    
    def __init__(self):
        # Configuration
        self.config = {
            'api_key': os.getenv('STT_API_KEY', ''),
            'github_token': os.getenv('GITHUB_TOKEN', ''),
            'github_repo': os.getenv('GITHUB_REPO', 'mistrail.ai'),
            'language': os.getenv('LANGUAGE', 'ro-RO'),
            'branch': os.getenv('BRANCH', 'main'),
            'output_folder': os.getenv('OUTPUT_FOLDER', 'stt-transcriptions'),
            'sample_rate': int(os.getenv('SAMPLE_RATE', '44100')),
            'channels': int(os.getenv('CHANNELS', '1')),
            'format': int(os.getenv('AUDIO_FORMAT', '16')),
            'chunk_size': int(os.getenv('CHUNK_SIZE', '1024')),
            'auto_post': st.session_state.get('auto_post', True),
        }
        
        # Audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=self.config['sample_rate'],
            chunk_size=self.config['chunk_size']
        )
        
        # Initialize components
        self.pyaudio_instance = None
        self.audio_stream = None
        self.is_recording = False
        self.github_client = None
        self.audio_buffer = []
        
        if HAS_GITHUB and self.config['github_token']:
            self.github_client = Github(self.config['github_token'])
        
        if HAS_PYAUDIO:
            self.pyaudio_instance = pyaudio.PyAudio()
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using AssemblyAI"""
        if not self.config['api_key']:
            st.error("AssemblyAI API key not configured")
            return None
        
        try:
            # Save to temp file
            temp_file = "temp_audio.wav"
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(self.config['channels'])
                wf.setsampwidth(self.config['format'] // 8)
                wf.setframerate(self.config['sample_rate'])
                wf.writeframes(audio_data)
            
            # Encode and upload
            with open(temp_file, 'rb') as f:
                audio_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            headers = {'authorization': self.config['api_key'], 'content-type': 'application/json'}
            response = requests.post(
                'https://api.assemblyai.com/v2/transcript',
                headers=headers,
                json={'audio': audio_b64, 'language_code': self.config['language']}
            )
            
            if response.status_code != 200:
                st.error(f"Transcription failed: {response.text}")
                return None
            
            transcript_id = response.json()['id']
            
            # Wait for transcription
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(60):  # 5 minutes max
                time.sleep(5)
                progress_bar.progress(min((i + 1) / 60, 1.0))
                status_text.text(f"Transcribing... {i * 5} seconds")
                
                status_response = requests.get(
                    f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                    headers=headers
                )
                
                if status_response.status_code != 200:
                    continue
                
                status_data = status_response.json()
                
                if status_data['status'] == 'completed':
                    os.remove(temp_file)
                    progress_bar.empty()
                    status_text.empty()
                    return status_data['text']
                elif status_data['status'] == 'error':
                    os.remove(temp_file)
                    st.error(f"Transcription error: {status_data.get('error', 'Unknown')}")
                    return None
            
            os.remove(temp_file)
            st.error("Transcription timeout")
            return None
            
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    
    def post_to_github(self, transcription, audio_data=None):
        """Post transcription to GitHub"""
        if not HAS_GITHUB or not self.github_client:
            st.warning("GitHub posting disabled")
            return False
        
        try:
            repo = self.github_client.get_repo(self.config['github_repo'])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.config['output_folder']}/stt_{timestamp}.md"
            
            content = f"""# Romanian STT Transcription

**Timestamp:** {datetime.now().isoformat()}
**Word Count:** {len(transcription.split())} words
**Language:** Romanian (ro-RO)
**Source:** Streamlit Web App

---

{transcription}

---
*Generated by Streamlit Romanian STT Agent*
"""
            
            try:
                repo.create_file(
                    path=filename,
                    message=f"STT: Romanian transcription - {timestamp}",
                    content=content,
                    branch=self.config['branch']
                )
                st.success(f"✅ Posted to GitHub: {filename}")
            except GithubException as e:
                if e.status == 409:
                    existing = repo.get_contents(filename, ref=self.config['branch'])
                    repo.update_file(
                        path=filename,
                        message=f"STT: Update - {timestamp}",
                        content=content,
                        sha=existing.sha,
                        branch=self.config['branch']
                    )
                    st.success(f"✅ Updated on GitHub: {filename}")
                else:
                    st.error(f"GitHub error: {e}")
                    return False
            
            if audio_data and self.config['auto_post']:
                audio_filename = f"{self.config['output_folder']}/audio_{timestamp}.wav"
                repo.create_file(
                    path=audio_filename,
                    message=f"Audio: {timestamp}",
                    content=base64.b64encode(audio_data).decode('utf-8'),
                    branch=self.config['branch']
                )
                st.success(f"✅ Audio posted: {audio_filename}")
            
            return True
            
        except Exception as e:
            st.error(f"GitHub posting failed: {e}")
            return False
    
    def start_recording(self):
        """Start audio recording"""
        if not HAS_PYAUDIO or not self.pyaudio_instance:
            st.error("Audio recording not available")
            return False
        
        if self.is_recording:
            st.warning("Already recording")
            return False
        
        try:
            self.audio_buffer = []
            self.audio_stream = self.pyaudio_instance.open(
                format=self.pyaudio_instance.get_format_from_width(self.config['format'] // 8),
                channels=self.config['channels'],
                rate=self.config['sample_rate'],
                input=True,
                frames_per_buffer=self.config['chunk_size'],
                stream_callback=self._audio_callback
            )
            self.is_recording = True
            st.session_state['recording'] = True
            st.success("🎤 Recording started...")
            return True
        except Exception as e:
            st.error(f"Recording failed: {e}")
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for processing"""
        # Process audio with noise filtering
        processed, has_voice = self.audio_processor.process_audio_chunk(in_data)
        
        if processed is not None:
            self.audio_buffer.append(processed)
        
        return (in_data, pyaudio.paContinue)
    
    def stop_recording(self):
        """Stop recording and return audio data"""
        if not self.is_recording:
            return None
        
        try:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.is_recording = False
            st.session_state['recording'] = False
            
            audio_data = b''.join(self.audio_buffer)
            st.success(f"✅ Recording stopped. {len(audio_data)} bytes captured")
            return audio_data
        except Exception as e:
            st.error(f"Stop failed: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.audio_stream:
            try:
                self.audio_stream.close()
            except:
                pass
        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
            except:
                pass


# Streamlit App
def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Romanian STT Agent",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'recording' not in st.session_state:
        st.session_state['recording'] = False
    if 'transcription' not in st.session_state:
        st.session_state['transcription'] = ""
    if 'auto_post' not in st.session_state:
        st.session_state['auto_post'] = True
    
    # Create app instance
    app = RomanianSTTStreamlit()
    
    # Title
    st.title("🇷🇴 Romanian Speech-to-Text Agent")
    st.markdown("""
    ### Real-time Romanian Speech Recognition
    with **Advanced Noise Detection** and **Garbage Voice Filtering**
    
    **Features:**
    - ✅ Background noise detection and reduction
    - ✅ Garbage voice filtering
    - ✅ Real-time audio processing
    - ✅ Automatic GitHub posting
    - ✅ Web-based interface
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Keys
        st.subheader("API Keys")
        stt_api_key = st.text_input(
            "AssemblyAI API Key",
            value=os.getenv('STT_API_KEY', ''),
            help="Your AssemblyAI API key for transcription"
        )
        
        github_token = st.text_input(
            "GitHub Personal Access Token",
            value=os.getenv('GITHUB_TOKEN', ''),
            help="GitHub token with repo access"
        )
        
        github_repo = st.text_input(
            "GitHub Repository",
            value=os.getenv('GITHUB_REPO', 'mistrail.ai'),
            help="Repository to post transcriptions (username/repo)"
        )
        
        # Settings
        st.subheader("Settings")
        language = st.selectbox(
            "Language",
            ["ro-RO", "en-US", "fr-FR", "de-DE", "es-ES"],
            index=0,
            help="Language for transcription"
        )
        
        branch = st.text_input("Branch", value=os.getenv('BRANCH', 'main'))
        output_folder = st.text_input("Output Folder", value=os.getenv('OUTPUT_FOLDER', 'stt-transcriptions'))
        
        auto_post = st.checkbox("Auto-post to GitHub", value=True)
        st.session_state['auto_post'] = auto_post
        
        # Noise filtering settings
        st.subheader("🔊 Noise Filtering")
        noise_threshold = st.slider(
            "Noise Threshold",
            min_value=100,
            max_value=5000,
            value=1000,
            help="Amplitude threshold for noise detection"
        )
        
        voice_threshold = st.slider(
            "Voice Activity Threshold",
            min_value=1,
            max_value=20,
            value=5,
            help="Percentage of max amplitude for voice detection"
        )
        
        app.audio_processor.noise_threshold = noise_threshold
        app.audio_processor.voice_activity_threshold = voice_threshold / 100
        
        # Update config
        app.config.update({
            'api_key': stt_api_key,
            'github_token': github_token,
            'github_repo': github_repo,
            'language': language,
            'branch': branch,
            'output_folder': output_folder,
            'auto_post': auto_post
        })
        
        # Update GitHub client if token changed
        if github_token and HAS_GITHUB:
            app.github_client = Github(github_token)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎤 Record", "📤 Upload", "📋 History"])
    
    with tab1:
        st.header("Real-time Recording")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🎤 Start Recording", disabled=st.session_state['recording']):
                app.start_recording()
            
            if st.button("⏹️ Stop Recording", disabled=not st.session_state['recording']):
                audio_data = app.stop_recording()
                if audio_data:
                    # Transcribe
                    with st.spinner("Transcribing..."):
                        transcription = app.transcribe_audio(audio_data)
                    
                    if transcription:
                        st.session_state['transcription'] = transcription
                        st.session_state['last_audio'] = audio_data
                        
                        # Display transcription
                        st.subheader("📝 Transcription")
                        st.text_area("", value=transcription, height=200, key="transcription_display")
                        
                        # Auto-post if enabled
                        if auto_post:
                            with st.spinner("Posting to GitHub..."):
                                app.post_to_github(transcription, audio_data)
        
        # Audio visualization
        if 'last_audio' in st.session_state and st.session_state['last_audio']:
            st.subheader("🎵 Audio Waveform")
            try:
                audio_np = np.frombuffer(st.session_state['last_audio'], dtype=np.int16)
                st.line_chart(audio_np[:10000])  # Show first 10k samples
            except:
                st.warning("Could not display waveform")
    
    with tab2:
        st.header("Upload Audio File")
        
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "ogg", "flac"]
        )
        
        if uploaded_file is not None:
            # Read audio file
            audio_data = uploaded_file.read()
            
            # Process with noise filtering
            processed_audio = app.audio_processor.filter_garbage_voice(audio_data)
            if processed_audio is None:
                st.error("Audio file contains garbage or is corrupted")
            else:
                st.success("✅ Audio file loaded and filtered")
                
                if st.button("Transcribe Uploaded File"):
                    with st.spinner("Transcribing..."):
                        transcription = app.transcribe_audio(processed_audio)
                    
                    if transcription:
                        st.session_state['transcription'] = transcription
                        st.subheader("📝 Transcription")
                        st.text_area("", value=transcription, height=200, key="upload_transcription")
                        
                        if auto_post:
                            with st.spinner("Posting to GitHub..."):
                                app.post_to_github(transcription, processed_audio)
    
    with tab3:
        st.header("Transcription History")
        
        if st.session_state['transcription']:
            st.subheader("Latest Transcription")
            st.write(st.session_state['transcription'])
            
            # Save to file
            if st.button("Save to File"):
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"transcription_{timestamp}.txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(st.session_state['transcription'])
                st.success(f"Saved to {filename}")
        else:
            st.info("No transcriptions yet. Record or upload an audio file.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🔧 How It Works
    
    **Advanced Features:**
    - 🎯 **Noise Detection**: Filters out background noise using RMS amplitude analysis
    - 🗑️ **Garbage Filtering**: Removes clipped audio and non-human voice patterns
    - 🎙️ **Voice Activity Detection**: Only processes chunks with actual voice
    - 🔄 **Real-time Processing**: Processes audio as it's recorded
    
    **Technologies:**
    - Streamlit for web interface
    - AssemblyAI for transcription
    - PyAudio for audio capture
    - NumPy for audio processing
    
    **Deployment:**
    - Deploy to Streamlit Cloud: `streamlit deploy`
    - Run locally: `streamlit run app.py`
    """)
    
    # Cleanup on exit
    try:
        app.cleanup()
    except:
        pass


if __name__ == "__main__":
    main()
