import os
import time
import base64
import wave
import argparse
from datetime import datetime
import requests

try:
    import pyaudio
    HAS_PYAUDIO = True
except ImportError:
    HAS_PYAUDIO = False
    print("⚠️ PyAudio not installed - run: pip install PyAudio")

try:
    from github import Github, GithubException
    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False
    print("⚠️ PyGithub not installed - run: pip install PyGithub")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed - run: pip install python-dotenv")


class RomanianSTTAgent:
    """
    Romanian Speech-to-Text Agent with GitHub Integration
    Uses AssemblyAI for transcription
    Posts results directly to mistrail.ai repository
    """
    
    def __init__(self):
        # Load configuration from environment
        self.config = {
            'github_repo': os.getenv('GITHUB_REPO', 'mistrail.ai'),
            'github_token': os.getenv('GITHUB_TOKEN', ''),
            'api_key': os.getenv('STT_API_KEY', ''),
            'language': os.getenv('LANGUAGE', 'ro-RO'),
            'branch': os.getenv('BRANCH', 'main'),
            'output_folder': os.getenv('OUTPUT_FOLDER', 'stt-transcriptions'),
            'sample_rate': int(os.getenv('SAMPLE_RATE', '44100')),
            'channels': int(os.getenv('CHANNELS', '1')),
            'format': int(os.getenv('AUDIO_FORMAT', '16')),
            'chunk_size': int(os.getenv('CHUNK_SIZE', '1024')),
            'auto_push': os.getenv('AUTO_PUSH', 'true').lower() == 'true',
            'timeout': int(os.getenv('TRANSCRIPTION_TIMEOUT', '300'))
        }
        
        # Initialize components
        self.github_client = None
        self.pyaudio_instance = None
        self.audio_stream = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize GitHub and audio components"""
        # Initialize GitHub client
        if HAS_GITHUB and self.config['github_token']:
            try:
                self.github_client = Github(self.config['github_token'])
                print("✅ GitHub client initialized")
            except Exception as e:
                print(f"❌ GitHub init failed: {e}")
        else:
            print("⚠️ GitHub token not found or PyGithub not installed")
        
        # Initialize audio
        if HAS_PYAUDIO:
            try:
                self.pyaudio_instance = pyaudio.PyAudio()
                print("✅ Audio system initialized")
            except Exception as e:
                print(f"❌ Audio init failed: {e}")
        else:
            print("⚠️ PyAudio not installed")
    
    def record_audio(self, duration=30):
        """Record audio from microphone"""
        if not HAS_PYAUDIO or not self.pyaudio_instance:
            print("❌ Cannot record - audio not available")
            return None
        
        try:
            print(f"🎤 Recording for {duration} seconds...")
            
            frames = []
            stream = self.pyaudio_instance.open(
                format=self.pyaudio_instance.get_format_from_width(self.config['format'] // 8),
                channels=self.config['channels'],
                rate=self.config['sample_rate'],
                input=True,
                frames_per_buffer=self.config['chunk_size']
            )
            
            # Calculate number of chunks needed
            chunks = int(self.config['sample_rate'] / self.config['chunk_size'] * duration)
            
            for _ in range(chunks):
                data = stream.read(self.config['chunk_size'], exception_on_overflow=False)
                frames.append(data)
            
            stream.stop_stream()
            stream.close()
            
            audio_data = b''.join(frames)
            print(f"✅ Recorded {len(audio_data)} bytes of audio")
            return audio_data
            
        except Exception as e:
            print(f"❌ Recording failed: {e}")
            return None
    
    def save_temp_audio(self, audio_data, filename="temp_audio.wav"):
        """Save audio data to temporary WAV file"""
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(self.config['channels'])
                wf.setsampwidth(self.config['format'] // 8)
                wf.setframerate(self.config['sample_rate'])
                wf.writeframes(audio_data)
            return filename
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
            return None
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using AssemblyAI"""
        if not self.config['api_key']:
            print("❌ AssemblyAI API key not provided")
            return None
        
        # Save to temp file
        temp_file = self.save_temp_audio(audio_data)
        if not temp_file:
            return None
        
        try:
            print("🔊 Uploading to AssemblyAI...")
            
            # Read and encode audio
            with open(temp_file, 'rb') as f:
                audio_content = base64.b64encode(f.read()).decode('utf-8')
            
            # Upload request
            headers = {
                'authorization': self.config['api_key'],
                'content-type': 'application/json'
            }
            
            payload = {
                'audio': audio_content,
                'language_code': self.config['language'],
                'auto_chapters': False
            }
            
            response = requests.post(
                'https://api.assemblyai.com/v2/transcript',
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code != 200:
                print(f"❌ API upload failed: {response.status_code} - {response.text}")
                return None
            
            transcript_id = response.json()['id']
            print(f"📝 Transcript ID: {transcript_id}")
            
            # Wait for completion
            print("⏳ Waiting for transcription...")
            for i in range(self.config['timeout'] // 5):
                time.sleep(5)
                
                # Check status
                status_response = requests.get(
                    f'https://api.assemblyai.com/v2/transcript/{transcript_id}',
                    headers=headers,
                    timeout=30
                )
                
                if status_response.status_code != 200:
                    print(f"❌ Status check failed: {status_response.text}")
                    continue
                
                status_data = status_response.json()
                
                if status_data['status'] == 'completed':
                    transcription = status_data['text']
                    print("✅ Transcription completed!")
                    os.remove(temp_file)
                    return transcription
                elif status_data['status'] == 'error':
                    print(f"❌ Transcription error: {status_data.get('error', 'Unknown')}")
                    os.remove(temp_file)
                    return None
                
                # Progress update
                if (i + 1) % 6 == 0:  # Every 30 seconds
                    print(f"⏳ Still processing... ({i * 5} seconds)")
            
            print("❌ Transcription timeout")
            os.remove(temp_file)
            return None
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            try:
                os.remove(temp_file)
            except:
                pass
            return None
    
    def post_to_github(self, transcription, audio_data=None):
        """Post transcription to GitHub repository"""
        if not HAS_GITHUB or not self.github_client:
            print("⚠️ GitHub client not available")
            return False
        
        try:
            repo = self.github_client.get_repo(self.config['github_repo'])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create markdown filename
            filename = f"{self.config['output_folder']}/stt_{timestamp}.md"
            
            # Format content with metadata
            word_count = len(transcription.split())
            content = f"""# Romanian STT Transcription

**Timestamp:** {datetime.now().isoformat()}
**Word Count:** {word_count} words
**Language:** Romanian (ro-RO)
**Generated by:** Romanian STT GitHub Agent

---

## Transcription

{transcription}

---

*Automatically generated and posted to GitHub*
"""
            
            # Create file in repository
            try:
                repo.create_file(
                    path=filename,
                    message=f"STT: Romanian transcription - {timestamp}",
                    content=content,
                    branch=self.config['branch']
                )
                print(f"✅ Posted transcription to GitHub: {filename}")
            except GithubException as e:
                if e.status == 409:  # File exists
                    # Update existing file
                    existing = repo.get_contents(filename, ref=self.config['branch'])
                    repo.update_file(
                        path=filename,
                        message=f"STT: Update transcription - {timestamp}",
                        content=content,
                        sha=existing.sha,
                        branch=self.config['branch']
                    )
                    print(f"✅ Updated transcription on GitHub: {filename}")
                else:
                    raise
            
            # Optionally post audio file
            if audio_data and self.config['auto_push']:
                audio_filename = f"{self.config['output_folder']}/audio_{timestamp}.wav"
                audio_content = base64.b64encode(audio_data).decode('utf-8')
                
                repo.create_file(
                    path=audio_filename,
                    message=f"STT: Audio file - {timestamp}",
                    content=audio_content,
                    branch=self.config['branch']
                )
                print(f"✅ Posted audio file to GitHub: {audio_filename}")
            
            return True
            
        except Exception as e:
            print(f"❌ GitHub posting failed: {e}")
            return False
    
    def run(self, duration=30):
        """Main run method - record, transcribe, and post"""
        print("🚀 Starting Romanian STT Agent")
        print("=" * 40)
        
        # Record audio
        audio_data = self.record_audio(duration)
        if not audio_data:
            print("❌ No audio recorded")
            return False
        
        # Transcribe
        transcription = self.transcribe_audio(audio_data)
        if not transcription:
            print("❌ No transcription obtained")
            return False
        
        print(f"\n📝 Transcription Result:")
        print("-" * 40)
        print(transcription)
        print("-" * 40)
        
        # Post to GitHub
        if self.config['auto_push']:
            print("\n📤 Posting to GitHub...")
            success = self.post_to_github(transcription, audio_data)
            if success:
                print("🎉 All done!")
            else:
                print("⚠️ Posting failed, but transcription is ready")
        else:
            print("\n⚠️ Auto-post disabled. Transcription ready above.")
        
        return True
    
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


# Main CLI interface
def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Romanian Speech-to-Text GitHub Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent.py                    # Record 30 seconds
  python agent.py --duration 60     # Record 60 seconds
  python agent.py --test            # Test configuration
  python agent.py --no-post         # Don't post to GitHub
        """
    )
    
    parser.add_argument('--duration', type=int, default=30,
                       help='Recording duration in seconds (default: 30)')
    parser.add_argument('--test', action='store_true',
                       help='Test configuration and exit')
    parser.add_argument('--no-post', action='store_true',
                       help='Disable auto-posting to GitHub')
    parser.add_argument('--env', action='store_true',
                       help='Show required environment variables')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = RomanianSTTAgent()
    
    if args.env:
        print("Required Environment Variables:")
        print("=" * 40)
        print("GITHUB_TOKEN=your_github_personal_access_token")
        print("STT_API_KEY=your_assemblyai_api_key")
        print("\nOptional:")
        print("GITHUB_REPO=mistrail.ai (default)")
        print("LANGUAGE=ro-RO (default)")
        print("BRANCH=main (default)")
        print("OUTPUT_FOLDER=stt-transcriptions (default)")
        print("AUTO_PUSH=true (default)")
        return
    
    if args.test:
        print("🧪 Configuration Test")
        print("=" * 40)
        print(f"GitHub Repository: {agent.config['github_repo']}")
        print(f"GitHub Token: {'***' + agent.config['github_token'][-4:] if agent.config['github_token'] else 'NOT SET'}")
        print(f"STT API Key: {'***' + agent.config['api_key'][-4:] if agent.config['api_key'] else 'NOT SET'}")
        print(f"Language: {agent.config['language']}")
        print(f"Branch: {agent.config['branch']}")
        print(f"Output Folder: {agent.config['output_folder']}")
        print(f"Auto Push: {agent.config['auto_push']}")
        
        # Test connections
        if HAS_GITHUB and agent.config['github_token']:
            try:
                repo = agent.github_client.get_repo(agent.config['github_repo'])
                print(f"✅ GitHub connection: OK")
            except:
                print(f"❌ GitHub connection: FAILED")
        else:
            print(f"⚠️ GitHub connection: NOT CONFIGURED")
        
        if agent.config['api_key']:
            print(f"✅ AssemblyAI API Key: CONFIGURED")
        else:
            print(f"❌ AssemblyAI API Key: NOT CONFIGURED")
        
        if HAS_PYAUDIO:
            print(f"✅ PyAudio: INSTALLED")
        else:
            print(f"❌ PyAudio: NOT INSTALLED")
        
        return
    
    # Override auto_push if --no-post
    if args.no_post:
        agent.config['auto_push'] = False
    
    # Run the agent
    try:
        agent.run(args.duration)
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    finally:
        agent.cleanup()


if __name__ == "__main__":
    main()
