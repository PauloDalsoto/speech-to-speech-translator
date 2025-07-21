from typing import Optional
import logging
import speech_recognition as sr
import numpy as np
import tempfile
import os
import librosa

from.config import Config

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

class STTProcessor:
    """Speech-to-Text processing with multiple backend support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.backend = config.stt_backend
        self.recognizer = sr.Recognizer()
        
        # Initialize backend-specific components
        if self.backend == "whisper_local" and WHISPER_AVAILABLE:
            self.whisper_model = whisper.load_model("base")
            logging.info("Whisper local model loaded")
        elif self.backend == "openai" and OPENAI_AVAILABLE:
            openai.api_key = config.openai_api_key
            logging.info("OpenAI STT configured")
        elif self.backend == "groq" and GROQ_AVAILABLE:
            self.groq_client = Groq(api_key=config.groq_api_key)
            logging.info("Groq STT configured")
        else:
            logging.info("Using Google Speech Recognition")

    def process_audio(self, audio_data: sr.AudioData) -> Optional[str]:
        """Convert audio to text using configured backend."""
        try:
            if self.backend == "google":
                return self._google_stt(audio_data)
            elif self.backend == "whisper_local" and WHISPER_AVAILABLE:
                return self._whisper_local_stt(audio_data)
            elif self.backend == "openai" and OPENAI_AVAILABLE:
                return self._openai_stt(audio_data)
            elif self.backend == "groq" and GROQ_AVAILABLE:
                return self._groq_stt(audio_data)
            else:
                # Fallback to Google
                return self._google_stt(audio_data)
                
        except Exception as e:
            logging.error(f"STT processing error: {e}")
            return None

    def _google_stt(self, audio_data: sr.AudioData) -> Optional[str]:
        """Google Speech Recognition backend."""
        try:
            text = self.recognizer.recognize_google(
                audio_data, 
                language=self.config.source_language
            )
            logging.debug(f"Google STT result: {text}")
            return text
        except sr.UnknownValueError:
            logging.debug("Google STT could not understand audio")
            return None
        except sr.RequestError as e:
            logging.error(f"Google STT request error: {e}")
            return None

    def _whisper_local_stt(self, audio_data: sr.AudioData) -> Optional[str]:
        """Whisper local model backend."""
        try:
            # Convert audio data to numpy array
            audio_np = np.frombuffer(audio_data.get_raw_data(), dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0
            
            # Resample if necessary
            if audio_data.sample_rate != 16000:
                audio_float = librosa.resample(
                    audio_float, 
                    orig_sr=audio_data.sample_rate, 
                    target_sr=16000
                )
            
            result = self.whisper_model.transcribe(audio_float)
            text = result["text"].strip()
            logging.debug(f"Whisper STT result: {text}")
            return text if text else None
            
        except Exception as e:
            logging.error(f"Whisper STT error: {e}")
            return None

    def _openai_stt(self, audio_data: sr.AudioData) -> Optional[str]:
        """OpenAI Whisper API backend."""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data.get_wav_data())
                temp_filename = temp_file.name
            
            # Send to OpenAI
            with open(temp_filename, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                text = transcript["text"].strip()
                logging.debug(f"OpenAI STT result: {text}")
                
            # Clean up
            os.unlink(temp_filename)
            return text if text else None
            
        except Exception as e:
            logging.error(f"OpenAI STT error: {e}")
            return None

    def _groq_stt(self, audio_data: sr.AudioData) -> Optional[str]:
        """Groq Whisper API backend."""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data.get_wav_data())
                temp_filename = temp_file.name
            
            # Send to Groq
            with open(temp_filename, "rb") as audio_file:
                transcription = self.groq_client.audio.transcriptions.create(
                    file=audio_file,
                    model="whisper-large-v3",
                    language="en"
                )
                text = transcription.text.strip()
                logging.debug(f"Groq STT result: {text}")
                
            # Clean up
            os.unlink(temp_filename)
            return text if text else None
            
        except Exception as e:
            logging.error(f"Groq STT error: {e}")
            return None