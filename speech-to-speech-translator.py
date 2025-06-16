# Speech-to-Speech Translation System
# A modular, low-latency English to Portuguese translation system

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

# Core dependencies
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from googletrans import Translator
import pyttsx3
import pygame
from io import BytesIO
import tempfile
import os

# Optional dependencies for different backends
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


@dataclass
class Config:
    """Configuration class for the S2ST system."""
    # Audio settings
    sample_rate: int = 16000
    chunk_duration: float = 2.0  # seconds
    silence_threshold: float = 0.01
    
    # Processing settings
    stt_backend: str = "google"  # google, whisper_local, openai, groq
    translation_backend: str = "groq"  # google, openai, groq
    tts_backend: str = "pyttsx3"  # pyttsx3, openai
    
    # API keys (set via environment or config)
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    
    # Performance settings
    max_queue_size: int = 10
    processing_timeout: float = 10.0
    
    # Languages
    source_language: str = "en"
    target_language: str = "pt"
    
    # Logging
    log_level: str = "INFO"

    @classmethod
    def load_from_file(cls, config_path: str) -> 'Config':
        """Load configuration from JSON file."""
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()

    def save_to_file(self, config_path: str):
        """Save configuration to JSON file."""
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class AudioCapture:
    """Handles continuous audio capture from microphone."""
    
    def __init__(self, config: Config):
        self.config = config
        self.is_recording = False
        self.audio_queue = queue.Queue(maxsize=config.max_queue_size)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=config.sample_rate)
        
        # Adjust for ambient noise
        logging.info("Adjusting for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        logging.info("Audio capture initialized")

    def start_capture(self):
        """Start continuous audio capture."""
        self.is_recording = True
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()
        logging.info("Audio capture started")

    def stop_capture(self):
        """Stop audio capture."""
        self.is_recording = False
        logging.info("Audio capture stopped")

    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        while self.is_recording:
            try:
                with self.microphone as source:
                    # Listen for audio with timeout
                    audio = self.recognizer.listen(
                        source, 
                        timeout=1, 
                        phrase_time_limit=self.config.chunk_duration
                    )
                    
                    if not self.audio_queue.full():
                        self.audio_queue.put(audio)
                        logging.debug("Audio chunk captured")
                    else:
                        logging.warning("Audio queue full, dropping chunk")
                        
            except sr.WaitTimeoutError:
                # No speech detected, continue
                continue
            except Exception as e:
                logging.error(f"Audio capture error: {e}")
                time.sleep(0.1)

    def get_audio(self) -> Optional[sr.AudioData]:
        """Get next audio chunk from queue."""
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None


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
                import librosa
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


class TranslationProcessor:
    """Text translation with multiple backend support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.backend = config.translation_backend
        
        if self.backend == "google":
            self.translator = Translator()
            logging.info("Google Translator initialized")
        elif self.backend == "openai" and OPENAI_AVAILABLE:
            openai.api_key = config.openai_api_key
            logging.info("OpenAI translation configured")
        elif self.backend == "groq" and GROQ_AVAILABLE:
            self.groq_client = Groq(api_key=config.groq_api_key)
            logging.info("Groq translation configured")

    def translate_text(self, text: str) -> Optional[str]:
        """Translate text using configured backend."""
        try:
            if self.backend == "google":
                return self._google_translate(text)
            elif self.backend == "openai" and OPENAI_AVAILABLE:
                return self._openai_translate(text)
            elif self.backend == "groq" and GROQ_AVAILABLE:
                return self._groq_translate(text)
            else:
                # Fallback to Google
                return self._google_translate(text)
                
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return None

    def _google_translate(self, text: str) -> Optional[str]:
        """Google Translate backend."""
        try:
            result = self.translator.translate(
                text,
                src=self.config.source_language,
                dest=self.config.target_language
            )
            translated = result.text
            logging.debug(f"Google translation: {text} -> {translated}")
            return translated
        except Exception as e:
            logging.error(f"Google translation error: {e}")
            return None

    def _openai_translate(self, text: str) -> Optional[str]:
        """OpenAI GPT translation backend."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a translator. Translate the following English text to Portuguese. Return only the translation, no explanations."},
                    {"role": "user", "content": text}
                ],
                max_tokens=200,
                temperature=0.1
            )
            translated = response.choices[0].message.content.strip()
            logging.debug(f"OpenAI translation: {text} -> {translated}")
            return translated
        except Exception as e:
            logging.error(f"OpenAI translation error: {e}")
            return None

    def _groq_translate(self, text: str) -> Optional[str]:
        """Groq translation backend using Llama models."""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um tradutor especializado. Traduza o texto em inglês para português brasileiro de forma natural e fluente. Retorne apenas a tradução, sem explicações."
                    },
                    {
                        "role": "user",
                        "content": f"Traduza para português: {text}"
                    }
                ],
                model="meta-llama/llama-4-scout-17b-16e-instruct",  
                max_tokens=500,
                temperature=0.1,
                stream=False
            )
            
            translated = chat_completion.choices[0].message.content.strip()
            logging.debug(f"Groq translation: {text} -> {translated}")
            return translated
            
        except Exception as e:
            logging.error(f"Groq translation error: {e}")
            return None


class TTSProcessor:
    """Text-to-Speech processing with multiple backend support."""
    
    def __init__(self, config: Config):
        self.config = config
        self.backend = config.tts_backend
        
        if self.backend == "pyttsx3":
            self.engine = pyttsx3.init()
            self._configure_pyttsx3()
            logging.info("pyttsx3 TTS initialized")
        elif self.backend == "openai" and OPENAI_AVAILABLE:
            openai.api_key = config.openai_api_key
            logging.info("OpenAI TTS configured")
        
        # Initialize pygame for audio playback
        pygame.mixer.init()

    def _configure_pyttsx3(self):
        """Configure pyttsx3 engine for Portuguese."""
        voices = self.engine.getProperty('voices')
        
        # Try to find Portuguese voice
        for voice in voices:
            if 'pt' in voice.id.lower() or 'brazil' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                logging.info(f"Using Portuguese voice: {voice.name}")
                break
        
        # Set speech rate
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 0.9)

    def generate_speech(self, text: str) -> Optional[str]:
        """Generate speech audio from text."""
        try:
            if self.backend == "pyttsx3":
                return self._pyttsx3_generate(text)
            elif self.backend == "openai" and OPENAI_AVAILABLE:
                return self._openai_generate(text)
            else:
                # Fallback to pyttsx3
                return self._pyttsx3_generate(text)
                
        except Exception as e:
            logging.error(f"TTS generation error: {e}")
            return None

    def _pyttsx3_generate(self, text: str) -> Optional[str]:
        """Generate speech using pyttsx3."""
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_filename = temp_file.name
            temp_file.close()
            
            # Generate speech
            self.engine.save_to_file(text, temp_filename)
            self.engine.runAndWait()
            
            logging.debug(f"pyttsx3 generated speech for: {text}")
            return temp_filename
            
        except Exception as e:
            logging.error(f"pyttsx3 generation error: {e}")
            return None

    def _openai_generate(self, text: str) -> Optional[str]:
        """Generate speech using OpenAI TTS."""
        try:
            response = openai.Audio.create_speech(
                model="tts-1",
                voice="alloy",
                input=text
            )
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
            temp_filename = temp_file.name
            temp_file.write(response.content)
            temp_file.close()
            
            logging.debug(f"OpenAI TTS generated speech for: {text}")
            return temp_filename
            
        except Exception as e:
            logging.error(f"OpenAI TTS error: {e}")
            return None

    def play_audio(self, audio_file: str):
        """Play audio file."""
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            logging.debug(f"Played audio: {audio_file}")
            
            # Clean up temp file
            try:
                os.unlink(audio_file)
            except:
                pass
                
        except Exception as e:
            logging.error(f"Audio playback error: {e}")


class S2STSystem:
    """Main Speech-to-Speech Translation System."""
    
    def __init__(self, config: Config):
        self.config = config
        self.is_running = False
        
        # Initialize components
        self.audio_capture = AudioCapture(config)
        self.stt_processor = STTProcessor(config)
        self.translation_processor = TranslationProcessor(config)
        self.tts_processor = TTSProcessor(config)
        
        # Processing queues
        self.text_queue = queue.Queue(maxsize=config.max_queue_size)
        self.translation_queue = queue.Queue(maxsize=config.max_queue_size)
        
        logging.info("S2ST System initialized")

    def start(self):
        """Start the complete translation pipeline."""
        self.is_running = True
        
        # Start audio capture
        self.audio_capture.start_capture()
        
        # Start processing threads
        threading.Thread(target=self._stt_worker, daemon=True).start()
        threading.Thread(target=self._translation_worker, daemon=True).start()
        threading.Thread(target=self._tts_worker, daemon=True).start()
        
        logging.info("S2ST System started - speak in English!")
        
        try:
            # Main loop
            while self.is_running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logging.info("Stopping system...")
            self.stop()

    def stop(self):
        """Stop the translation system."""
        self.is_running = False
        self.audio_capture.stop_capture()
        logging.info("S2ST System stopped")

    def _stt_worker(self):
        """STT processing worker thread."""
        while self.is_running:
            try:
                audio_data = self.audio_capture.get_audio()
                if audio_data:
                    start_time = time.time()
                    text = self.stt_processor.process_audio(audio_data)
                    processing_time = time.time() - start_time
                    
                    if text and text.strip():
                        logging.info(f"STT ({processing_time:.2f}s): {text}")
                        
                        if not self.text_queue.full():
                            self.text_queue.put(text)
                        else:
                            logging.warning("Text queue full, dropping text")
                else:
                    time.sleep(0.01)
                    
            except Exception as e:
                logging.error(f"STT worker error: {e}")
                time.sleep(0.1)

    def _translation_worker(self):
        """Translation processing worker thread."""
        while self.is_running:
            try:
                try:
                    text = self.text_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                start_time = time.time()
                translated = self.translation_processor.translate_text(text)
                processing_time = time.time() - start_time
                
                if translated:
                    logging.info(f"Translation ({processing_time:.2f}s): {translated}")
                    
                    if not self.translation_queue.full():
                        self.translation_queue.put(translated)
                    else:
                        logging.warning("Translation queue full, dropping translation")
                
                self.text_queue.task_done()
                
            except Exception as e:
                logging.error(f"Translation worker error: {e}")
                time.sleep(0.1)

    def _tts_worker(self):
        """TTS processing worker thread."""
        while self.is_running:
            try:
                try:
                    translated_text = self.translation_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                start_time = time.time()
                audio_file = self.tts_processor.generate_speech(translated_text)
                
                if audio_file:
                    self.tts_processor.play_audio(audio_file)
                    processing_time = time.time() - start_time
                    logging.info(f"TTS + Playback ({processing_time:.2f}s): Complete")
                
                self.translation_queue.task_done()
                
            except Exception as e:
                logging.error(f"TTS worker error: {e}")
                time.sleep(0.1)


def setup_logging(level: str):
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Speech-to-Speech Translation System")
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--stt", choices=["google", "whisper_local", "openai", "groq"], help="STT backend override")
    parser.add_argument("--translation", choices=["google", "openai", "groq"], help="Translation backend override")
    parser.add_argument("--tts", choices=["pyttsx3", "openai"], help="TTS backend override")
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config.load_from_file(args.config)
    
    # Apply command line overrides
    if args.stt:
        config.stt_backend = args.stt
    if args.translation:
        config.translation_backend = args.translation
    if args.tts:
        config.tts_backend = args.tts
    
    # Setup logging
    setup_logging(config.log_level)
    
    # Check API keys if needed
    if config.openai_api_key is None and (
        config.stt_backend == "openai" or 
        config.translation_backend == "openai" or 
        config.tts_backend == "openai"
    ):
        config.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not config.openai_api_key:
            logging.error("OpenAI API key required but not found. Set OPENAI_API_KEY environment variable.")
            return
    
    if config.groq_api_key is None and (
        config.stt_backend == "groq" or 
        config.translation_backend == "groq"
    ):
        config.groq_api_key = os.getenv("GROQ_API_KEY")
        if not config.groq_api_key:
            logging.error("Groq API key required but not found. Set GROQ_API_KEY environment variable.")
            return
    
    # Save current configuration
    config.save_to_file(args.config)
    
    # Print configuration
    logging.info("Configuration:")
    logging.info(f"  STT Backend: {config.stt_backend}")
    logging.info(f"  Translation Backend: {config.translation_backend}")
    logging.info(f"  TTS Backend: {config.tts_backend}")
    logging.info(f"  Languages: {config.source_language} -> {config.target_language}")
    
    # Create and start system
    try:
        system = S2STSystem(config)
        system.start()
    except KeyboardInterrupt:
        logging.info("System interrupted by user")
    except Exception as e:
        logging.error(f"System error: {e}")


if __name__ == "__main__":
    main()