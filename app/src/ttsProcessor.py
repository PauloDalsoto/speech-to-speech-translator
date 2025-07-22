import logging
import pyttsx3
from typing import Optional
import pygame
import time
import tempfile
import os

from src.config import Config

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