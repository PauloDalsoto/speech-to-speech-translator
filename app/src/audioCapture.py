from typing import Optional
import queue
import threading
import logging
import time
import speech_recognition as sr

from src.config import Config

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