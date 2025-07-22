
import logging
import queue
import threading
import time

from src.config import Config
from src.audioCapture import AudioCapture
from src.sttProcessor import STTProcessor
from src.ttsProcessor import TTSProcessor
from src.translationProcessor import TranslationProcessor

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
