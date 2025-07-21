from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import json

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
    
    # API keys
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