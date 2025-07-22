import argparse
import logging
import os

from src.config import Config
from src.s2stSystem import S2STSystem

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