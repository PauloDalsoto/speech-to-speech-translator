from typing import Optional
import logging
from googletrans import Translator

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