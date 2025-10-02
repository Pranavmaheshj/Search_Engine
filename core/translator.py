from deep_translator import GoogleTranslator
from deep_translator.exceptions import LanguageNotSupportedException
import logging

logger = logging.getLogger(__name__)

class CachingTranslator:
    def __init__(self):
        self.cache = {}
        # This line adds the dictionary of supported languages to the class instance
        self.supported_languages = GoogleTranslator().get_supported_languages(as_dict=True)

    def translate(self, text: str, target_lang: str) -> str:
        """Translates text to a given language name or code (e.g., 'spanish', 'es')."""
        try:
            # The library can handle both full names and codes
            translated_text = GoogleTranslator(source='auto', target=target_lang).translate(text)
            return translated_text
        except LanguageNotSupportedException:
            return f"Language '{target_lang}' is not supported by the translation service."
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return "Sorry, the translation service is currently unavailable."