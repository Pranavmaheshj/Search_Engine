from gtts import gTTS
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

class TextToSpeechService:
    def __init__(self, output_dir: str = "data/audio_output"):
        self.output_path = Path(output_dir)
        self.output_path.mkdir(exist_ok=True, parents=True)

    def speak(self, text: str, lang: str = 'en', query: str = "output"):
        """Generates an MP3 file and returns the file path."""
        try:
            safe_filename = re.sub(r'[^\w\-_\. ]', '_', query)[:50] + ".mp3"
            filepath = self.output_path / safe_filename
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(str(filepath))
            logger.info(f"Audio saved to {filepath}")
            # This is the fix: return only the path
            return filepath
        except Exception as e:
            logger.error(f"gTTS failed: {e}")
            return "Sorry, could not generate audio at this time."