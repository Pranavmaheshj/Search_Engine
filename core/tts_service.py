from gtts import gTTS
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TextToSpeechService:
    def __init__(self, output_dir: str = "data/audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def speak(self, text: str, lang: str = 'en', query: str = "summary") -> Path | str:
        """Generates an MP3 file from text using gTTS and returns its path."""
        try:
            # Create a safe filename from the query
            safe_filename = "".join(c for c in query if c.isalnum() or c in " _-").rstrip()[:50]
            output_path = self.output_dir / f"{safe_filename}.mp3"
            
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(str(output_path))
            
            logger.info(f"Audio saved successfully to: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"gTTS failed: {e}")
            return f"Could not generate audio. Error: {e}"