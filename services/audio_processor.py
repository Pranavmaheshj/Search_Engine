from pathlib import Path
import yt_dlp
from groq import Groq
import os
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, summarizer):
        self.summarizer = summarizer
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.temp_dir = Path("data/temp_audio")
        self.summary_dir = Path("data/video_summaries")
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        self.summary_dir.mkdir(exist_ok=True, parents=True)

    def search_for_video(self, query: str) -> dict | None:
        """Quickly searches for the top video using yt-dlp and returns its metadata."""
        print(f"Searching YouTube for: '{query}'...")
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'default_search': 'ytsearch1',
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                result = ydl.extract_info(query, download=False)
                if result and result.get('entries'):
                    video_info = result['entries'][0]
                    return {
                        "id": video_info.get('id'),
                        "title": video_info.get('title'),
                        "url": video_info.get('url'),
                        "thumbnail": video_info.get('thumbnail'),
                    }
        except Exception as e:
            logger.error(f"YouTube search with yt-dlp failed: {e}")
        return None

    async def get_summary_from_youtube_audio(self, video_id: str, video_title: str):
        """Downloads, transcribes via API, and summarizes audio for a given video ID."""
        audio_path = None
        try:
            print(f"Starting background audio processing for '{video_title}'...")
            ydl_opts = {
                'format': 'bestaudio[ext=mp3]/bestaudio',
                'outtmpl': str(self.temp_dir / f"{video_id}.%(ext)s"),
                'quiet': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_id, download=True)
                audio_path = Path(ydl.prepare_filename(info))
            
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(audio_path.name, file.read()), model="whisper-large-v3"
                )
            
            if transcription.text:
                summary = self.summarizer.generate_summary(
                    context=transcription.text,
                    query=f"the YouTube video titled '{video_title}'",
                    age_group="adult"
                )
                safe_title = "".join(c for c in video_title if c.isalnum() or c in " _-").rstrip()[:100]
                summary_path = self.summary_dir / f"{safe_title}_summary.txt"
                summary_path.write_text(f"Summary of '{video_title}':\n\n{summary}", encoding='utf-8')
                logger.info(f"YouTube audio summary saved to {summary_path}")
        except Exception as e:
            logger.error(f"Could not process YouTube audio for '{video_title}': {e}")
        finally:
            if audio_path and audio_path.exists():
                audio_path.unlink()