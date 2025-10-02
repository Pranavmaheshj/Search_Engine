import os
import asyncio
from pathlib import Path
import yt_dlp
from groq import Groq
from core.summarizer import GeminiSummarizer

class VideoProcessor:
    def __init__(self, summarizer: GeminiSummarizer):
        self.summarizer = summarizer
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.temp_dir = Path("data/temp_video_processing")
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        # ✅ Hardcode ffmpeg path here
        self.ffmpeg_path = r"C:\Users\prana\Downloads\ffmpeg-2025-09-28-git-0fdb5829e3-full_build\ffmpeg-2025-09-28-git-0fdb5829e3-full_build\bin\ffmpeg.exe"   # change to your system path

    async def summarize_video(self, video_path_or_url: str, age_group: str) -> str:
        audio_path = None
        try:
            print(f"Processing video source: {video_path_or_url}")

            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.temp_dir / '%(id)s.%(ext)s'),
                'quiet': True,
                'postprocessors': [
                    {
                        'key': 'FFmpegExtractAudio',
                        'preferredcodec': 'mp3',
                        'preferredquality': '192',
                    }
                ],
                'ffmpeg_location': self.ffmpeg_path,  # ✅ use hardcoded path
            }

            loop = asyncio.get_running_loop()

            def download_and_extract():
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(video_path_or_url, download=True)
                    return Path(self.temp_dir) / f"{info['id']}.mp3"

            audio_path = await loop.run_in_executor(None, download_and_extract)

            if not audio_path.exists():
                raise FileNotFoundError(f"Failed to extract audio at {audio_path}.")

            # --- Transcribe with Groq ---
            print(f"Transcribing '{audio_path.name}' with Groq Whisper...")
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=file,
                    model="whisper-large-v3"
                )

            transcript_text = transcription.text.strip()
            if not transcript_text:
                return "Could not transcribe any text from the video."

            print("Generating final summary from transcript...")
            return self.summarizer.generate_summary(
                context=transcript_text,
                query="the content of a video",
                age_group=age_group
            )

        except Exception as e:
            return f"Could not process video. Error: {e}"

        finally:
            if audio_path and audio_path.exists():
                audio_path.unlink()
