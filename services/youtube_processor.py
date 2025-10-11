import os
import asyncio
from pathlib import Path
import yt_dlp
import whisper
import torch
import ffmpeg
from core.summarizer import GeminiSummarizer
from services.image_processor import ImageProcessor
from typing import List, Dict

class VideoProcessor:
    def __init__(self, summarizer: GeminiSummarizer):
        self.summarizer = summarizer
        self.image_processor = ImageProcessor() # For analyzing frames
        self.temp_dir = Path("data/temp_video_processing")
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading local Whisper model on {self.device}...")
        self.whisper_model = whisper.load_model("base", device=self.device)
        print("Whisper model loaded.")

        # --- Hardcoded FFmpeg path ---
        # IMPORTANT: Replace this placeholder with your actual path, using double backslashes \\
        self.ffmpeg_location = r"C:\Users\prana\Downloads\ffmpeg-2025-09-28-git-0fdb5829e3-full_build\ffmpeg-2025-09-28-git-0fdb5829e3-full_build\bin\ffmpeg.exe"
        if not Path(self.ffmpeg_location).exists():
            raise FileNotFoundError(f"FFmpeg not found at the hardcoded path: {self.ffmpeg_location}")

    async def summarize_video(self, video_path_or_url: str, age_group: str) -> dict:
        local_video_path = None
        audio_path = None
        frame_paths = []
        is_url = video_path_or_url.lower().startswith("http")
        
        try:
            print(f"Processing video source: {video_path_or_url}")
            loop = asyncio.get_running_loop()
            
            # --- Step 1: Get Video Metadata and File Path ---
            video_metadata = {}
            if is_url:
                def download_video_and_get_metadata():
                    ydl_opts = {
                        'format': 'best[ext=mp4][height<=480]',
                        'outtmpl': str(self.temp_dir / '%(id)s.%(ext)s'),
                        'quiet': True,
                        'ffmpeg_location': self.ffmpeg_location
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(video_path_or_url, download=True)
                        filepath = Path(ydl.prepare_filename(info))
                        metadata = { "title": info.get("title"), "uploader": info.get("uploader"), "duration_string": info.get("duration_string") }
                        return filepath, metadata
                local_video_path, video_metadata = await loop.run_in_executor(None, download_video_and_get_metadata)
            else:
                local_video_path = Path(video_path_or_url)
                video_metadata = {"title": local_video_path.name}

            if not local_video_path or not local_video_path.exists():
                raise FileNotFoundError("Failed to get local video file.")

            # --- Step 2 & 3: Run Audio and Visual Analysis in Parallel ---
            async def get_audio_transcript():
                def extract_and_transcribe():
                    output_audio_path = self.temp_dir / f"{local_video_path.stem}_audio.mp3"
                    ffmpeg.input(str(local_video_path)).output(str(output_audio_path), acodec='mp3').run(cmd=self.ffmpeg_location, overwrite_output=True, quiet=True)
                    if not output_audio_path.exists() or os.path.getsize(output_audio_path) == 0:
                        return "", output_audio_path # Return empty transcript if no audio
                    
                    decode_options = {"fp16": False}
                    transcription = self.whisper_model.transcribe(str(output_audio_path), **decode_options)
                    return transcription.get('text', '').strip(), output_audio_path
                return await loop.run_in_executor(None, extract_and_transcribe)
            
            async def get_visual_description():
                frame_paths_local = self._extract_frames(local_video_path, self.ffmpeg_location, num_frames=5)
                if not frame_paths_local: return "No visual content could be analyzed.", []
                
                tasks = [self.image_processor.get_summary_for_image(frame, "adult") for frame in frame_paths_local]
                frame_descriptions = await asyncio.gather(*tasks)
                
                combined_desc = "\n".join(f"- {desc}" for desc in frame_descriptions if "Could not process" not in desc)
                return self.summarizer.generate_summary(combined_desc, "a summary of key visual scenes in a video", "adult"), frame_paths_local

            (transcript_text, audio_path), (visual_description, frame_paths) = await asyncio.gather(get_audio_transcript(), get_visual_description())

            # --- Step 4: Combine Analyses and Generate Final Summary ---
            combined_context = f"AUDIO TRANSCRIPT:\n{transcript_text or 'None'}\n\nVISUAL DESCRIPTION:\n{visual_description}"
            final_summary = self.summarizer.generate_summary(
                context=combined_context, query=f"a comprehensive summary of this video's audio and visuals, titled '{video_metadata.get('title', 'Unknown')}'", age_group=age_group
            )
            
            return {
                "metadata": {k: v for k, v in video_metadata.items() if v},
                "transcript": transcript_text,
                "summary": final_summary
            }
            
        except Exception as e:
            return {"summary": f"Could not process video. Error: {type(e).__name__}: {e}"}
        finally:
            # Final cleanup
            if audio_path and Path(audio_path).exists(): Path(audio_path).unlink()
            if local_video_path and Path(local_video_path).exists() and is_url: Path(local_video_path).unlink()
            for frame in frame_paths:
                if Path(frame).exists(): Path(frame).unlink()

    def _extract_frames(self, video_path: Path, ffmpeg_cmd: str, num_frames: int = 5) -> List[Path]:
        frames_dir = self.temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        try:
            probe = ffmpeg.probe(str(video_path), cmd=ffmpeg_cmd)
            duration = float(next((s['duration'] for s in probe['streams'] if 'duration' in s), 0))
            if duration == 0: return []
            
            interval = duration / (num_frames + 1)
            frame_paths = []
            
            for i in range(1, num_frames + 1):
                timestamp = interval * i
                frame_path = frames_dir / f"frame_{i:03d}.jpg"
                ffmpeg.input(str(video_path), ss=timestamp).output(str(frame_path), vframes=1).run(cmd=ffmpeg_cmd, quiet=True, overwrite_output=True)
                frame_paths.append(frame_path)
            
            return frame_paths
        except Exception as e:
            print(f"Frame extraction failed: {e}")
            return []


# Backwards-compatibility alias
class YouTubeProcessor(VideoProcessor):
    """Alias kept for compatibility with older imports that expect a
    `YouTubeProcessor` class. Inherit from `VideoProcessor` so behaviour
    is identical.
    """
    def search_and_summarize_video(self, query: str) -> dict:
        """Search for a video matching the query and return its metadata."""
        print(f"Searching YouTube for: '{query}'...")
        ydl_opts = {
            'quiet': True,
            'extract_flat': True,
            'default_search': 'ytsearch1:' + query,
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Use 'ytsearch:query' format for searching
                result = ydl.extract_info(f'ytsearch:{query}', download=False)
                videos = result.get('entries', []) if result else []
                if videos:
                    video = videos[0]
                    return {
                        "id": video.get('id', 'unknown'),
                        "title": video.get('title', query),
                        "url": video.get('url', ''),
                        "thumbnail": video.get('thumbnail', ''),
                        "status": "success"
                    }
        except Exception as e:
            print(f"YouTube search with yt-dlp failed: {e}")
        
        return {
            "id": None,
            "title": None,
            "url": None,
            "thumbnail": None,
            "status": "error",
            "error": "Failed to find or process video"
        }