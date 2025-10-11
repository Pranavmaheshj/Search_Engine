from .youtube_processor import VideoProcessor

# Re-export for backward compatibility with imports that expect
# `services.video_processor.VideoProcessor`
__all__ = ["VideoProcessor"]
