import time
import logging
import asyncio

from core.rag_system import TextRAGSystem
from core.web_fetcher import WebFetcher
from core.summarizer import GeminiSummarizer
from services.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class SearchEngine:
    def __init__(self):
        self.rag_system = TextRAGSystem()
        self.web_fetcher = WebFetcher()
        self.summarizer = GeminiSummarizer()
        self.audio_processor = AudioProcessor(self.summarizer)

    async def _get_web_content_and_summary(self, query: str, age_group: str) -> dict:
        """Gets content from RAG or Web and generates a summary."""
        rag_results = self.rag_system.search(query, k=1)
        context, metadata, source_type, confidence = "", {}, "No Results", 0.0

        if rag_results and rag_results[0]['score'] > 0.65:
            context = rag_results[0]['text']
            metadata = rag_results[0]['metadata']
            source_type, confidence = "Knowledge Base (RAG)", rag_results[0]['score']
        else:
            web_result = await self.web_fetcher.fetch_and_parse_best_result(query)
            if web_result:
                context = web_result['text']
                metadata = web_result['metadata']
                source_type, confidence = "Web Learned", 0.5
                self.rag_system.add_documents([web_result])
        
        summary = self.summarizer.generate_summary(context, query, age_group) if context else "Sorry, I could not find information on that topic."
        return {"summary": summary, "metadata": metadata, "source_type": source_type, "confidence": confidence}

    async def search(self, query: str, age_group: str) -> dict:
        """Orchestrates a multi-source search and suggests a video."""
        start_time = time.time()
        
        # Quickly search for a video suggestion first
        video_suggestion = self.audio_processor.search_for_video(query)
        
        # Run the main web search
        web_result = await self._get_web_content_and_summary(query, age_group)
        
        # If a video was found, start its slow audio processing in the background
        if video_suggestion:
            asyncio.create_task(
                self.audio_processor.get_summary_from_youtube_audio(
                    video_id=video_suggestion['id'],
                    video_title=video_suggestion['title']
                )
            )

        # Combine results for the final response
        final_result = {
            "query": query,
            "age_group": age_group,
            "summary": web_result["summary"],
            "source": web_result["metadata"].get('source', 'N/A'),
            "title": web_result["metadata"].get('title', query),
            "type": web_result["source_type"],
            "confidence": web_result["confidence"],
            "youtube_suggestion": video_suggestion, # This adds the video info to the response
            "processing_time": time.time() - start_time
        }
        
        return final_result