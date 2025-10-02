import os
from pathlib import Path
from transformers import pipeline
from core.summarizer import GeminiSummarizer

class ImageProcessor:
    def __init__(self):
        print("Loading local image captioning pipeline (BLIP)...")
        print("This may take a while and download a large model on first startup.")
        
        try:
            # --- THIS IS THE FIX ---
            # Use the high-level pipeline for the "image-to-text" task.
            # It automatically handles the model, processor, tokenizer, and device placement.
            self.captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")
            
            print("BLIP pipeline loaded successfully.")
        except Exception as e:
            print(f"Error loading BLIP pipeline: {e}")
            self.captioner = None
        
        # This uses your main Groq summarizer for the final age-specific summary
        self.summarizer = GeminiSummarizer()

    def get_summary_for_image(self, filepath: Path, age_group: str) -> str:
        """
        Generates a caption using the local BLIP pipeline, then creates an age-specific summary.
        """
        if not self.captioner:
            return "Could not process image: Image captioning pipeline is not loaded."

        try:
            # 1. Generate a base caption using the pipeline.
            # The pipeline can take the file path directly.
            caption_result = self.captioner(str(filepath))
            
            # The pipeline returns a list of dictionaries, like: [{'generated_text': '...'}]
            if not caption_result or "generated_text" not in caption_result[0]:
                 return "The local model could not describe the image."
            
            caption = caption_result[0]['generated_text']

            # 2. Use the generated caption as context for the final age-specific summary
            print(f"Generated base caption: '{caption}'. Now creating final summary...")
            final_summary = self.summarizer.generate_summary(
                context=caption,
                query="an image that was analyzed by a local AI",
                age_group=age_group
            )
            return final_summary

        except Exception as e:
            return f"Could not process image with local model. Error: {e}"