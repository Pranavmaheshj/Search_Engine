import os
import base64
from pathlib import Path
from groq import Groq
import logging
from typing import Union
from core.summarizer import GeminiSummarizer

logger = logging.getLogger(__name__)

class ImageProcessor:
    def __init__(self):
        """
        Initialize the Image Processor with Groq's vision model.
        """
        try:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
            
            self.groq_client = Groq(api_key=api_key)
            # Use Groq's actual vision model - Llama 4 Scout supports vision
            self.vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
            
            logger.info("Image Processor initialized with Groq Llama 4 Scout Vision model.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Groq Vision model: {e}") from e
        
        self.summarizer = GeminiSummarizer()

    def _encode_image_to_base64(self, filepath: Path) -> str:
        """
        Convert an image file to base64 string.
        
        Args:
            filepath: Path to the image file
            
        Returns:
            Base64 encoded string of the image
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Image not found: {filepath}")
        
        with open(filepath, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def get_summary_for_image(self, filepath: Union[str, Path], age_group: str) -> str:
        """
        Analyzes an image using Groq Vision and then summarizes the findings.
        
        Args:
            filepath: Path to the image file
            age_group: Target age group (child, teen, adult, senior)
            
        Returns:
            Age-appropriate summary of the image analysis
        """
        try:
            filepath = Path(filepath)
            
            # Encode image to base64
            base64_image = self._encode_image_to_base64(filepath)
            
            # Comprehensive analysis prompt
            prompt = """
Analyze this image carefully and provide:
1. **Authenticity Analysis:** Determine if this image is likely real or AI-generated and briefly explain why.
2. **Identify People:** Identify any famous people or celebrities and what they are known for (describe in general terms if not famous).
3. **OCR (Text Extraction):** Transcribe any text visible in the image.
4. **Visual Description:** Provide a detailed description of the scene, objects, colors, composition, and overall context.
"""
            
            # Call Groq Vision API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            response = self.groq_client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
                max_tokens=2000,
                temperature=0.7
            )
            
            image_analysis = response.choices[0].message.content.strip()
            
            if not image_analysis.strip():
                return "The vision model could not analyze the image."
            
            # Generate age-appropriate summary using the summarizer
            return self.summarizer.generate_summary(
                context=image_analysis,
                query="an analysis of an image",
                age_group=age_group
            )
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return f"Could not process the image. Error: {e}"