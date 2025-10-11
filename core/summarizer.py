import os
from groq import Groq
import logging

logger = logging.getLogger(__name__)

class GeminiSummarizer: # Keeping the class name for consistency
    def __init__(self):
        try:
            # Initialize the Groq client
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables.")
            self.client = Groq(api_key=api_key)
            # Use the current, stable Llama 3.1 model on Groq
            self.model_name = "llama-3.1-8b-instant" 
            logger.info("Groq (Llama 3.1) Summarizer initialized successfully.")
        except Exception as e:
            self.client = None
            logger.error(f"Failed to initialize Groq Summarizer: {e}")

    def _get_system_prompt(self, age_group: str, query: str) -> str:
        """Creates the instruction part of the prompt for the AI model."""
        prompts = {
            "child": f"You are a fun teacher explaining '{query}' to a 6-year-old. The summary must be written in English. Cover the main idea and interesting details. Use simple words and fun emojis.",
            "teen": f"You are a cool creator explaining '{query}' to a teenager. The summary must be written in English. Explain the key takeaways and why it matters. Be thorough.",
            "adult": f"You are a professional analyst providing a complete, detailed summary of '{query}'. The summary must be written in English. Capture all main arguments, facts, and conclusions. Prioritize completeness.",
            "senior": f"You are a calm narrator explaining '{query}'. The summary must be written in English. Explain all essential points in a logical order so the full picture is clear."
        }
        return prompts.get(age_group, prompts["adult"])

    def generate_summary(self, context: str, query: str, age_group: str) -> str:
        if not self.client:
            return "Sorry, the summarization service is currently unavailable."
        if not context or not context.strip():
            return "There is not enough content to summarize."
            
        # Truncate long contexts to stay within the API's free tier limits
        max_chars = 12000
        if len(context) > max_chars:
            logger.warning(f"Context is too long ({len(context)} chars). Truncating to {max_chars} chars.")
            context = context[:max_chars]

        system_instruction = self._get_system_prompt(age_group, query)
        
        try:
            # The 'system' message gives the AI its instructions.
            # The 'user' message provides the text to work on.
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_instruction,
                    },
                    {
                        "role": "user",
                        "content": context,
                    }
                ],
                model=self.model_name,
            )
            summary = chat_completion.choices[0].message.content
            return summary.strip()

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return f"Sorry, the summary could not be generated. API Error: {e}"