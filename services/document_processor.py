from pathlib import Path
import fitz  # PyMuPDF library for reading PDFs
import docx  # Library for reading .docx files
from core.summarizer import GeminiSummarizer

class DocumentProcessor:
    def __init__(self):
        # We reuse the main summarizer for the keyword extraction task
        self.summarizer = GeminiSummarizer()

    def extract_text_from_file(self, filepath: Path) -> str:
        """Extracts all text from a given .txt, .pdf, or .docx file."""
        suffix = filepath.suffix.lower()
        
        if suffix == '.txt':
            return filepath.read_text(encoding='utf-8', errors='ignore')
        
        elif suffix == '.pdf':
            try:
                with fitz.open(filepath) as doc:
                    return "".join(page.get_text() for page in doc)
            except Exception as e:
                return f"Error reading PDF file: {e}"
        
        elif suffix == '.docx':
            try:
                doc = docx.Document(filepath)
                return "\n".join([para.text for para in doc.paragraphs])
            except Exception as e:
                return f"Error reading DOCX file: {e}"
        
        else:
            raise ValueError("Unsupported file type. Please use .txt, .pdf, or .docx.")

    def extract_keywords(self, text: str) -> list[str]:
        """Uses the configured LLM to extract the most important keywords from text."""
        if not self.summarizer.client:
            return ["Keyword extraction service unavailable."]
        
        keyword_prompt = "Extract the 5 most important and relevant keywords from the following text. Return only a comma-separated list. For example: keyword one, keyword two, keyword three"
        
        try:
            keywords_str = self.summarizer.generate_summary(
                context=text[:4000], # Limit context size
                query="keyword extraction",
                age_group="adult"
            )
            return [keyword.strip() for keyword in keywords_str.split(',')]
        except Exception as e:
            print(f"Could not extract keywords: {e}")
            return ["Keyword extraction failed."]