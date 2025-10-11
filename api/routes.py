from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path

# Import all services and utilities
from core.search_engine import SearchEngine
from services.document_processor import DocumentProcessor
from services.image_processor import ImageProcessor
from services.video_processor import VideoProcessor
from core.translator import CachingTranslator
from core.tts_service import TextToSpeechService
from core.utils import save_text_to_file

router = APIRouter()

# Initialize all service classes
engine = SearchEngine()
doc_processor = DocumentProcessor()
image_processor = ImageProcessor()
video_processor = VideoProcessor(engine.summarizer)
translator = CachingTranslator()
tts_service = TextToSpeechService()

UPLOADS_DIR = Path("data/uploads")
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)

SUPPORTED_LANGUAGES = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de", "italian": "it", "portuguese": "pt",
    "dutch": "nl", "russian": "ru", "japanese": "ja", "korean": "ko", "chinese (simplified)": "zh-CN",
    "arabic": "ar", "hindi": "hi", "bengali": "bn", "turkish": "tr", "vietnamese": "vi", "polish": "pl",
    "swedish": "sv", "finnish": "fi", "indonesian": "id", "tamil": "ta", "telugu": "te", "marathi": "mr",
    "urdu": "ur", "malayalam": "ml", "kannada": "kn", "gujarati": "gu", "punjabi": "pa"
}

# --- Pydantic Models for Request Bodies ---
class TranslationRequest(BaseModel):
    text: str
    target_language: str

class TTSRequest(BaseModel):
    text: str
    language_code: str = 'en'

# --- API ENDPOINTS ---

@router.get("/search-and-process", summary="All-in-One Search Endpoint")
async def search_and_process(
    query: str, age_group: str = "adult",
    translate_to: str = Query(None), speak: bool = Query(False), download: bool = Query(False)
):
    result = await engine.search(query, age_group)
    summary = result.get("summary", "")
    if "Sorry" not in summary and "API Error" not in summary:
        if translate_to: result["translated_summary"] = translator.translate(summary, translate_to)
        if speak:
            audio_filepath = tts_service.speak(summary, lang='en', query=query)
            if isinstance(audio_filepath, Path):
                path_str = audio_filepath.as_posix()
                result["audio_download_url"] = path_str.replace("data", "/static", 1)
        if download:
            summary_path = save_text_to_file(summary, query)
            path_str = summary_path.as_posix()
            result["summary_download_url"] = path_str.replace("data", "/static", 1)
    return result

@router.post("/summarize-document/", summary="Summarize a Document")
async def summarize_document(
    age_group: str = Form("adult"), file: UploadFile = File(...), download: bool = Form(False)
):
    filepath = UPLOADS_DIR / file.filename
    with open(filepath, "wb") as buffer: buffer.write(await file.read())
    text = doc_processor.extract_text_from_file(filepath)
    keywords = doc_processor.extract_keywords(text)
    summary = engine.summarizer.generate_summary(text, f"the document {file.filename}", age_group)
    result = {"filename": file.filename, "keywords": keywords, "summary": summary}
    if download:
        summary_path = save_text_to_file(result["summary"], file.filename)
        path_str = summary_path.as_posix()
        result["summary_download_url"] = path_str.replace("data", "/static", 1)
    return result

@router.post("/summarize-image/", summary="Summarize an Image")
async def summarize_image(
    age_group: str = Form("adult"), file: UploadFile = File(...), download: bool = Form(False)
):
    filepath = UPLOADS_DIR / file.filename
    with open(filepath, "wb") as buffer: buffer.write(await file.read())
    summary = await image_processor.get_summary_for_image(filepath, age_group)
    result = {"filename": file.filename, "summary": summary}
    if download:
        summary_path = save_text_to_file(result["summary"], file.filename)
        path_str = summary_path.as_posix()
        result["summary_download_url"] = path_str.replace("data", "/static", 1)
    return result

@router.post("/summarize-video/", summary="Summarize a Video from URL or File")
async def summarize_video(
    age_group: str = Form("adult"),
    video_url: str = Form(None),
    file: UploadFile = File(None)
):
    if not video_url and not file:
        raise HTTPException(status_code=400, detail="Please provide either a video_url or an uploaded file.")
    
    input_source = ""
    if file:
        filepath = UPLOADS_DIR / file.filename
        with open(filepath, "wb") as buffer:
            buffer.write(await file.read())
        input_source = str(filepath)
    else:
        input_source = video_url
        
    result_dict = await video_processor.summarize_video(input_source, age_group)
    return {"input_source": input_source, **result_dict}

@router.get("/languages", summary="Get Available Translation Languages")
def get_available_languages():
    return SUPPORTED_LANGUAGES