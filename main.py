import sys
import webbrowser
from pathlib import Path

# Adds the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from api.routes import router

app = FastAPI(title="Multi-Modal AI Assistant")

# Add CORS Middleware to allow the UI to connect
origins = ["null", "http://localhost", "http://127.0.0.1:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folders for downloads
app.mount("/static", StaticFiles(directory="data"), name="static")

# Include all your API endpoints
app.include_router(router, prefix="/api")

# Serve the index.html file at the root URL ("/")
@app.get("/", response_class=FileResponse)
def read_root():
    return "index.html"

if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    
    # --- THIS IS THE FIX ---
    # Open the URL in a new browser tab when the script runs
    webbrowser.open_new_tab(url)
    
    # Run the server
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)