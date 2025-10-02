import sys
from pathlib import Path

# Adds the project root to the Python path to solve import issues
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv() # Loads API keys at the very beginning

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="Multi-Modal Search and Summarization API",
    version="1.0.0"
)

# This section is crucial for allowing the UI to communicate with the server
origins = [
    "null", # Allows opening the index.html file directly
    "http://localhost",
    "http://127.0.0.1",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# This makes the 'data' folder accessible for downloads
app.mount("/static", StaticFiles(directory="data"), name="static")

# Includes all your API endpoints
app.include_router(router, prefix="/api")

@app.get("/", tags=["Root"])
def read_root():
    return {"message": "Welcome to the API. Go to /docs for the interactive playground."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)