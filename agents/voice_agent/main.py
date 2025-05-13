# finance_assistant/agents/voice_agent/main.py

import fastapi
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
# from fastapi.responses import StreamingResponse # No longer streaming TTS
from pydantic import BaseModel
import logging
import os
# import whisper # Replaced by Groq API
# import pyttsx3 # Removed
import soundfile as sf # Still needed for processing input audio
import io
import tempfile
import numpy as np
from pathlib import Path
import asyncio
from groq import Groq, AsyncGroq # Import Groq clients
from dotenv import load_dotenv

# --- Configuration ---
# Load .env file for Groq API Key
dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

# WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base") # Local model name
GROQ_STT_MODEL_NAME = os.getenv("GROQ_STT_MODEL_NAME", "whisper-large-v3") # Groq STT model
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY environment variable not set.")
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Temporary directory for potential intermediate files if needed
# TEMP_DIR = Path(tempfile.gettempdir()) / "finance_assistant_tts"
# TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Basic Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dedicated logger for AI model usage
ai_usage_logger = logging.getLogger("ai_usage")
# Configure ai_usage_logger if needed

# Global variable for Whisper model (loaded on startup)
# whisper_model = None # No longer loading local model

# Global lock for TTS engine to prevent concurrent access issues
# pyttsx3 might not be thread-safe, especially its runAndWait()
# tts_lock = asyncio.Lock() # Removed TTS lock

# Groq client (can be initialized globally or per request)
# Initializing globally for potential reuse, ensure thread safety if needed
# Note: Groq SDK might handle client pooling/thread safety internally
groq_client = AsyncGroq(api_key=GROQ_API_KEY)


# --- Pydantic Models ---
class STTResponse(BaseModel):
    text: str
    # Groq might not return language by default, handle if needed
    # language: str

# class TTSRequest(BaseModel): # Removed TTS
#     text: str


# --- FastAPI App ---
app = FastAPI(
    title="Finance Voice Agent",
    # description="Microservice for Speech-to-Text (STT) and Text-to-Speech (TTS).",
    description="Microservice for Speech-to-Text (STT) using Groq Cloud.",
    version="0.2.0", # Updated version
)

# --- Helper Functions ---

# def load_whisper_model(): # No longer loading local model
#     """Loads the Whisper model into the global variable."""
#     pass

async def transcribe_audio_groq(audio_data: bytes) -> str:
    """Transcribes audio data using the Groq STT API."""
    global groq_client
    logger.info(f"Starting audio transcription using Groq ({GROQ_STT_MODEL_NAME})...")
    
    # AI Usage Logging - Input
    ai_usage_logger.info(
        f"AI_CALL_START - Model: Groq STT ({GROQ_STT_MODEL_NAME}), Purpose: stt, "
        f"InputSizeBytes: {len(audio_data)}"
    )
    
    try:
        # Groq SDK expects a file-like object or tuple (filename, bytes)
        # Create an in-memory file
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "input_audio.wav" # Groq might need a filename hint
        
        # Make the API call to Groq
        transcription = await groq_client.audio.transcriptions.create(
            file=(audio_file.name, audio_file.read()),
            model=GROQ_STT_MODEL_NAME,
            # Optional parameters: prompt, response_format, language, temperature
            # language="en", # Explicitly set language if known
            # response_format="json" # Default is json with text field
        )
        
        transcribed_text = transcription.text
        logger.info("Groq STT transcription successful.")
        
        # AI Usage Logging - Output
        ai_usage_logger.info(
            f"AI_CALL_END - Model: Groq STT ({GROQ_STT_MODEL_NAME}), Purpose: stt, "
            f"OutputLength: {len(transcribed_text)}, Status: Success"
        )
        
        return transcribed_text

    except Exception as e:
        logger.error(f"Error during Groq STT transcription: {e}", exc_info=True)
        # AI Usage Logging - Error
        ai_usage_logger.error(
            f"AI_CALL_END - Model: Groq STT ({GROQ_STT_MODEL_NAME}), Purpose: stt, Status: Error, ErrorMsg: {str(e)[:100]}"
        )
        # Map potential Groq API errors if needed, otherwise raise generic internal error
        raise HTTPException(status_code=500, detail=f"Groq STT transcription failed: {e}")

# async def synthesize_speech(text: str) -> Path: # Removed TTS
#     """Synthesizes speech using pyttsx3 and saves to a temporary file."""
#     pass


# --- API Endpoints ---

# @app.on_event("startup") # No model loading on startup anymore
# async def startup_event():
#     """Load the Whisper model on application startup."""
#     load_whisper_model()


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(audio: UploadFile = File(...)):
    """
    Receives an audio file, transcribes it using the Groq STT API, and returns the text.
    """
    logger.info(f"Received audio file: {audio.filename}, content type: {audio.content_type}")
    # Basic check for audio content type
    if not audio.content_type or not audio.content_type.startswith("audio/"):
         raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        # Read audio file contents
        audio_bytes = await audio.read()
        logger.info(f"Read {len(audio_bytes)} bytes from audio file.")

        # Transcribe using Groq helper
        transcribed_text = await transcribe_audio_groq(audio_bytes)

        return STTResponse(
            text=transcribed_text.strip(),
            # language="unknown" # Language detection might not be default in Groq response
        )
    except HTTPException as http_exc:
        # Re-raise HTTP exceptions from helpers
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing STT request for {audio.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process audio file: {e}")
    finally:
        # Ensure file resources are closed
        await audio.close()


# @app.post("/tts") # Removed TTS endpoint
# async def text_to_speech(request: TTSRequest):
#     """
#     Receives text, synthesizes it into speech using pyttsx3,
#     and returns the audio as a WAV file stream.
#     """
#     pass

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    # Check connectivity to Groq API? (Might be slow or cost tokens)
    # For now, just confirm service is running
    # whisper_status = "loaded" if whisper_model else "not loaded"
    return {"status": "ok", "dependencies": {"groq_api": "configured" if GROQ_API_KEY else "key_missing"}}


# --- Running the App (for local development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Voice Agent service (STT via Groq)...")
    # Run on port 8006
    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="info") 