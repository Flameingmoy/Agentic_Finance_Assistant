from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sec_edgar_downloader import Downloader
import logging
import os
from pathlib import Path
from datetime import date
import httpx # Added for calling retriever agent

# --- Configuration ---
# Use an environment variable for the base data path, default to ../../data relative to this file
# Resolve makes it absolute, ensuring consistency
BASE_DATA_PATH = Path(os.getenv("FINANCE_ASSISTANT_DATA_PATH", "../../data")).resolve()
DOWNLOAD_PATH = BASE_DATA_PATH / "filings"
DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)
COMPANY_EMAIL = os.getenv("COMPANY_EMAIL", "your_email@example.com") # SEC requires an email for the User-Agent
# Add Retriever Agent URL
RETRIEVER_AGENT_URL = os.getenv("RETRIEVER_AGENT_URL", "http://localhost:8003")

# Basic Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class FilingRequest(BaseModel):
    ticker: str
    filing_type: str = "10-K" # Default to 10-K, kept as string
    limit: int = 1 # Default to the single most recent filing
    after_date: date | None = None # Use standard date objects
    before_date: date | None = None # Use standard date objects

class DownloadResponse(BaseModel):
    message: str
    ticker: str
    filing_type: str
    num_downloaded: int # -1 indicates task queued, actual count will be logged
    download_location: str

# --- FastAPI App ---
app = FastAPI(
    title="Finance Scraping Agent",
    description="Microservice for downloading financial filings (e.g., SEC EDGAR).",
    version="0.1.0",
)

# --- Helper Function (for Background Task) ---
def download_filings_sync(ticker: str, filing_type_str: str, limit: int, after_date_iso: str | None, before_date_iso: str | None, download_path_str: str, company_email: str):
    """Synchronous function to download filings, designed for background execution."""
    num_downloaded = 0 # Initialize
    try:
        logger.info(f"[Background Task] Initializing downloader for {ticker}, type {filing_type_str}, limit {limit}")
        dl = Downloader(company_name=ticker, email_address=company_email, download_path=download_path_str)

        # Convert ISO date strings back to date objects if provided
        after = date.fromisoformat(after_date_iso) if after_date_iso else None
        before = date.fromisoformat(before_date_iso) if before_date_iso else None

        num_downloaded = dl.get(filing_type_str, ticker, limit=limit, after=after, before=before)
        logger.info(f"[Background Task] Downloaded {num_downloaded} filings for {ticker} ({filing_type_str}) to {download_path_str}")

    except Exception as e:
        logger.error(f"[Background Task] Error during filing download for {ticker} ({filing_type_str}): {e}", exc_info=True)
        return 0 # Indicate download failure

    # --- Trigger Retriever Agent Indexing --- 
    # We attempt this even if num_downloaded is 0, in case previous downloads need indexing
    # A more robust system might track download batches
    # if num_downloaded > 0: # Optionally, only trigger if new files were downloaded
    try:
        logger.info(f"[Background Task] Attempting to trigger indexing on Retriever Agent at {RETRIEVER_AGENT_URL}")
        # Note: This call is synchronous within the background task.
        # For true async behavior within the sync func, would need asyncio.run or similar.
        with httpx.Client() as client: # Use sync client here as this function is sync
            index_url = f"{RETRIEVER_AGENT_URL}/index-filings"
            # The index endpoint might not need a payload, send empty json if needed
            response = client.post(index_url, json={}, timeout=30.0) # Longer timeout for potential indexing start
            response.raise_for_status()
            logger.info(f"[Background Task] Successfully triggered indexing on Retriever Agent. Response: {response.text[:100]}")
    except httpx.RequestError as exc:
        logger.error(f"[Background Task] HTTP error triggering indexing on Retriever Agent: {exc}")
    except Exception as e:
        logger.error(f"[Background Task] Error triggering indexing on Retriever Agent: {e}", exc_info=True)
    # --- End Trigger --- 

    return num_downloaded # Return the number downloaded from the initial download step

# --- API Endpoints ---
@app.post("/download-filings", response_model=DownloadResponse)
async def trigger_filing_download(request: FilingRequest, background_tasks: BackgroundTasks):
    """
    Triggers the download of specified SEC filings for a given ticker.
    Downloads occur in the background using FastAPI's BackgroundTasks.
    Requires COMPANY_EMAIL env var or default for SEC User-Agent.
    """
    # Validate Filing Type String before queuing the task
    # For sec-edgar-downloader, the filing type is a string.
    # We can do a basic sanity check here, but the library will ultimately validate it.
    # Example: Check if it's empty or excessively long, or matches a known pattern.
    # For now, we'll rely on the library's internal validation when dl.get() is called.
    # Consider adding a list of common valid types if stricter pre-validation is needed.
    if not request.filing_type or len(request.filing_type) > 10: # Basic check
        logger.error(f"Potentially invalid filing type requested: {request.filing_type}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or malformed filing type: '{request.filing_type}'. Please provide a valid SEC form type string (e.g., '10-K', '8-K')."
        )

    if not COMPANY_EMAIL or "your_email@example.com" in COMPANY_EMAIL:
        logger.warning("Using default/missing COMPANY_EMAIL. SEC requests a unique email for identification.")
        # Depending on requirements, you might raise an HTTPException here
        # raise HTTPException(status_code=400, detail="COMPANY_EMAIL environment variable not set.")

    logger.info(f"Received request to download filings: Ticker={request.ticker}, Type={request.filing_type}, Limit={request.limit}")

    # Convert dates to ISO strings for background task compatibility
    after_date_str = request.after_date.isoformat() if request.after_date else None
    before_date_str = request.before_date.isoformat() if request.before_date else None

    # Ensure download path is passed as a string
    download_path_str = str(DOWNLOAD_PATH)

    background_tasks.add_task(
        download_filings_sync,
        ticker=request.ticker,
        filing_type_str=request.filing_type,
        limit=request.limit,
        after_date_iso=after_date_str,
        before_date_iso=before_date_str,
        download_path_str=download_path_str,
        company_email=COMPANY_EMAIL
    )

    # Return an immediate response confirming the task is queued
    return DownloadResponse(
        message="Filing download task queued. Check logs for completion and details.",
        ticker=request.ticker,
        filing_type=request.filing_type,
        num_downloaded=-1, # Indicates task queued, not completed yet
        download_location=download_path_str
    )

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}

# --- Running the App (for local development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Scraping Agent service... Filings will be downloaded to: {DOWNLOAD_PATH}")
    # Run on port 8002
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info") 