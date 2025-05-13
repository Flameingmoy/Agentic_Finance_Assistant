from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import os
import httpx  # For making async requests to other agents
from typing import List, Dict, Any, Optional
import asyncio # For potential parallel API calls

# --- Configuration ---
API_AGENT_URL = os.getenv("API_AGENT_URL", "http://localhost:8001") # URL of the API Agent

# Basic Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---

# --- Risk Exposure ---
class PortfolioAsset(BaseModel):
    symbol: str
    value: float # Current market value of the holding
    # Optional metadata for filtering
    region: Optional[str] = None
    sector: Optional[str] = None

class RiskExposureRequest(BaseModel):
    assets: List[PortfolioAsset]
    total_aum: float = Field(..., gt=0, description="Total Assets Under Management for percentage calculation.")
    filter_criteria: Dict[str, str] = Field(default_factory=dict, description="Criteria to filter assets (e.g., {\"region\": \"Asia\", \"sector\": \"Tech\"})")

class RiskExposureResponse(BaseModel):
    criteria: Dict[str, str]
    matched_value: float
    total_aum: float
    exposure_percentage: float

# --- Earnings Data ---
class EarningsRequest(BaseModel):
    tickers: List[str]
    # Optional: Pass estimates if available from another source
    estimates: Optional[Dict[str, float]] = Field(default_factory=dict, description="Optional dictionary mapping ticker to estimated EPS.")

class TickerEarningsData(BaseModel):
    symbol: str
    # Fields from yfinance.info - may be None if not available
    trailing_eps: Optional[float] = None
    forward_eps: Optional[float] = None
    earnings_quarterly_growth: Optional[float] = None # yFinance 'earningsQuarterlyGrowth'
    # Add more fields as needed, e.g., report date
    estimate_eps: Optional[float] = None # Provided estimate
    surprise_percentage: Optional[float] = None # Calculated if actual and estimate exist
    error: Optional[str] = None # If fetching data failed for this ticker

class EarningsResponse(BaseModel):
    results: List[TickerEarningsData]


# --- FastAPI App ---
app = FastAPI(
    title="Finance Analysis Agent",
    description="Microservice for performing financial analysis and calculations.",
    version="0.1.0",
)

# --- Helper Functions ---

async def get_stock_data_from_api_agent(symbol: str, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
    """Calls the API Agent to get stock data."""
    try:
        url = f"{API_AGENT_URL}/stock/{symbol}"
        logger.info(f"Querying API Agent for {symbol} at {url}")
        response = await client.get(url, timeout=10.0) # Add timeout
        response.raise_for_status() # Raise exception for 4xx or 5xx errors
        data = response.json()
        logger.info(f"Successfully received data for {symbol} from API Agent.")
        return data
    except httpx.RequestError as exc:
        logger.error(f"HTTP error calling API Agent for {symbol}: {exc}")
        return None
    except Exception as e:
        logger.error(f"Error processing data for {symbol} from API Agent: {e}", exc_info=True)
        return None


# --- API Endpoints ---

@app.post("/analyze/risk-exposure", response_model=RiskExposureResponse)
async def analyze_risk_exposure(request: RiskExposureRequest):
    """
    Calculates risk exposure for a segment of the portfolio based on filtering criteria.
    Requires the portfolio composition (assets with values and metadata) and total AUM.
    """
    logger.info(f"Received risk exposure analysis request with criteria: {request.filter_criteria}")

    if not request.assets:
        raise HTTPException(status_code=400, detail="Asset list cannot be empty.")

    matched_value = 0.0
    try:
        for asset in request.assets:
            match = True
            # Check if the asset matches all filter criteria
            for key, value in request.filter_criteria.items():
                asset_attr = getattr(asset, key, None)
                if asset_attr is None or str(asset_attr).lower() != str(value).lower():
                    match = False
                    break
            if match:
                matched_value += asset.value

        if request.total_aum <= 0:
             raise HTTPException(status_code=400, detail="Total AUM must be positive.")

        exposure_percentage = (matched_value / request.total_aum) * 100 if request.total_aum > 0 else 0

        logger.info(f"Risk exposure calculation complete. Matched Value: {matched_value}, Exposure: {exposure_percentage:.2f}%")

        return RiskExposureResponse(
            criteria=request.filter_criteria,
            matched_value=matched_value,
            total_aum=request.total_aum,
            exposure_percentage=round(exposure_percentage, 2) # Return rounded percentage
        )
    except Exception as e:
      logger.error(f"Error calculating risk exposure: {e}", exc_info=True)
      # Re-raise as internal server error after logging
      raise HTTPException(status_code=500, detail=f"Internal server error during risk calculation: {e}")


@app.post("/analyze/earnings", response_model=EarningsResponse)
async def analyze_earnings(request: EarningsRequest):
    """
    Fetches available earnings data for a list of tickers from the API Agent.
    Calculates surprise percentage if estimates are provided.
    """
    logger.info(f"Received earnings analysis request for tickers: {request.tickers}")
    results: List[TickerEarningsData] = []

    # Use httpx.AsyncClient for efficient connection pooling
    async with httpx.AsyncClient() as client:
        # Create tasks to fetch data for all tickers concurrently
        tasks = [get_stock_data_from_api_agent(ticker, client) for ticker in request.tickers]
        api_responses = await asyncio.gather(*tasks)

    for i, ticker in enumerate(request.tickers):
        data = api_responses[i]
        ticker_result = TickerEarningsData(symbol=ticker.upper())
        provided_estimate = request.estimates.get(ticker) # Get estimate if provided
        ticker_result.estimate_eps = provided_estimate

        if data:
            # Extract relevant fields - use .get() for safety
            ticker_result.trailing_eps = data.get('trailingEps') # Use correct yfinance key
            ticker_result.forward_eps = data.get('forwardEps')   # Use correct yfinance key
            ticker_result.earnings_quarterly_growth = data.get('earningsQuarterlyGrowth') # Use correct yfinance key

            # --- Calculate Surprise ---
            actual_eps = ticker_result.trailing_eps # Using trailing EPS as the 'actual' value for surprise calc
            if actual_eps is not None and provided_estimate is not None:
                try:
                    if provided_estimate == 0: # Avoid division by zero
                         surprise = (actual_eps - provided_estimate) * 100 # Treat as absolute difference * 100 ? Or handle differently?
                    else:
                         surprise = ((actual_eps - provided_estimate) / abs(provided_estimate)) * 100
                    ticker_result.surprise_percentage = round(surprise, 2)
                except ZeroDivisionError:
                     logger.warning(f"Cannot calculate surprise percentage for {ticker} due to zero estimate.")
                except Exception as calc_e:
                     logger.error(f"Error calculating surprise for {ticker}: {calc_e}")

        else:
            ticker_result.error = f"Failed to fetch data from API Agent for {ticker}"
            logger.warning(f"No data retrieved for {ticker} from API Agent.")

        results.append(ticker_result)

    logger.info(f"Earnings analysis complete for {len(request.tickers)} tickers.")
    return EarningsResponse(results=results)


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    # Optionally, check connectivity to the API Agent
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_AGENT_URL}/health", timeout=2.0)
            api_agent_status = "ok" if response.status_code == 200 else "error"
    except Exception:
        api_agent_status = "unreachable"

    return {"status": "ok", "dependencies": {"api_agent": api_agent_status}}

# --- Running the App (for local development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Analysis Agent service...")
    # Run on port 8004
    uvicorn.run(app, host="0.0.0.0", port=8004, log_level="info") 