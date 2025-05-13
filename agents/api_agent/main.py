from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import yfinance as yf
import logging
import os
from typing import Optional, Dict, Any

# Basic Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---

class StockData(BaseModel):
    symbol: str
    current_price: Optional[float] = Field(None, description="Current market price")
    previous_close: Optional[float] = Field(None, description="Previous closing price")
    day_high: Optional[float] = Field(None, description="Highest price during the current trading day")
    day_low: Optional[float] = Field(None, description="Lowest price during the current trading day")
    volume: Optional[int] = Field(None, description="Trading volume for the current day")
    market_cap: Optional[int] = Field(None, description="Market capitalization")
    fifty_two_week_high: Optional[float] = Field(None, description="52-week high price")
    fifty_two_week_low: Optional[float] = Field(None, description="52-week low price")
    # For earnings analysis agent
    trailing_eps: Optional[float] = Field(None, alias="trailingEps", description="Trailing Earnings Per Share")
    forward_eps: Optional[float] = Field(None, alias="forwardEps", description="Forward Earnings Per Share")
    earnings_quarterly_growth: Optional[float] = Field(None, alias="earningsQuarterlyGrowth", description="Quarterly earnings growth yoy")
    # Add any other fields you might find useful from yf.Ticker(symbol).info
    raw_info: Optional[Dict[str, Any]] = Field(None, description="Raw info dictionary from yfinance for additional fields")


class StockDataResponse(BaseModel):
    data: Optional[StockData] = None
    error: Optional[str] = None


# --- FastAPI App ---
app = FastAPI(
    title="Finance API Agent",
    description="Microservice for fetching stock market data using yfinance.",
    version="0.1.0",
)

# --- Helper Functions ---

def get_stock_info(symbol: str) -> Dict[str, Any]:
    """
    Fetches stock information for a given symbol using yfinance.
    Returns the .info dictionary.
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info # .info can be empty for some symbols or if data is unavailable

        # Check if essential price data is missing from .info
        if not info or (info.get('regularMarketPrice') is None and info.get('currentPrice') is None):
            logger.warning(f"No detailed info found for symbol: {symbol} via .info. Ticker info keys: {list(info.keys()) if info else 'None'}. Attempting fallback.")
            # Fallback for some ETFs or indices that might not have comprehensive 'info' but have 'history'
            hist = ticker.history(period="2d") # Fetch last 2 days to get current and previous
            if not hist.empty and 'Close' in hist.columns and len(hist['Close']) > 0:
                fallback_info = {'symbol': symbol}
                fallback_info['currentPrice'] = hist['Close'].iloc[-1]
                if len(hist['Close']) > 1:
                    fallback_info['previousClose'] = hist['Close'].iloc[-2]
                if 'Open' in hist.columns and len(hist['Open']) > 0:
                    fallback_info['regularMarketOpen'] = hist['Open'].iloc[-1]
                if 'High' in hist.columns and len(hist['High']) > 0:
                    fallback_info['regularMarketDayHigh'] = hist['High'].iloc[-1]
                if 'Low' in hist.columns and len(hist['Low']) > 0:
                    fallback_info['regularMarketDayLow'] = hist['Low'].iloc[-1]
                if 'Volume' in hist.columns and len(hist['Volume']) > 0:
                    fallback_info['regularMarketVolume'] = hist['Volume'].iloc[-1]
                
                logger.info(f"Using fallback data for {symbol} from history: Price={fallback_info.get('currentPrice')}")
                return fallback_info
            else:
                logger.warning(f"Fallback failed for {symbol}, no history data found or 'Close' column missing.")
                return {} # Return empty if no info and no fallback history data
        return info # Return .info if it seems to contain price data
    except Exception as e:
        logger.error(f"Exception fetching data for {symbol} with yfinance: {e}", exc_info=True)
        return {} # Return empty dict on error during yfinance call


# --- API Endpoints ---

@app.get("/stock/{symbol}", response_model=StockDataResponse)
async def get_stock_data_endpoint(symbol: str):
    """
    Fetches and returns stock data for a given symbol.
    This endpoint is used by the Analysis agent.
    """
    logger.info(f"Received request for stock data: {symbol}")
    symbol_upper = symbol.upper()
    # No try-except here, get_stock_info handles yfinance exceptions
    stock_info = get_stock_info(symbol_upper)

    if not stock_info or (stock_info.get('regularMarketPrice') is None and stock_info.get('currentPrice') is None and not stock_info.get('previousClose')):
        logger.warning(f"No usable data returned from yfinance helper for symbol: {symbol_upper}")
        # Still return a 200 but with an error message in the payload and minimal data
        return StockDataResponse(
            data=StockData(symbol=symbol_upper, raw_info=stock_info if stock_info else {}), # Pass empty dict if stock_info is None
            error=f"No detailed stock data found for symbol {symbol_upper} after attempting fallbacks."
        )

    # Map yfinance fields to our Pydantic model
    current_price_options = [stock_info.get('currentPrice'), stock_info.get('regularMarketPrice')]
    current_price = next((price for price in current_price_options if price is not None), None)

    data = StockData(
        symbol=symbol_upper,
        current_price=current_price,
        previous_close=stock_info.get('previousClose'),
        day_high=stock_info.get('dayHigh', stock_info.get('regularMarketDayHigh')),
        day_low=stock_info.get('dayLow', stock_info.get('regularMarketDayLow')),
        volume=stock_info.get('volume', stock_info.get('regularMarketVolume')),
        market_cap=stock_info.get('marketCap'),
        fifty_two_week_high=stock_info.get('fiftyTwoWeekHigh'),
        fifty_two_week_low=stock_info.get('fiftyTwoWeekLow'),
        trailing_eps=stock_info.get('trailingEps'), # Already aliased in Pydantic model
        forward_eps=stock_info.get('forwardEps'),   # Already aliased in Pydantic model
        earnings_quarterly_growth=stock_info.get('earningsQuarterlyGrowth'), # Already aliased
        raw_info=stock_info
    )
    logger.info(f"Successfully processed data for {symbol_upper}: Price={data.current_price}")
    return StockDataResponse(data=data)
# Removed the broad try-except from the endpoint as get_stock_info handles yfinance specific errors
# and returns {}. The endpoint now focuses on formatting that output.
# If get_stock_info itself raises an unhandled exception (it shouldn't if coded as above),
# FastAPI's default 500 error handling will take over, which is fine.


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    # Could add a quick yfinance check for a common symbol if needed,
    # but for now, just checks if the service is running.
    logger.info("Health check successful for API Agent")
    return {"status": "ok", "service": "Finance API Agent"}

# --- Running the App (for local development) ---
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Finance API Agent service...")
    # Run on port 8001 as per project description
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info") 