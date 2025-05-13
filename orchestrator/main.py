from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import os
import httpx
from typing import List, Dict, Any, Optional
import json # For potential parsing/extraction if needed
import asyncio
import re # For basic ticker pattern matching
import spacy # For NLP parsing

# --- Configuration ---
# URLs for the dependent agent services
ANALYSIS_AGENT_URL = os.getenv("ANALYSIS_AGENT_URL", "http://localhost:8004")
LANGUAGE_AGENT_URL = os.getenv("LANGUAGE_AGENT_URL", "http://localhost:8005")
RETRIEVER_AGENT_URL = os.getenv("RETRIEVER_AGENT_URL", "http://localhost:8003") # URL for Retriever Agent

# Basic Logging Setup
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load spaCy model
try:
    # Using the small English model for efficiency
    nlp = spacy.load("en_core_web_sm")
    logger.info("spaCy model 'en_core_web_sm' loaded successfully.")
except OSError:
    logger.error("spaCy model 'en_core_web_sm' not found. ")
    logger.error("Please download it by running: python -m spacy download en_core_web_sm")
    nlp = None # Set nlp to None if model loading fails
except ImportError:
    logger.error("spacy library not installed. Please install it: pip install spacy")
    nlp = None

# --- Pydantic Models ---

# Simplified input for now - just a text query
class UserQueryRequest(BaseModel):
    query: str
    # In a real scenario, add user ID, session ID, portfolio context, etc.
    # Example portfolio structure - needed for risk analysis
    # This might come from a DB or user profile in a real app
    portfolio_assets: Optional[List[Dict[str, Any]]] = Field(None, description="User's portfolio holdings, e.g., [{'symbol': 'AAPL', 'value': 10000, 'region': 'NA', 'sector':'Tech'}, ...]")
    portfolio_aum: Optional[float] = Field(None, description="User's total Assets Under Management")


# Define expected structures from Analysis Agent for clarity
class RiskExposureResponseModel(BaseModel):
    criteria: Dict[str, str]
    matched_value: float
    total_aum: float
    exposure_percentage: float

class TickerEarningsDataModel(BaseModel):
    symbol: str
    trailing_eps: Optional[float] = None
    forward_eps: Optional[float] = None
    earnings_quarterly_growth: Optional[float] = None
    estimate_eps: Optional[float] = None
    surprise_percentage: Optional[float] = None
    error: Optional[str] = None

class EarningsResponseModel(BaseModel):
    results: List[TickerEarningsDataModel]

# Retriever Agent Models (mirroring Retriever Agent for clarity)
class RetrieverSearchQueryModel(BaseModel):
    query: str
    top_k: int = 4

class RetrieverSearchResultModel(BaseModel):
    content: str
    metadata: dict
    score: float

class RetrieverSearchResponseModel(BaseModel):
    results: List[RetrieverSearchResultModel]

# Output of the orchestrator (intermediate, before language agent)
class OrchestratorResponse(BaseModel):
    query: str
    risk_analysis: Optional[RiskExposureResponseModel] = None
    earnings_analysis: Optional[EarningsResponseModel] = None
    # Add fields for retrieved context, errors, etc.
    status: str = "Processed"
    message: Optional[str] = None

# Language Agent Request Model (matches Language Agent's SynthesisRequest)
class SynthesisRequestModel(BaseModel):
    original_query: str
    risk_analysis: Optional[RiskExposureResponseModel] = None
    earnings_analysis: Optional[EarningsResponseModel] = None
    retrieved_context: Optional[List[str]] = None

# Language Agent Response Model (matches Language Agent's SynthesisResponse)
class SynthesisResponseModel(BaseModel):
    narrative: str

# Final Response Model for the Orchestrator endpoint
class FinalNarrativeResponse(BaseModel):
    query: str
    narrative: str
    status: str = "Success"
    error_message: Optional[str] = None # In case synthesis fails

# --- FastAPI App ---
app = FastAPI(
    title="Finance Orchestrator",
    description="Coordinates tasks between various financial agent microservices and synthesizes a final response.",
    version="0.1.2", # Incremented version
)

# --- Helper Functions ---

async def call_analysis_agent_risk(assets: List[Dict[str, Any]], aum: float, criteria: Dict[str, str], client: httpx.AsyncClient) -> Optional[RiskExposureResponseModel]:
    """Calls the Analysis Agent for risk exposure."""
    if not assets or not aum:
        logger.warning("Cannot calculate risk exposure without portfolio assets and AUM.")
        return None
    try:
        url = f"{ANALYSIS_AGENT_URL}/analyze/risk-exposure"
        payload = {
            "assets": assets,
            "total_aum": aum,
            "filter_criteria": criteria
        }
        logger.info(f"Calling Analysis Agent (Risk) at {url} with criteria: {criteria}")
        response = await client.post(url, json=payload, timeout=15.0)
        response.raise_for_status()
        logger.info("Received risk analysis response from Analysis Agent.")
        # Parse response using the defined Pydantic model
        return RiskExposureResponseModel(**response.json())
    except httpx.RequestError as exc:
        logger.error(f"HTTP error calling Analysis Agent (Risk): {exc}")
        return None
    except Exception as e:
        logger.error(f"Error processing risk analysis response: {e}", exc_info=True)
        return None


async def call_analysis_agent_earnings(tickers: List[str], estimates: Dict[str, float], client: httpx.AsyncClient) -> Optional[EarningsResponseModel]:
    """Calls the Analysis Agent for earnings data."""
    if not tickers:
        return None
    try:
        url = f"{ANALYSIS_AGENT_URL}/analyze/earnings"
        payload = {
            "tickers": tickers,
            "estimates": estimates
        }
        logger.info(f"Calling Analysis Agent (Earnings) at {url} for tickers: {tickers}")
        response = await client.post(url, json=payload, timeout=15.0)
        response.raise_for_status()
        logger.info("Received earnings analysis response from Analysis Agent.")
         # Parse response using the defined Pydantic model
        return EarningsResponseModel(**response.json())
    except httpx.RequestError as exc:
        logger.error(f"HTTP error calling Analysis Agent (Earnings): {exc}")
        return None
    except Exception as e:
        logger.error(f"Error processing earnings analysis response: {e}", exc_info=True)
        return None

async def call_retriever_agent_search(query_text: str, top_k: int, client: httpx.AsyncClient) -> Optional[RetrieverSearchResponseModel]:
    """Calls the Retriever Agent to search for relevant documents."""
    try:
        url = f"{RETRIEVER_AGENT_URL}/search"
        payload = RetrieverSearchQueryModel(query=query_text, top_k=top_k).model_dump()
        logger.info(f"Calling Retriever Agent at {url} with query: '{query_text[:50]}...'")
        response = await client.post(url, json=payload, timeout=20.0)
        response.raise_for_status()
        logger.info("Received search results from Retriever Agent.")
        return RetrieverSearchResponseModel(**response.json())
    except httpx.RequestError as exc:
        logger.error(f"HTTP error calling Retriever Agent: {exc}")
        return None
    except Exception as e:
        logger.error(f"Error processing retriever search response: {e}", exc_info=True)
        return None

async def call_language_agent_synthesize(payload: SynthesisRequestModel, client: httpx.AsyncClient) -> Optional[SynthesisResponseModel]:
    """Calls the Language Agent to synthesize a narrative."""
    try:
        url = f"{LANGUAGE_AGENT_URL}/synthesize-brief"
        logger.info(f"Calling Language Agent at {url} to synthesize brief.")
        payload_dict = payload.model_dump(exclude_unset=True)
        response = await client.post(url, json=payload_dict, timeout=30.0)
        response.raise_for_status()
        logger.info("Received synthesis response from Language Agent.")
        return SynthesisResponseModel(**response.json())
    except httpx.RequestError as exc:
        logger.error(f"HTTP error calling Language Agent: {exc}")
        return None
    except Exception as e:
        logger.error(f"Error processing synthesis response: {e}", exc_info=True)
        return None

# --- Enhanced Query Parsing using spaCy ---

# Define known sectors and regions for better filter extraction
KNOWN_SECTORS = {
    "tech": "Tech", "technology": "Tech",
    "finance": "Finance", "financials": "Finance",
    "health": "Healthcare", "healthcare": "Healthcare",
    "energy": "Energy",
    "industrial": "Industrials", "industrials": "Industrials",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "utilities": "Utilities",
    "materials": "Materials",
    "real estate": "Real Estate",
    "communication services": "Communication Services", "telecom": "Communication Services"
}

KNOWN_REGIONS = {
    "asia": "Asia",
    "north america": "North America", "na": "North America",
    "northamerica": "North America", # Added just in case
    "europe": "Europe", "eu": "Europe",
    "usa": "USA", "united states": "USA",
    "china": "China",
    "japan": "Japan",
    "india": "India"
    # Add more as needed
}


def parse_query(query: str) -> Dict[str, Any]:
    """
    Parses the user query using spaCy for basic intent and entity extraction.
    Identifies tickers (simple pattern), keywords for intent, and basic filters.
    """
    logger.info(f"Parsing query with spaCy: '{query}'")
    parsed = {
        "intent": "unknown",
        "tickers": [],
        "risk_filters": {},
        "earnings_estimates": {}, # Still placeholder
        "needs_retrieval": False,
        "retrieval_query_text": query # Default retrieval query
    }

    query_lower = query.lower() # For case-insensitive matching of keywords

    if not nlp:
        logger.warning("spaCy model not loaded. Falling back to basic keyword matching.")
        # Minimal fallback (can be expanded)
        if "risk" in query_lower or "exposure" in query_lower:
            parsed["intent"] = "get_risk_exposure"
        elif "earnings" in query_lower or "surprise" in query_lower:
            parsed["intent"] = "get_earnings_analysis"
        
        potential_tickers = re.findall(r'\\b[A-Z]{3,5}\\b', query) # Regex for tickers
        parsed["tickers"] = list(set(potential_tickers))

        # Basic filter extraction for fallback
        # This is very rudimentary and won't handle multi-word sectors/regions well
        tokens = query_lower.split()
        for token in tokens:
            if token in KNOWN_SECTORS:
                parsed["risk_filters"]["sector"] = KNOWN_SECTORS[token]
            if token in KNOWN_REGIONS:
                parsed["risk_filters"]["region"] = KNOWN_REGIONS[token]
        
        # Fallback needs_retrieval logic
        if any(kw in query_lower for kw in ["brief", "news", "filings", "what happened", "context"]):
             parsed["needs_retrieval"] = True
        return parsed

    doc = nlp(query)
    extracted_tickers = set()

    # 1. Extract Tickers (using spaCy Entities and Regex fallback)
    for ent in doc.ents:
        if ent.label_ == "ORG" and re.fullmatch(r'[A-Z]{3,5}', ent.text):
            extracted_tickers.add(ent.text)
    # Supplement with Regex for any uppercase words that might be tickers
    potential_tickers_re = re.findall(r'\\b[A-Z]{3,5}\\b', query)
    extracted_tickers.update(potential_tickers_re)
    parsed["tickers"] = list(extracted_tickers)

    # 2. Extract Risk Filters (Sectors and Regions)
    # Iterate through n-grams for better matching of multi-word filters
    logger.debug("--- Starting N-gram Filter Extraction ---") # DEBUG
    for n in range(3, 0, -1): # Check for 3-grams, then 2-grams, then 1-grams
        found_sector_for_n = False
        found_region_for_n = False
        if "sector" in parsed["risk_filters"] and "region" in parsed["risk_filters"]: # Optimization
            logger.debug("Both sector and region already found, skipping further n-gram checks.") # DEBUG
            break

        logger.debug(f"Checking {n}-grams:") # DEBUG
        for i in range(len(doc) - n + 1):
            ngram_doc = doc[i : i + n]
            ngram_text = ngram_doc.text.lower().strip() # Ensure lowercasing and strip whitespace
            logger.debug(f"  N-gram text: '{ngram_text}' (Original: '{ngram_doc.text}')") # DEBUG

            # Check and add sector if found and not already present
            if not found_sector_for_n and "sector" not in parsed["risk_filters"]:
                if ngram_text in KNOWN_SECTORS:
                    parsed["risk_filters"]["sector"] = KNOWN_SECTORS[ngram_text]
                    logger.info(f"Found sector: '{KNOWN_SECTORS[ngram_text]}' from n-gram: '{ngram_text}'") # INFO
                    found_sector_for_n = True # Mark as found for this n-gram size

            # Check and add region if found and not already present
            if not found_region_for_n and "region" not in parsed["risk_filters"]:
                if ngram_text in KNOWN_REGIONS:
                    parsed["risk_filters"]["region"] = KNOWN_REGIONS[ngram_text]
                    logger.info(f"Found region: '{KNOWN_REGIONS[ngram_text]}' from n-gram: '{ngram_text}'") # INFO
                    found_region_for_n = True # Mark as found for this n-gram size
            
            if found_sector_for_n and found_region_for_n: # If both found for this n-gram size, no need to check other i for this n
                break 
        if "sector" in parsed["risk_filters"] and "region" in parsed["risk_filters"]: # If both found, can break outer loop too
             logger.debug("Both sector and region populated by n-grams, breaking n-gram loop.") # DEBUG
             break
    logger.debug("--- Finished N-gram Filter Extraction ---") # DEBUG

    # Check single tokens again if n-grams didn't catch them
    # This helps if a region/sector is mentioned standalone and an n-gram including it wasn't in KNOWN_X
    if "sector" not in parsed["risk_filters"]:
        logger.debug("Sector not found by n-grams, checking single tokens...") # DEBUG
        for token in doc:
            token_text_lower = token.text.lower()
            logger.debug(f"  Single token (sector check): '{token_text_lower}'") # DEBUG
            if token_text_lower in KNOWN_SECTORS:
                parsed["risk_filters"]["sector"] = KNOWN_SECTORS[token_text_lower]
                logger.info(f"Found sector: '{KNOWN_SECTORS[token_text_lower]}' from single token: '{token_text_lower}'") # INFO
                break # Found one, stop
    if "region" not in parsed["risk_filters"]:
        logger.debug("Region not found by n-grams, checking single tokens...") # DEBUG
        for token in doc:
            token_text_lower = token.text.lower()
            logger.debug(f"  Single token (region check): '{token_text_lower}'") # DEBUG
            if token_text_lower in KNOWN_REGIONS:
                parsed["risk_filters"]["region"] = KNOWN_REGIONS[token_text_lower]
                logger.info(f"Found region: '{KNOWN_REGIONS[token_text_lower]}' from single token: '{token_text_lower}'") # INFO
                break # Found one, stop


    # 3. Determine Intent based on keywords (can be more sophisticated)
    # Prioritize more specific intents if multiple keywords are present
    intent_keywords = {
        "get_earnings_analysis": ["earnings", "surprise", "eps", "estimates", "performance"],
        "get_risk_exposure": ["risk", "exposure", "allocation"],
        "get_market_brief": ["brief", "overview", "summary", "market update"],
        "fetch_news": ["news", "headlines", "latest updates", "update on", "what's new"],
        "fetch_filings": ["filing", "sec document", "10-k", "10-q"]
    }

    # Detect primary intents
    is_earnings_query = any(kw in query_lower for kw in intent_keywords["get_earnings_analysis"])
    is_risk_query = any(kw in query_lower for kw in intent_keywords["get_risk_exposure"])
    is_news_query = any(kw in query_lower for kw in intent_keywords["fetch_news"])
    is_brief_query = any(kw in query_lower for kw in intent_keywords["get_market_brief"])
    is_filings_query = any(kw in query_lower for kw in intent_keywords["fetch_filings"])

    if is_earnings_query and is_risk_query and is_news_query:
        parsed["intent"] = "get_risk_earnings_and_news"
        parsed["needs_retrieval"] = True
    elif is_earnings_query and is_risk_query:
        parsed["intent"] = "get_risk_and_earnings"
    elif is_earnings_query and is_news_query:
        parsed["intent"] = "get_earnings_and_news"
        parsed["needs_retrieval"] = True
    elif is_risk_query and is_news_query:
        parsed["intent"] = "get_risk_and_news"
        parsed["needs_retrieval"] = True
    elif is_earnings_query:
        parsed["intent"] = "get_earnings_analysis"
    elif is_risk_query:
        parsed["intent"] = "get_risk_exposure"
    elif is_news_query:
        parsed["intent"] = "fetch_news"
        parsed["needs_retrieval"] = True
        if not parsed["tickers"]: # Adjust retrieval query for general news
            parsed["retrieval_query_text"] = "latest financial market news"
        else:
            parsed["retrieval_query_text"] = f"latest news for {', '.join(parsed['tickers'])}"
    elif is_brief_query:
        parsed["intent"] = "get_market_brief"
        parsed["needs_retrieval"] = True
    elif is_filings_query:
        parsed["intent"] = "fetch_filings"
        parsed["needs_retrieval"] = True
        if not parsed["tickers"]:
            parsed["retrieval_query_text"] = "recent SEC filings"
        else:
            parsed["retrieval_query_text"] = f"recent SEC filings for {', '.join(parsed['tickers'])}"
    
    if parsed["intent"] == "unknown":
        if parsed["tickers"] or parsed["risk_filters"]:
            parsed["intent"] = "general_analysis"
            # General analysis might need retrieval depending on keywords
            if any(kw in query_lower for kw in ["details", "explain", "context", "more info", "what about"]):
                parsed["needs_retrieval"] = True
        else: 
            parsed["intent"] = "general_query"
            parsed["needs_retrieval"] = True 

    # If earnings are asked for, tickers become more important
    if "earnings" in parsed["intent"]: # Check if 'earnings' is part of the intent string
        if not parsed["tickers"] and "portfolio" in query_lower:
             logger.info("Earnings query mentions portfolio but no specific tickers identified by parser.")
             # Future: could set a flag to use all portfolio tickers.

    # 4. Refine Retrieval Need further
    # Already set for news, brief, filings, some general cases.
    # For analysis intents, if the query has qualitative aspects or asks for "surprise".
    if parsed["intent"] not in ["fetch_news", "get_market_brief", "fetch_filings"] and not parsed["needs_retrieval"]:
        qualitative_keywords = ["outlook", "guidance", "sentiment", "why", "explain", "details", "impact of", "comment on", "thoughts on"]
        if any(kw in query_lower for kw in qualitative_keywords):
            logger.info(f"Flagging for retrieval for intent '{parsed['intent']}' due to qualitative keywords.")
            parsed["needs_retrieval"] = True
        
        if "surprise" in query_lower and "earnings" in parsed["intent"]: # e.g. get_earnings_analysis, get_risk_and_earnings
            logger.info("Flagging for retrieval due to 'surprise' in earnings query.")
            parsed["needs_retrieval"] = True
            
    # Ensure retrieval_query_text is reasonable if needs_retrieval is true but intent isn't news-specific
    if parsed["needs_retrieval"] and parsed["intent"] not in ["fetch_news", "fetch_filings", "get_market_brief"]:
        if parsed["retrieval_query_text"] == query: # If not already specialized by news/filings logic
            # Potentially refine retrieval_query_text for other intents, 
            # e.g., focus on parts of the query relevant for context.
            # For now, using the full query is the default.
            pass


    # Final check: if the query is just tickers and "risk" or "earnings", maybe no retrieval
    # This logic is complex and might be too aggressive in turning off retrieval. Context is often good.
    # Consider removing or simplifying if it causes issues.
    if parsed["intent"] in ["get_risk_exposure", "get_earnings_analysis"] and parsed["tickers"] and not parsed["risk_filters"] and parsed["needs_retrieval"]:
        is_simple_request = True
        simplified_query = query_lower
        # Remove intent keywords and tickers
        for intent_kw_list in intent_keywords.values():
            for kw in intent_kw_list:
                simplified_query = simplified_query.replace(kw, "")
        for ticker_text in parsed["tickers"]: # Ensure using the actual ticker text
            simplified_query = simplified_query.replace(ticker_text.lower(), "")
        
        # If what's left is mostly non-alphabetic (e.g. spaces, question marks)
        if not any(c.isalpha() for c in simplified_query.strip()):
             # And no other strong retrieval keywords are present
             if not any(kw in query_lower for kw in ["brief", "overview", "context", "outlook", "guidance", "surprise", "news", "headlines", "filings", "details", "explain"]):
                logger.info(f"Potentially simple request for {parsed['intent']} on tickers. Overriding to needs_retrieval=False.")
                # parsed["needs_retrieval"] = False # Keeping this commented for now, as context is generally useful.

    logger.info(f"Final parsed query: {parsed}")
    return parsed


# --- API Endpoints ---

@app.post("/process-query", response_model=FinalNarrativeResponse)
async def process_user_query(request: UserQueryRequest):
    """
    Receives a user query, parses it, calls relevant agents (analysis, retrieval),
    calls the language agent for synthesis, and returns the final narrative.
    """
    logger.info(f"Received query: '{request.query}'")

    parsed_info = parse_query(request.query)
    intent = parsed_info.get("intent", "unknown") # Get the determined intent

    # Check for portfolio data if risk analysis is part of the intent
    if "risk" in intent:
        if request.portfolio_assets is None or request.portfolio_aum is None:
            logger.error("Risk analysis requires portfolio assets and AUM.")
            return FinalNarrativeResponse(
                query=request.query,
                narrative="I need your portfolio details (assets and total value) to calculate risk exposure.",
                status="Failed",
                error_message="Portfolio data missing for risk analysis."
            )

    risk_result: Optional[RiskExposureResponseModel] = None
    earnings_result: Optional[EarningsResponseModel] = None
    retrieved_docs_content: Optional[List[str]] = None 

    async with httpx.AsyncClient() as client:
        tasks = []

        # Schedule Risk Analysis Call
        if "risk" in intent and parsed_info.get("risk_filters"):
            logger.info(f"Scheduling risk analysis based on intent: {intent} and filters: {parsed_info['risk_filters']}")
            tasks.append(call_analysis_agent_risk(
                request.portfolio_assets, # Already checked for None if 'risk' in intent
                request.portfolio_aum,    # Already checked for None if 'risk' in intent
                parsed_info["risk_filters"],
                client
            ))
        else:
            tasks.append(asyncio.sleep(0, result=None)) # Risk task placeholder

        # Schedule Earnings Analysis Call
        if "earnings" in intent and parsed_info.get("tickers"):
            logger.info(f"Scheduling earnings analysis based on intent: {intent} and tickers: {parsed_info['tickers']}")
            tasks.append(call_analysis_agent_earnings(
                parsed_info["tickers"],
                parsed_info.get("earnings_estimates", {}),
                client
            ))
        else:
            tasks.append(asyncio.sleep(0, result=None)) # Earnings task placeholder

        # Schedule Retriever Call
        if parsed_info.get("needs_retrieval"):
            logger.info(f"Scheduling retriever call for query: '{parsed_info['retrieval_query_text']}'")
            tasks.append(call_retriever_agent_search(
                parsed_info["retrieval_query_text"],
                top_k=3, # Default top_k, can be made configurable
                client=client
            ))
        else:
            tasks.append(asyncio.sleep(0, result=None)) # Retriever task placeholder

        # Execute calls concurrently
        logger.debug(f"Gathering {len(tasks)} tasks...")
        results = await asyncio.gather(*tasks)
        logger.debug(f"Tasks gathered. Results: {results}")


        risk_result = results[0]
        earnings_result = results[1]
        retriever_response = results[2] # This is RetrieverSearchResponseModel or None

        if retriever_response and isinstance(retriever_response, RetrieverSearchResponseModel) and retriever_response.results:
            retrieved_docs_content = [doc.content for doc in retriever_response.results]
            logger.info(f"Retrieved {len(retrieved_docs_content)} document snippets for context.")
        elif parsed_info.get("needs_retrieval"):
            logger.warning("Retrieval was needed, but no documents were returned by the retriever agent or an error occurred.")


    # 3. Call Language Agent for Synthesis
    if not risk_result and not earnings_result and not retrieved_docs_content:
        logger.warning("No data gathered from downstream agents to synthesize.")
        # Return a specific narrative if no data could be processed
        return FinalNarrativeResponse(
            query=request.query,
            narrative="I couldn't retrieve or analyze the requested information. Please try rephrasing your query or check if the backend services are running.",
            status="Failed",
            error_message="No data from analysis/retrieval agents."
        )

    synthesis_payload = SynthesisRequestModel(
        original_query=request.query,
        risk_analysis=risk_result,
        earnings_analysis=earnings_result,
        retrieved_context=retrieved_docs_content # Pass retrieved docs here
    )

    async with httpx.AsyncClient() as client:
        synthesis_result = await call_language_agent_synthesize(synthesis_payload, client)

    # 4. Format Final Response
    if synthesis_result:
        return FinalNarrativeResponse(
            query=request.query,
            narrative=synthesis_result.narrative,
            status="Success"
        )
    else:
        logger.error("Failed to synthesize narrative using Language Agent.")
        # Return structured data as fallback? Or just an error message?
        # For now, return an error narrative.
        return FinalNarrativeResponse(
            query=request.query,
            narrative="Sorry, I gathered the information but encountered an error while trying to summarize it.",
            status="Failed",
            error_message="Language agent synthesis failed."
        )


@app.get("/health")
async def health_check():
    """Basic health check endpoint. Checks connectivity to downstream services."""
    dependencies = {}
    async with httpx.AsyncClient() as client:
        # Check Analysis Agent
        try:
            response = await client.get(f"{ANALYSIS_AGENT_URL}/health", timeout=2.0)
            dependencies["analysis_agent"] = "ok" if response.status_code == 200 else "error"
        except Exception:
            dependencies["analysis_agent"] = "unreachable"

        # Check Language Agent
        try:
            response = await client.get(f"{LANGUAGE_AGENT_URL}/health", timeout=2.0)
            dependencies["language_agent"] = "ok" if response.status_code == 200 else "error"
        except Exception:
            dependencies["language_agent"] = "unreachable"

        # Add checks for other agents here later (Retriever, etc.)

    all_ok = all(status == "ok" for status in dependencies.values())
    return {"status": "ok" if all_ok else "degraded", "dependencies": dependencies}


# --- Running the App (for local development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Orchestrator service...")
    # Run on port 8000 (main entry point)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info") 