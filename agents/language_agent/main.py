from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from pathlib import Path

# LangChain components for LLM interaction
# from langchain_openai import ChatOpenAI # Replaced by ChatGroq
from langchain_groq import ChatGroq # Import Groq LLM interface
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Load environment variables (especially OPENAI_API_KEY)
# Assumes .env file is in the project root directory (finance_assistant/../.env)
dotenv_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
logger = logging.getLogger(__name__) # Logger needs to be defined before use
logger.info(f"Attempting to load .env file from: {dotenv_path}")
if not dotenv_path.exists():
    logger.warning(".env file not found at expected location.")

# Dedicated logger for AI model usage
ai_usage_logger = logging.getLogger("ai_usage")
# Configure ai_usage_logger if needed (e.g., to a separate file)
# For now, it will inherit root logger's config

# --- Configuration ---
# LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo") # OpenAI model name
GROQ_LLM_MODEL_NAME = os.getenv("GROQ_LLM_MODEL_NAME", "llama-3.3-70b-versatile") # Use Groq model (Updated)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # No longer needed
GROQ_API_KEY = os.getenv("GROQ_API_KEY") # Groq API Key

# if not OPENAI_API_KEY: # Check for Groq key instead
if not GROQ_API_KEY:
    # For this agent, the API key is essential.
    # logger.error("OPENAI_API_KEY environment variable not set.")
    logger.error("GROQ_API_KEY environment variable not set.")
    # In a real app, consider more robust secret management.
    # raise ValueError("OPENAI_API_KEY environment variable not set.")
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Basic Logging Setup
logging.basicConfig(level=logging.INFO)



# --- Pydantic Models ---

# Re-define input data structures for clarity, matching Orchestrator output
class RiskExposureData(BaseModel):
    criteria: Dict[str, str]
    matched_value: float
    total_aum: float
    exposure_percentage: float

class TickerEarningsDetail(BaseModel):
    symbol: str
    trailing_eps: Optional[float] = None
    forward_eps: Optional[float] = None
    earnings_quarterly_growth: Optional[float] = None
    estimate_eps: Optional[float] = None
    surprise_percentage: Optional[float] = None
    error: Optional[str] = None

class EarningsData(BaseModel):
    results: List[TickerEarningsDetail]

# Input to the Language Agent
class SynthesisRequest(BaseModel):
    original_query: str
    risk_analysis: Optional[RiskExposureData] = None
    earnings_analysis: Optional[EarningsData] = None
    # Add field for retrieved context (e.g., from Retriever Agent) later
    retrieved_context: Optional[List[str]] = None

# Output from the Language Agent
class SynthesisResponse(BaseModel):
    narrative: str


# --- LangChain Setup ---

# Define the prompt template
# This template takes the structured data and guides the LLM
prompt_template = ChatPromptTemplate.from_template(
    """Synthesize a concise and professional morning market brief based on the user's query and the provided data. Address all parts of the query. If some data is missing or unavailable, state that clearly. Do not invent data. Focus on the key information requested.

User Query: "{query}"

Available Data:
Risk Exposure Analysis ({risk_criteria}): {risk_percentage}% (${risk_value:,.0f} out of ${risk_aum:,.0f} AUM)
Earnings Analysis:
{earnings_summary}
{retrieved_context_summary}

Generate the brief:"""
)

# Initialize the LLM using Groq
# llm = ChatOpenAI(model_name=LLM_MODEL_NAME, openai_api_key=OPENAI_API_KEY, temperature=0.2)
llm = ChatGroq(model_name=GROQ_LLM_MODEL_NAME, groq_api_key=GROQ_API_KEY, temperature=0.2)

# Create the generation chain
# Chain: Input -> Prompt -> LLM -> String Output
chain = prompt_template | llm | StrOutputParser()


# --- Helper Function for Formatting ---

def format_earnings_summary(earnings_data: Optional[EarningsData]) -> str:
    """Creates a formatted string summary of earnings data for the prompt."""
    if not earnings_data or not earnings_data.results:
        return "No specific earnings data available or processed."

    summary_lines = []
    processed_tickers = set()
    for item in earnings_data.results:
        if item.symbol in processed_tickers:
            continue # Avoid duplicate entries if any
        processed_tickers.add(item.symbol)

        line = f"- {item.symbol}: "
        parts = []
        if item.error:
            parts.append(f"(Could not fetch data: {item.error})")
        elif item.surprise_percentage is not None:
            if item.surprise_percentage > 0:
                parts.append(f"Beat estimates by {item.surprise_percentage:.1f}%. ")
            elif item.surprise_percentage < 0:
                parts.append(f"Missed estimates by {abs(item.surprise_percentage):.1f}%. ")
            else:
                 parts.append("Met estimates exactly. ")
            # Optionally add actual/estimate figures if needed for context
            # if item.trailing_eps is not None: parts.append(f"Actual EPS: {item.trailing_eps}")
            # if item.estimate_eps is not None: parts.append(f"Estimated EPS: {item.estimate_eps}")
        elif item.trailing_eps is not None:
             parts.append(f"Trailing EPS reported ({item.trailing_eps}), but estimate/surprise unavailable. ")
        elif item.estimate_eps is not None:
             parts.append(f"Estimate provided ({item.estimate_eps}), but actual/surprise unavailable. ")
        else:
            parts.append("No specific earnings surprise or EPS data processed.")
        line += "".join(parts)
        summary_lines.append(line)

    return "\n".join(summary_lines) if summary_lines else "No earnings details processed."

def format_retrieved_context(context: Optional[List[str]]) -> str:
    """Formats retrieved context for the prompt."""
    if not context:
        return "" # Return empty string if no context
    else:
        # Simple concatenation for now
        return "\nRetrieved Context Highlights:\n" + "\n".join([f"- {c}" for c in context])


# --- FastAPI App ---
app = FastAPI(
    title="Finance Language Agent",
    description="Microservice for synthesizing natural language responses using an LLM.",
    version="0.1.0",
)


# --- API Endpoints ---

@app.post("/synthesize-brief", response_model=SynthesisResponse)
async def synthesize_market_brief(request: SynthesisRequest):
    """
    Generates a natural language market brief using an LLM based on structured input data.
    """
    logger.info(f"Received synthesis request for query: '{request.original_query}'")

    # Prepare inputs for the prompt template
    risk_criteria_str = "N/A"
    risk_percentage_val = 0.0
    risk_value_val = 0.0
    risk_aum_val = 0.0
    risk_available = False
    if request.risk_analysis:
        risk_available = True
        risk_criteria_str = ", ".join(f"{k}={v}" for k, v in request.risk_analysis.criteria.items()) or "Overall Portfolio"
        risk_percentage_val = request.risk_analysis.exposure_percentage
        risk_value_val = request.risk_analysis.matched_value
        risk_aum_val = request.risk_analysis.total_aum

    earnings_summary_str = format_earnings_summary(request.earnings_analysis)
    context_summary_str = format_retrieved_context(request.retrieved_context)


    # Construct the input dictionary for the LangChain chain
    chain_input = {
        "query": request.original_query,
        "risk_criteria": risk_criteria_str if risk_available else "(Not Requested or Available)",
        "risk_percentage": risk_percentage_val if risk_available else 0.0,
        "risk_value": risk_value_val if risk_available else 0.0,
        "risk_aum": risk_aum_val if risk_available else 0.0,
        "earnings_summary": earnings_summary_str,
        "retrieved_context_summary": context_summary_str,
    }

    # Refine prompt based on available data to avoid confusing the LLM
    prompt_string = "Synthesize a concise and professional morning market brief based on the user's query and the provided data. Address all parts of the query. If some data is missing or unavailable, state that clearly. Do not invent data. Focus on the key information requested.\n\nUser Query: \"{query}\"\n\nAvailable Data:\n"
    if risk_available:
        prompt_string += "Risk Exposure Analysis ({risk_criteria}): {risk_percentage:.1f}% (${risk_value:,.0f} out of ${risk_aum:,.0f} AUM)\n"
    if request.earnings_analysis and request.earnings_analysis.results:
        prompt_string += "Earnings Analysis:\n{earnings_summary}\n"
    if context_summary_str:
        prompt_string += "{retrieved_context_summary}\n"
    if not risk_available and not (request.earnings_analysis and request.earnings_analysis.results) and not context_summary_str:
        prompt_string += "No specific analytical data was requested or available for this query. Please provide a general response or ask for clarification if needed."
    else:
        prompt_string += "\nGenerate the brief:"

    # Create a dynamic prompt template for this specific request
    dynamic_prompt = ChatPromptTemplate.from_template(prompt_string)
    dynamic_chain = dynamic_prompt | llm | StrOutputParser()

    logger.info(f"Invoking LLM for synthesis. Input keys: {list(chain_input.keys())}")
    
    # AI Usage Logging - Input
    try:
        # Prepare the full prompt string for logging length
        full_prompt_for_log = dynamic_prompt.format_prompt(**chain_input).to_string()
        ai_usage_logger.info(
            f"AI_CALL_START - Model: Groq ({llm.model_name}), Purpose: synthesize_brief, " # Log Groq model
            f"InputLength: {len(full_prompt_for_log)}"
        )
    except Exception as log_ex:
        logger.error(f"Error during pre-LLM AI usage logging: {log_ex}")

    try:
        # Invoke the chain with the input
        narrative = await dynamic_chain.ainvoke(chain_input)
        logger.info("LLM synthesis successful.")
        
        # AI Usage Logging - Output
        ai_usage_logger.info(
            f"AI_CALL_END - Model: Groq ({llm.model_name}), Purpose: synthesize_brief, " # Log Groq model
            f"OutputLength: {len(narrative)}, Status: Success"
        )
        return SynthesisResponse(narrative=narrative)
    except Exception as e:
        logger.error(f"Error during LLM chain invocation: {e}", exc_info=True)
        # AI Usage Logging - Error
        ai_usage_logger.error(
            f"AI_CALL_END - Model: Groq ({llm.model_name}), Purpose: synthesize_brief, Status: Error, ErrorMsg: {str(e)[:100]}" # Log Groq model
        )
        raise HTTPException(status_code=500, detail=f"Error synthesizing response: {e}")


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    # Could potentially add a check to ensure LLM connectivity if feasible/fast
    return {"status": "ok"}

# --- Running the App (for local development) ---
# Note: Ensure OPENAI_API_KEY is set in your environment or .env file
# Updated Note: Ensure GROQ_API_KEY is set in your environment or .env file
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Language Agent service...")
    # Run on port 8005
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info") 