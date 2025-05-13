# finance_assistant/streamlit_app/app.py

import streamlit as st
import requests
import os
import json
import pandas as pd # For potentially displaying portfolio later
import io

# Import the audio recorder component
from audio_recorder_streamlit import audio_recorder

# --- Configuration ---
# Get agent URLs from environment variables or use defaults
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
VOICE_AGENT_URL = os.getenv("VOICE_AGENT_URL", "http://localhost:8006")

# --- Dummy Portfolio Data (Replace with actual data loading/management) ---
# In a real app, this would come from a database, user login, etc.
DEFAULT_PORTFOLIO = {
    "assets": [
        {"symbol": "AAPL", "value": 18000, "region": "North America", "sector": "Tech"},
        {"symbol": "TSMC", "value": 22000, "region": "Asia", "sector": "Tech"},
        {"symbol": "005930.KS", "value": 15000, "region": "Asia", "sector": "Tech"}, # Samsung
        {"symbol": "JPM", "value": 45000, "region": "North America", "sector": "Finance"},
        {"symbol": "NVDA", "value": 30000, "region": "North America", "sector": "Tech"}
    ],
    "total_aum": 130000 # Sum of values above
}

# --- API Client Functions ---

def call_orchestrator(query: str, portfolio: dict) -> dict | None:
    """Sends query and portfolio data to the Orchestrator."""
    url = f"{ORCHESTRATOR_URL}/process-query"
    payload = {
        "query": query,
        "portfolio_assets": portfolio.get("assets", []),
        "portfolio_aum": portfolio.get("total_aum", 0)
    }
    try:
        with st.spinner("Thinking..."): # Show spinner during processing
            response = requests.post(url, json=payload, timeout=60) # Longer timeout for full chain
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting Orchestrator: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Received invalid response from Orchestrator.")
        return None

# def call_tts(text: str) -> bytes | None: # Removed TTS call
#     """Sends text to the Voice Agent for TTS synthesis."""
#     url = f"{VOICE_AGENT_URL}/tts"
#     payload = {"text": text}
#     try:
#         with st.spinner("Generating audio..."):
#             response = requests.post(url, json=payload, timeout=30, stream=True)
#             response.raise_for_status()
#             if 'audio/wav' in response.headers.get('Content-Type', ''):
#                 return response.content
#             else:
#                 try:
#                     error_data = response.json()
#                     st.error(f"TTS Agent returned an error: {error_data}")
#                 except json.JSONDecodeError:
#                      st.error(f"Received non-audio response from TTS Agent: {response.text}")
#                 return None
#     except requests.exceptions.RequestException as e:
#         st.error(f"Error contacting TTS Agent: {e}")
#         return None

def call_stt(audio_bytes: bytes) -> dict | None:
    """Sends audio data to the Voice Agent for STT transcription."""
    url = f"{VOICE_AGENT_URL}/stt"
    files = {'audio': ('recorded_audio.wav', audio_bytes, 'audio/wav')}
    try:
        with st.spinner("Transcribing audio..."):
            response = requests.post(url, files=files, timeout=30)
            response.raise_for_status()
            return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error contacting STT Agent: {e}")
        return None
    except json.JSONDecodeError:
        st.error("Received invalid response from STT Agent.")
        return None

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="Finance Assistant")
st.title("📈 Multi-Agent Financial Assistant")

# Initialize session state for the query text if it doesn't exist
if 'user_query' not in st.session_state:
    st.session_state.user_query = ""

# --- Sidebar (Optional: for settings or portfolio view) ---
with st.sidebar:
    st.header("Portfolio Overview")
    # Display portfolio data (read-only for now)
    if DEFAULT_PORTFOLIO["assets"]:
        df = pd.DataFrame(DEFAULT_PORTFOLIO["assets"])
        st.dataframe(df)
        st.metric("Total AUM", f"${DEFAULT_PORTFOLIO['total_aum']:,.2f}")
    else:
        st.write("No portfolio data loaded.")

    st.header("Configuration")
    st.text_input("Orchestrator URL", value=ORCHESTRATOR_URL, disabled=True)
    st.text_input("Voice Agent URL", value=VOICE_AGENT_URL, disabled=True)

# --- Main Interaction Area ---
st.header("Ask a Question")

# Row for audio recorder and text input
col1, col2 = st.columns([1, 4])

with col1:
    st.write("Record Query:")
    # Use the audio recorder component
    audio_bytes = audio_recorder(
        text="", # No button text
        recording_color="#e8b62c",
        neutral_color="#6a6a6a",
        icon_size="2x",
        pause_threshold=2.0, # Seconds of silence to automatically stop
    )

with col2:
    st.write("Or Enter/Edit Query Text:")
    # Text input - uses session state to allow updates from audio recorder
    st.text_input(
        "Query Text",
        key='user_query', # Assign key to access/update value via session_state
        placeholder="e.g., What's my risk exposure in Asia tech?",
        label_visibility="collapsed"
    )

# Handle recorded audio *after* the text input widget is defined
if audio_bytes:
    st.info("Audio recorded. Processing...")
    stt_response = call_stt(audio_bytes)
    if stt_response and stt_response.get("text"):
        # Update the text input field using session_state
        st.session_state.user_query = stt_response["text"]
        st.success("Audio transcribed!")
        st.rerun() # Rerun to update the text input field immediately
    else:
        st.error("Failed to transcribe audio.")

# Submit button placed below input methods
if st.button("Submit Query"):
    query_to_submit = st.session_state.user_query
    if not query_to_submit:
        st.warning("Please enter or record a query.")
    else:
        st.info(f"Processing query: {query_to_submit}")
        orchestrator_response = call_orchestrator(query_to_submit, DEFAULT_PORTFOLIO)

        if orchestrator_response:
            st.subheader("Response:")
            narrative = orchestrator_response.get("narrative")
            status = orchestrator_response.get("status")
            error_msg = orchestrator_response.get("error_message")

            if status == "Success" and narrative:
                st.markdown(narrative) # Display the text response

                # # Call TTS Agent to get audio # Removed TTS call
                # with st.spinner("Generating audio..."):
                #     audio_bytes_response = call_tts(narrative)
                #
                # if audio_bytes_response:
                #     st.audio(audio_bytes_response, format='audio/wav', start_time=0)
                #     st.success("Audio response ready.")
                # else:
                #     st.warning("Could not generate audio response.")
            elif error_msg:
                 st.error(f"Assistant Error: {error_msg} (Narrative: {narrative or 'N/A'})")
            else:
                 st.error(f"Received an unexpected response structure from the Orchestrator: {orchestrator_response}")
        else:
            # Error handled and displayed by call_orchestrator
            pass # Keep Streamlit running


# --- Optional: Add voice input later ---
# Requires additional libraries like streamlit_webrtc or similar
# st.header("Speak your Question (Future Feature)")
# audio_bytes_input = audio_recorder() # Example placeholder for audio input component
# if audio_bytes_input:
#     st.audio(audio_bytes_input, format="audio/wav")
#     # 1. Send audio_bytes_input to Voice Agent /stt
#     # 2. Get text back
#     # 3. Set the text_input field with the result
#     # 4. Trigger the rest of the process (maybe via button press) 