# Multi-Agent Financial Assistant

This project implements a multi-source, multi-agent financial assistant using FastAPI microservices for agent orchestration and a Streamlit application for the user interface.

## Features

*   **Multi-Source Data Ingestion:** Fetches data from APIs (`yfinance`) and scrapes financial filings (`sec-edgar-downloader`).
*   **Retrieval-Augmented Generation (RAG):** Indexes text content from filings into a local FAISS vector store using `SentenceTransformerEmbeddings` for context-aware responses.
*   **Multi-Agent Architecture:** Specialized agents handle tasks like data fetching, retrieval, analysis, language generation, and voice interaction. Agents communicate via REST APIs.
*   **Natural Language Understanding (NLU):** Uses `spaCy` for basic intent recognition and entity extraction (e.g., tickers, risk filters) in the Orchestrator.
*   **Voice Interface (STT only):** Supports spoken queries (STT via Groq Cloud `whisper-large-v3` API). TTS functionality has been temporarily removed.
*   **Microservice-Based:** Agents run as independent FastAPI services, orchestrated centrally.
*   **Streamlit UI:** Provides a web-based interface for text and voice input.
*   **AI Usage Logging:** Basic logging implemented for Groq LLM and STT API usage.

## Architecture Overview

The system consists of several FastAPI microservices (Agents) coordinated by an Orchestrator, with a Streamlit frontend.

1.  **Streamlit App (`streamlit_app/`):** User interface. Captures text/audio queries. Sends requests to Orchestrator and Voice Agent. Displays results and plays audio responses.
2.  **Orchestrator (`orchestrator/`):** Main entry point. Parses user query (`spaCy`), determines intent, calls relevant agents (Analysis, Retriever) concurrently, gathers results, calls Language Agent for synthesis, and returns the final narrative to the UI.
3.  **API Agent (`agents/api_agent/`):** Fetches stock information using `yfinance`.
4.  **Scraping Agent (`agents/scraping_agent/`):** Downloads SEC filings using `sec-edgar-downloader`. Triggers the Retriever Agent to index new filings after download.
5.  **Retriever Agent (`agents/retriever_agent/`):** Manages the FAISS vector store (`data/vector_store/`). Indexes downloaded filings (`**/full-submission.txt`) and performs semantic search based on queries from the Orchestrator.
6.  **Analysis Agent (`agents/analysis_agent/`):** Performs calculations like portfolio risk exposure based on filters and earnings surprise based on data from the API Agent.
7.  **Language Agent (`agents/language_agent/`):** Uses a LangChain chain with Groq Cloud (e.g., `llama-3.3-70b-versatile` model) to synthesize natural language narratives based on the original query and data provided by the Orchestrator.
8.  **Voice Agent (`agents/voice_agent/`):** Handles STT using Groq Cloud's `whisper-large-v3` API. (TTS functionality removed for now).

```mermaid
flowchart TD
    subgraph "Layer 1: User Interface"
        UI["Streamlit UI<br>(Streamlit)"]:::frontend
    end

    subgraph "Layer 2: Orchestration"
        ORC["Orchestrator<br>(FastAPI + spaCy)"]:::orchestrator
    end

    subgraph "Layer 3: Domain Agents"
        API["API Agent<br>(FastAPI + yfinance)"]:::agent
        SCR["Scraping Agent<br>(FastAPI + sec-edgar-downloader)"]:::agent
        RET["Retriever Agent<br>(FastAPI + FAISS & SentenceTransformers)"]:::agent
        ANA["Analysis Agent<br>(FastAPI + financial calculations)"]:::agent
        LANG["Language Agent<br>(FastAPI + LangChain & Groq SDK)"]:::agent
        VOICE["Voice Agent<br>(FastAPI + Groq Whisper STT)"]:::agent
    end

    subgraph "Layer 4: Data Stores & External Services"
        FAISS["FAISS Vector Store"]:::datastore
        YFINANCE["yfinance API"]:::external
        EDGAR["SEC EDGAR"]:::external
        GROQ["Groq Cloud<br>(LLM & STT)"]:::external
    end

    subgraph "Project Configuration"
        REQ["requirements.txt"]:::external
        DOC["README.md"]:::external
        GIT[".gitignore"]:::external
    end

    UI -->|HTTP POST (JSON/audio)| ORC
    UI -->|audio stream| VOICE
    ORC -->|REST JSON| API
    ORC -->|REST JSON| SCR
    SCR -->|indexes docs| RET
    ORC -->|REST JSON| RET
    ORC -->|REST JSON| ANA
    ORC -->|REST JSON| LANG
    LANG -->|JSON response| ORC
    VOICE -->|transcript| ORC

    API -->|market data| YFINANCE
    SCR -->|filings| EDGAR
    RET -->|vector query| FAISS
    LANG -->|LLM & STT| GROQ
    VOICE -->|STT| GROQ

    click UI "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/streamlit_app/app.py"
    click ORC "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/orchestrator/main.py"
    click API "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/agents/api_agent/main.py"
    click SCR "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/agents/scraping_agent/main.py"
    click RET "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/agents/retriever_agent/main.py"
    click ANA "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/agents/analysis_agent/main.py"
    click LANG "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/agents/language_agent/main.py"
    click VOICE "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/agents/voice_agent/main.py"
    click REQ "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/requirements.txt"
    click DOC "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/README.md"
    click GIT "https://github.com/flameingmoy/agentic_finance_assistant/blob/master/.gitignore"

    classDef frontend fill:#add8e6,stroke:#333,shape:rect
    classDef orchestrator fill:#90ee90,stroke:#333,shape:roundrect
    classDef agent fill:#ffa500,stroke:#333,shape:ellipse
    classDef datastore fill:#ffff99,stroke:#333,shape:cylinder
    classDef external fill:#d3d3d3,stroke:#333,shape:rect
```

## Setup and Usage

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Flameingmoy/Agentic_Finance_Assistant
    cd finance-assistant
    ```

2.  **Create and activate a Conda environment:**
    ```bash
    # Choose a Python version (e.g., 3.11) and environment name
    conda create -n finance_assistant_env python=3.11
    conda activate finance_assistant_env
    ```

3.  **Install Python dependencies using pip:**
    ```bash
    pip install -r requirements.txt
    ```
    *   **FAISS GPU Note:** The `requirements.txt` lists `faiss-gpu-cu12`. Installing this requires a compatible NVIDIA GPU and the corresponding CUDA Toolkit (version 12.x) installed on your system.

4.  **Download spaCy model:**
    ```bash
    python -m spacy download en_core_web_sm
    ```

5.  **Create `.env` file:**
    *   Copy the `.env.example` file (if provided) or create a new `.env` file in the project root (`finance-assistant/`).
    *   Add the required environment variables:
        ```dotenv
        # Required for Language Agent and Voice Agent (STT)
        GROQ_API_KEY=\"your_groq_api_key\"

        # Required for SEC scraping (use your own email)
        COMPANY_EMAIL=\"your_email@example.com\"

        # Optional: Groq Model Names (if different from defaults)
        # GROQ_LLM_MODEL_NAME=\"llama-3.3-70b-versatile\"
        # GROQ_STT_MODEL_NAME=\"whisper-large-v3\"

        # Optional: Override default agent URLs if needed
        # ORCHESTRATOR_URL=\"http://localhost:8000\"
        # API_AGENT_URL=\"http://localhost:8001\"
        # SCRAPING_AGENT_URL=\"http://localhost:8002\"
        # RETRIEVER_AGENT_URL=\"http://localhost:8003\"
        # ANALYSIS_AGENT_URL=\"http://localhost:8004\"
        # LANGUAGE_AGENT_URL=\"http://localhost:8005\"
        # VOICE_AGENT_URL=\"http://localhost:8006\"

        # Optional: Base path for data storage (defaults to ./data)
        # FINANCE_ASSISTANT_DATA_PATH=\"/path/to/your/data/directory\"
        ```

6.  **Run the services:**
    *   Open multiple terminals, activate the Conda environment in each (`conda activate finance_assistant_env`).
    *   Run each agent and the orchestrator using `uvicorn`:
        ```bash
        # Terminal 1: Orchestrator
        python finance_assistant/orchestrator/main.py
        # Terminal 2: API Agent
        python finance_assistant/agents/api_agent/main.py
        # Terminal 3: Scraping Agent
        python finance_assistant/agents/scraping_agent/main.py
        # Terminal 4: Retriever Agent
        python finance_assistant/agents/retriever_agent/main.py
        # Terminal 5: Analysis Agent
        python finance_assistant/agents/analysis_agent/main.py
        # Terminal 6: Language Agent
        python finance_assistant/agents/language_agent/main.py
        # Terminal 7: Voice Agent
        python finance_assistant/agents/voice_agent/main.py
        ```
    *   Run the Streamlit UI:
        ```bash
        # Terminal 8: Streamlit App
        streamlit run finance_assistant/streamlit_app/app.py
        ```
    *   Access the Streamlit UI in your browser (usually at `http://localhost:8501`).

## Agent Roles

*   **API Agent:** Fetches real-time and historical market data via `yfinance`.
*   **Scraping Agent:** Downloads financial filings (e.g., SEC EDGAR). Triggers indexing in the Retriever Agent upon completion.
*   **Retriever Agent:** Manages the FAISS vector store, indexes filings, and performs semantic search.
*   **Analysis Agent:** Performs financial calculations (risk exposure, earnings surprise).
*   **Language Agent:** Synthesizes natural language responses using Groq Cloud LLM (e.g., `llama-3.3-70b-versatile`).
*   **Voice Agent:** Handles Speech-to-Text using Groq Cloud `whisper-large-v3` API. (TTS functionality removed).
*   **Orchestrator:** Parses user queries (`spaCy`), routes requests to appropriate agents, and coordinates the final response synthesis.

## Technology Stack

*   Python
*   FastAPI
*   Streamlit
*   LangChain (for LLM interaction, RAG components)
*   Groq SDK (`groq`)
*   spaCy (for NLU in Orchestrator)
*   FAISS (`faiss-gpu-cu12`) - *Specific build for CUDA 12.x, listed in requirements.txt*
*   Sentence Transformers
*   `yfinance`
*   `sec-edgar-downloader`
*   Pandas
*   Uvicorn
*   `python-dotenv`
*   `httpx`
*   `streamlit-audio-recorder`
*   `soundfile` 
