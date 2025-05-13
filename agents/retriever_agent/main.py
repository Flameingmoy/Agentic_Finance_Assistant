from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
import logging
import os
from pathlib import Path
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, TextLoader
from langchain.docstore.document import Document
import numpy as np
from typing import List, Optional

# --- Configuration ---
# Use the same base data path as the scraping agent
BASE_DATA_PATH = Path(os.getenv("FINANCE_ASSISTANT_DATA_PATH", "../../data")).resolve()
FILINGS_PATH = BASE_DATA_PATH / "filings" / "sec-edgar-filings" # Default path from sec-edgar-downloader
VECTOR_STORE_PATH = BASE_DATA_PATH / "vector_store"
INDEX_FILE = VECTOR_STORE_PATH / "faiss_index.idx"
PKL_FILE = VECTOR_STORE_PATH / "faiss_index.pkl" # Langchain FAISS saves mapping here
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

# Ensure vector store directory exists
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

# Basic Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for embedding model and vector store (load lazily or on startup)
# Use a thread-safe approach if scaling with multiple workers in production
embeddings_model = None
vector_store = None

# --- Pydantic Models ---
class IndexRequest(BaseModel):
    # Specify a subdirectory within FILINGS_PATH to index, e.g., "AAPL/10-K"
    # Or leave empty to attempt indexing all found filings (can be slow)
    sub_directory: Optional[str] = None
    recreate_index: bool = Field(default=False, description="Set to true to delete and rebuild the index from scratch.")

class IndexResponse(BaseModel):
    message: str
    indexed_files_count: Optional[int] = None
    total_chunks_added: Optional[int] = None
    index_location: str = str(VECTOR_STORE_PATH)

class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=4, gt=0, description="Number of relevant chunks to retrieve.")

class SearchResult(BaseModel):
    content: str
    metadata: dict
    score: float # Similarity score

class SearchResponse(BaseModel):
    results: List[SearchResult]

# --- FastAPI App ---
app = FastAPI(
    title="Finance Retriever Agent",
    description="Microservice for indexing documents and retrieving relevant chunks using a vector store (FAISS).",
    version="0.1.0",
)

# --- Helper Functions ---

def initialize_embeddings():
    global embeddings_model
    if embeddings_model is None:
        logger.info(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
        # Use SentenceTransformerEmbeddings, compatible with FAISS
        embeddings_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        logger.info("Embedding model initialized.")
    return embeddings_model

def load_vector_store(recreate: bool = False):
    global vector_store
    initialize_embeddings() # Ensure embeddings are loaded

    if recreate and VECTOR_STORE_PATH.exists():
        logger.warning(f"Recreating index. Deleting existing files in {VECTOR_STORE_PATH}...")
        # Simple deletion - consider more robust cleanup if needed
        for item in VECTOR_STORE_PATH.iterdir():
            if item.is_file() and (item.name.endswith(".idx") or item.name.endswith(".pkl")):
                item.unlink()
        vector_store = None # Reset in-memory variable

    if vector_store is None:
        if INDEX_FILE.exists() and PKL_FILE.exists():
            try:
                logger.info(f"Loading existing FAISS index from {VECTOR_STORE_PATH}")
                # Allow dangerous deserialization for Langchain FAISS pickle loading
                vector_store = FAISS.load_local(
                    folder_path=str(VECTOR_STORE_PATH),
                    embeddings=embeddings_model,
                    index_name="faiss_index", # Base name without extension
                    allow_dangerous_deserialization=True # Required for Langchain FAISS
                )
                logger.info("Existing FAISS index loaded.")
            except Exception as e:
                logger.error(f"Error loading existing FAISS index: {e}. Will try to create a new one.", exc_info=True)
                vector_store = None # Ensure it's None if loading failed
        else:
            logger.info("No existing index found or loading failed. A new index will be created when documents are added.")
            # We don't create an empty index here; it's created when adding documents
            vector_store = None

    # If still None, it means we need to create it when adding docs
    return vector_store


def index_documents_sync(target_path_str: str, recreate: bool):
    """Synchronous function to load, chunk, embed, and index documents."""
    global vector_store
    try:
        target_path = Path(target_path_str)
        if not target_path.exists() or not target_path.is_dir():
            logger.error(f"Target directory for indexing does not exist: {target_path}")
            return 0, 0 # Indicate failure

        logger.info(f"Starting document indexing process for: {target_path}")
        embeddings = initialize_embeddings()
        load_vector_store(recreate=recreate) # Load or prepare for creation

        # Load documents - focus on 'full-submission.txt' or all .txt if it doesn't exist widely
        # Using TextLoader as UnstructuredFileLoader might pull in too much noise from HTML/XBRL without refinement
        # Glob pattern to find relevant text files within the target directory
        glob_pattern = "**/full-submission.txt" # Prioritize the full text dump
        # Alternative: glob_pattern = "**/*.txt" if full-submission isn't reliable

        logger.info(f"Loading documents using pattern: {glob_pattern} from {target_path}")
        loader = DirectoryLoader(
            path=str(target_path),
            glob=glob_pattern,
            loader_cls=TextLoader, # Use TextLoader for potentially cleaner text
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
            show_progress=True,
            use_multithreading=True, # Speed up loading
            silent_errors=True # Skip files that cause errors
        )
        documents = loader.load()

        if not documents:
            logger.warning(f"No documents found or loaded from {target_path} using pattern {glob_pattern}.")
            return 0, 0

        logger.info(f"Loaded {len(documents)} document files.")

        # Chunk documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(chunks)} chunks.")

        if not chunks:
            logger.warning("No chunks were generated from the loaded documents.")
            return len(documents), 0

        # Embed and add to FAISS
        if vector_store is None:
            logger.info("Creating new FAISS index.")
            # Create index from the first batch of chunks
            vector_store = FAISS.from_documents(chunks, embeddings)
            logger.info("New FAISS index created.")
            # Save immediately after creation
            vector_store.save_local(folder_path=str(VECTOR_STORE_PATH), index_name="faiss_index")
            logger.info(f"FAISS index saved to {VECTOR_STORE_PATH}")
        else:
            logger.info("Adding new chunks to existing FAISS index.")
            # Add subsequent chunks to the existing index
            vector_store.add_documents(chunks)
            logger.info("Chunks added to FAISS index.")
            # Save changes
            vector_store.save_local(folder_path=str(VECTOR_STORE_PATH), index_name="faiss_index")
            logger.info(f"FAISS index updated and saved to {VECTOR_STORE_PATH}")

        return len(documents), len(chunks)

    except Exception as e:
        logger.error(f"Error during indexing process for {target_path_str}: {e}", exc_info=True)
        return 0, 0 # Indicate failure

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Load embeddings and vector store on startup."""
    logger.info("Retriever Agent starting up...")
    initialize_embeddings()
    load_vector_store()
    logger.info("Retriever Agent ready.")

@app.post("/index-filings", response_model=IndexResponse)
async def trigger_indexing(request: IndexRequest, background_tasks: BackgroundTasks):
    """
    Triggers the indexing of documents found in a specified subdirectory
    of the main filings path. Runs in the background.
    """
    if request.sub_directory:
        target_path = FILINGS_PATH / request.sub_directory
    else:
        # Index the entire base filings directory if no subdirectory is specified
        target_path = FILINGS_PATH
        logger.warning(f"No sub_directory specified, attempting to index all filings under {FILINGS_PATH}. This might be slow.")

    if not target_path.exists() or not target_path.is_dir():
         raise HTTPException(status_code=404, detail=f"Target directory does not exist: {target_path}")

    logger.info(f"Queueing indexing task for path: {target_path}, Recreate: {request.recreate_index}")

    # Run indexing in the background
    background_tasks.add_task(index_documents_sync, str(target_path), request.recreate_index)

    return IndexResponse(
        message="Document indexing task queued. Check logs for progress and completion.",
        index_location=str(VECTOR_STORE_PATH)
    )

@app.post("/search", response_model=SearchResponse)
async def search_index(query: SearchQuery):
    """
    Performs similarity search on the indexed documents.
    """
    global vector_store
    if vector_store is None:
        # Attempt to load it if it wasn't loaded at startup or after indexing
        load_vector_store()
        if vector_store is None:
             raise HTTPException(status_code=503, detail="Vector store not initialized. Index documents first.")

    logger.info(f"Received search query: \"{query.query}\", top_k={query.top_k}")

    try:
        # Perform similarity search with score
        results_with_scores = vector_store.similarity_search_with_score(
            query.query,
            k=query.top_k
        )

        # Format results
        output_results = [
            SearchResult(
                content=doc.page_content,
                metadata=doc.metadata,
                score=float(score) # Convert numpy float if necessary
            ) for doc, score in results_with_scores
        ]

        logger.info(f"Returning {len(output_results)} search results.")
        return SearchResponse(results=output_results)

    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during search: {e}")


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    # Could add checks for embedding model and vector store readiness
    return {"status": "ok", "vector_store_loaded": vector_store is not None}

# --- Running the App (for local development) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Retriever Agent service...")
    # Run on port 8003
    uvicorn.run(app, host="0.0.0.0", port=8003, log_level="info") 