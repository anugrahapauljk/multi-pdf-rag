import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration variables
DATA_DIR = os.getenv("DATA_DIR", "data/PDFs")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "multi_pdf_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
