import os
import sys

# Ensure the project root is in sys.path when running this script directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import (
    DATA_DIR,
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

def ingest_pdfs():
    """
    Loads PDFs from the data directory, splits them into chunks,
    generates embeddings, and stores them in a local ChromaDB instance.
    """
    print(f"Loading PDFs from '{DATA_DIR}'...")
    
    # 1. Load PDFs
    # PyPDFDirectoryLoader extracts source filename and page metadata by default
    loader = PyPDFDirectoryLoader(DATA_DIR, silent_errors=True)
    documents = loader.load()
    
    if not documents:
        print(f"No documents found in {DATA_DIR}. Please add some PDFs.")
        return

    print(f"Loaded {len(documents)} pages.")

    # 2. Split documents
    print(f"Splitting documents into chunks of size {CHUNK_SIZE} with overlap {CHUNK_OVERLAP}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    # 3. Generate Embeddings & Store in ChromaDB
    print(f"Initializing Hugging Face embedding model: '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Storing chunks in persistent ChromaDB at '{CHROMA_DB_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR
    )
    
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_pdfs()
