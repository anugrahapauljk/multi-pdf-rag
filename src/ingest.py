import os
import sys
import hashlib

# Ensure the project root is in sys.path when running this script directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_community.document_loaders import PyPDFLoader
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

def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def ingest_pdfs():
    """
    Loads PDFs from the data directory with robust deduplication.
    Generates embeddings, and stores them in a local ChromaDB instance.
    """
    print(f"Scanning '{DATA_DIR}' for PDFs...")
    
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} does not exist.")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No documents found in {DATA_DIR}. Please add some PDFs.")
        return

    print(f"Initializing Hugging Face embedding model: '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Connecting to persistent ChromaDB at '{CHROMA_DB_DIR}'...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Pre-fetch existing metadatas to map filename -> doc_id
    existing_data = vectorstore.get(include=["metadatas"])
    existing_ids = existing_data.get("ids", [])
    existing_metadatas = existing_data.get("metadatas", [])
    
    file_to_docid = {}
    for meta in existing_metadatas:
        if meta and "filename" in meta and "doc_id" in meta:
            file_to_docid[meta["filename"]] = meta["doc_id"]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    for filename in pdf_files:
        filepath = os.path.join(DATA_DIR, filename)
        try:
            file_hash = compute_sha256(filepath)
            
            # 1. Deduplication Check
            if filename in file_to_docid:
                existing_hash = file_to_docid[filename]
                if existing_hash == file_hash:
                    print(f"Skipping '{filename}': Already indexed and up to date.")
                    continue
                else:
                    print(f"Updating '{filename}': Content changed. Removing old version...")
                    # Find and delete all chunks belonging to this filename
                    ids_to_delete = [
                        existing_ids[i] for i, meta in enumerate(existing_metadatas) 
                        if meta and meta.get("filename") == filename
                    ]
                    if ids_to_delete:
                        vectorstore.delete(ids=ids_to_delete)
            
            # 2. Process New/Updated File
            print(f"Processing '{filename}'...")
            loader = PyPDFLoader(filepath)
            docs = loader.load()
            
            # Check for extractable text
            total_text = "".join([d.page_content for d in docs]).strip()
            if not total_text:
                print(f"Warning: No extractable text found in '{filename}'. Skipping.")
                continue
                
            # Clean metadata
            for d in docs:
                d.metadata = {
                    "doc_id": file_hash,
                    "filename": filename,
                    "page": d.metadata.get("page", 0) + 1  # 1-indexed
                }
            
            # Split
            chunks = text_splitter.split_documents(docs)
            
            # 3. Assign Deterministic IDs & Store
            chunk_ids = [f"{file_hash}:{i}" for i in range(len(chunks))]
            vectorstore.add_documents(documents=chunks, ids=chunk_ids)
            
            print(f"Successfully indexed '{filename}': {len(chunks)} chunks.")
            
        except Exception as e:
            print(f"Error processing '{filename}': {e}")
            
    print("Ingestion sweep complete!")

if __name__ == "__main__":
    ingest_pdfs()
