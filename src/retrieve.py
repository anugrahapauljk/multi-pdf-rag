import os
import sys

# Force UTF-8 encoding for standard output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure the project root is in sys.path when running this script directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from src.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    RETRIEVAL_K,
    DISTANCE_THRESHOLD
)

def init_pipeline():
    print(f"Initializing embedding model: '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Loading ChromaDB from '{CHROMA_DB_DIR}'...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"Initializing Groq LLM: '{LLM_MODEL}'...")
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0, # Use 0 for more factual/grounded generation
    )
    
    return embeddings, vectorstore, llm

def retrieve(query: str, vectorstore, k=RETRIEVAL_K, threshold=DISTANCE_THRESHOLD):
    """
    Returns a list of (Document, score) tuples filtered by the distance threshold.
    For cosine distance, a lower score is better (0 = exact match).
    """
    print(f"\nPerforming similarity search for query: '{query}'")
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    # Filter by threshold
    filtered_results = [(doc, score) for doc, score in results if score <= threshold]
    return filtered_results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your search query: ")
        
    if user_query.strip():
        _, vectorstore, _ = init_pipeline()
        results = retrieve(user_query.strip(), vectorstore)
        if not results:
            print("No relevant results found below the distance threshold.")
        else:
            for i, (doc, distance_score) in enumerate(results, 1):
                source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown Source"))
                page = doc.metadata.get("page", "Unknown Page")
                
                print(f"Result #{i}")
                print(f"Distance Score: {distance_score:.4f} (lower is more similar)")
                print(f"Source: {source} | Page: {page}")
                print(f"Content:\n{doc.page_content.strip()}")
                print("-" * 60)
    else:
        print("No query provided. Exiting.")
