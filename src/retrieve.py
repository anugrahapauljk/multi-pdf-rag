import os
import sys

# Ensure the project root is in sys.path when running this script directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

def retrieve(query: str):
    print(f"Initializing embedding model: '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Loading ChromaDB from '{CHROMA_DB_DIR}'...")
    # Initialize Chroma connected to the existing persistent directory
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    print(f"\nPerforming similarity search for query: '{query}'")
    print("-" * 60)
    
    # similarity_search_with_score returns a list of (Document, score) tuples.
    # By default in LangChain's Chroma integration, the score represents the L2 distance.
    # Lower distance means higher similarity.
    results = vectorstore.similarity_search_with_score(query, k=5)

    if not results:
        print("No results found.")
        return

    for i, (doc, distance_score) in enumerate(results, 1):
        # Extract metadata
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "Unknown Page")
        
        # Display the result
        print(f"Result #{i}")
        print(f"Distance Score: {distance_score:.4f} (lower is more similar)")
        print(f"Source: {source} | Page: {page}")
        print(f"Content:\n{doc.page_content.strip()}")
        print("-" * 60)

if __name__ == "__main__":
    # Accept query from terminal arguments or prompt the user
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your search query: ")
        
    if user_query.strip():
        retrieve(user_query.strip())
    else:
        print("No query provided. Exiting.")
