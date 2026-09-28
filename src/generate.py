import os
import sys

# Force UTF-8 encoding for standard output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from config import (
    CHROMA_DB_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
)

# Define the prompt template for grounded generation
RAG_PROMPT_TEMPLATE = """
You are a helpful AI assistant. You must answer the user's question ONLY using the provided context from retrieved documents.
If the context does not contain enough information to answer the question, simply say: "I cannot answer this based on the provided context."
Do not use outside knowledge or hallucinate information.

Context:
{context}

Question:
{question}

Answer:
"""

def generate_answer(query: str):
    # Ensure the GROQ_API_KEY is available
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is not set. Please add it to your .env file.")
        return

    print(f"Initializing embedding model: '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Loading ChromaDB from '{CHROMA_DB_DIR}'...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings
    )

    print(f"\nRetrieving relevant documents for: '{query}'")
    # Retrieve top 5 most relevant chunks
    results = vectorstore.similarity_search(query, k=5)

    if not results:
        print("No relevant documents found. Cannot generate an answer.")
        return

    # Extract the text content and track the unique sources
    context_chunks = []
    sources = set()

    for doc in results:
        context_chunks.append(doc.page_content.strip())
        
        # Track the source metadata
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page", "Unknown Page")
        sources.add(f"{source} (Page: {page})")

    # Combine the retrieved chunks into a single string
    combined_context = "\n\n---\n\n".join(context_chunks)

    # Initialize the Groq LLM
    print(f"Initializing Groq LLM: '{LLM_MODEL}'...")
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0, # Use 0 for more factual/grounded generation
    )

    # Prepare the prompt
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
    chain = prompt | llm

    print("\n" + "="*60)
    print("Generating Answer...")
    print("="*60 + "\n")

    # Generate the response
    response = chain.invoke({
        "context": combined_context,
        "question": query
    })

    # Print the final output
    print("ANSWER:")
    print(response.content.strip())
    
    print("\n" + "="*60)
    print("SOURCES USED:")
    for src in sorted(sources):
        print(f"- {src}")
    print("="*60)

if __name__ == "__main__":
    # Accept query from terminal arguments or prompt the user
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your search query: ")
        
    if user_query.strip():
        generate_answer(user_query.strip())
    else:
        print("No query provided. Exiting.")
