import os
import sys

# Force UTF-8 encoding for standard output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure the project root is in sys.path when running this script directly
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from langchain_core.prompts import PromptTemplate
from src.retrieve import init_pipeline, retrieve

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

def generate_answer(query: str, vectorstore=None, llm=None):
    # Ensure the GROQ_API_KEY is available
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is not set. Please add it to your .env file.")
        return None

    if vectorstore is None or llm is None:
        _, vectorstore, llm = init_pipeline()

    # Call the new threshold-based retrieve function
    results = retrieve(query, vectorstore)

    if not results:
        msg = "I couldn't find this information in the provided documents."
        print(msg)
        return msg, []

    # Extract the text content and track the unique sources
    context_chunks = []
    sources = set()

    # Note: results are now tuples of (doc, score)
    for doc, score in results:
        context_chunks.append(doc.page_content.strip())
        
        # Track the source metadata (using filename if source isn't explicitly there)
        source = doc.metadata.get("filename", doc.metadata.get("source", "Unknown Source"))
        page = doc.metadata.get("page", "Unknown Page")
        sources.add(f"{source} (Page: {page})")

    # Combine the retrieved chunks into a single string
    combined_context = "\n\n---\n\n".join(context_chunks)

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
    
    return response.content.strip(), sorted(sources)

if __name__ == "__main__":
    # Accept query from terminal arguments or prompt the user
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter your search query: ")
        
    if user_query.strip():
        result = generate_answer(user_query.strip())
        if result:
            ans, srcs = result
            print("ANSWER:")
            print(ans)
            if srcs:
                print("\n" + "="*60)
                print("SOURCES USED:")
                for src in srcs:
                    print(f"- {src}")
                print("="*60)
    else:
        print("No query provided. Exiting.")
