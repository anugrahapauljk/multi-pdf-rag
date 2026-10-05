import streamlit as st
import os

from src.config import DATA_DIR
from src.ingest import ingest_pdfs
from src.generate import init_pipeline, generate_answer

st.set_page_config(page_title="Multi-PDF RAG Assistant", page_icon="📚", layout="wide")

# Ensure DATA_DIR exists
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------------------------------------------
# CACHED RESOURCES
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Initializing AI Models (this may take a few seconds)...")
def get_rag_pipeline():
    """
    Load embeddings, vectorstore, and llm only once to avoid 
    slow reloads on every Streamlit interaction.
    """
    return init_pipeline()

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def clear_chat():
    st.session_state.messages = []

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.title("📚 Multi-PDF RAG")
    st.markdown("Upload your PDFs and process them to update the knowledge base.")
    
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents", type="primary"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                # Save uploaded files to the data directory
                for file in uploaded_files:
                    file_path = os.path.join(DATA_DIR, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.getbuffer())
                
                # Run the existing ingest logic
                try:
                    ingest_pdfs()
                    # Clear the cache for the vectorstore so it loads the new chunks!
                    get_rag_pipeline.clear()
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"Error during ingestion: {e}")
        else:
            st.warning("Please upload PDF files first.")
            
    st.divider()
    
    st.subheader("Existing Documents")
    existing_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]
    if existing_files:
        for f in existing_files:
            st.text(f"📄 {f}")
    else:
        st.text("No documents found.")
        
    st.divider()
    st.button("New Chat", on_click=clear_chat)

# ---------------------------------------------------------
# MAIN CHAT INTERFACE
# ---------------------------------------------------------
st.title("Chat with your PDFs")

if not st.session_state.messages:
    st.info("Welcome! Upload and process some PDFs in the sidebar, then ask a question below to get started.")

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")

# Handle new user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render and save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Render and generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Initialize models right before needed so UI isn't blocked
                embeddings, vectorstore, llm = get_rag_pipeline()
                
                # Call existing generate_answer logic, passing the cached objects
                result = generate_answer(prompt, vectorstore=vectorstore, llm=llm)
                
                if result:
                    answer, sources = result
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            for src in sources:
                                st.markdown(f"- {src}")
                                
                    # Save assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    err_msg = "I'm sorry, I couldn't generate an answer. Please make sure documents are processed and your API key is set."
                    st.markdown(err_msg)
                    st.session_state.messages.append({"role": "assistant", "content": err_msg})
                    
            except Exception as e:
                st.error(f"Error generating answer: {e}")
