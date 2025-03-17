import streamlit as st
import os
import time
import tempfile
import pickle
from typing import List, Optional
import logging

# Import the RAG system components
from paste import (
    HybridRAGBot,
    AdvancedPDFProcessor,
    HybridRetriever,
    LLMGenerator,
    DocumentChunk,
    HF_TOKEN,
    MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Hybrid ML-LLM RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better UI
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    .stDownloadButton button {
        width: 100%;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .source-item {
        padding: 0.5rem;
        border-left: 3px solid #4CAF50;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_bot' not in st.session_state:
    st.session_state.rag_bot = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'pdf_directory' not in st.session_state:
    st.session_state.pdf_directory = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model_path' not in st.session_state:
    st.session_state.model_path = None

# Sidebar
with st.sidebar:
    st.title("Hybrid ML-LLM RAG System")
    st.markdown("### Configuration")
    
    hf_token = st.text_input("Hugging Face Token", value=HF_TOKEN, type="password")
    model_name = st.text_input("Model Name", value=MODEL_NAME)
    embedding_model = st.text_input("Embedding Model", value=EMBEDDING_MODEL_NAME)
    chunk_size = st.number_input("Chunk Size", value=CHUNK_SIZE, min_value=100, max_value=2000)
    chunk_overlap = st.number_input("Chunk Overlap", value=CHUNK_OVERLAP, min_value=0, max_value=500)
    top_k = st.number_input("Number of Chunks to Retrieve", value=5, min_value=1, max_value=20)
    
    st.markdown("---")
    st.markdown("### Model Info")
    if st.session_state.rag_bot:
        st.success("✅ RAG System initialized")
        if st.session_state.processed:
            st.success(f"✅ Documents processed: {len(st.session_state.rag_bot.chunks)} chunks")
        else:
            st.warning("⚠️ Documents not processed")
    else:
        st.warning("⚠️ RAG System not initialized")

# Main content
tab1, tab2, tab3 = st.tabs(["Upload & Process", "Query Documents", "System Status"])

# Tab 1: Upload & Process
with tab1:
    st.header("Upload & Process Documents")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload PDFs")
        
        # Create a file uploader for PDFs
        uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=["pdf"])
        
        if uploaded_files:
            # Create a temporary directory to store the uploaded PDFs
            temp_dir = tempfile.mkdtemp()
            st.session_state.pdf_directory = temp_dir
            
            # Save the uploaded PDFs to the temporary directory
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ {len(uploaded_files)} files uploaded to temporary directory")
    
    with col2:
        st.subheader("Process Documents")
        
        # Process the uploaded documents
        if st.button("Process Documents", key="process_docs"):
            if st.session_state.pdf_directory:
                with st.spinner("Processing documents... This may take a while."):
                    # Initialize the RAG bot
                    st.session_state.rag_bot = HybridRAGBot(
                        pdf_directory=st.session_state.pdf_directory,
                        model_name=model_name,
                        embedding_model=embedding_model,
                        auth_token=hf_token,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k
                    )
                    
                    # Process the documents
                    st.session_state.rag_bot.process_documents()
                    st.session_state.processed = True
                    
                    # Save the model for later use
                    model_path = os.path.join(tempfile.gettempdir(), "hybrid_rag_model.pkl")
                    st.session_state.rag_bot.save(model_path)
                    st.session_state.model_path = model_path
                    
                    st.success(f"✅ Documents processed successfully! Model saved.")
            else:
                st.error("❌ Please upload PDF files or specify a directory first.")
        
        # Load a previously saved model
        st.subheader("Load Existing Model")
        
        uploaded_model = st.file_uploader("Upload a saved model", type=["pkl"])
        
        if uploaded_model:
            with st.spinner("Loading model..."):
                # Save the uploaded model to a temporary file
                model_path = os.path.join(tempfile.gettempdir(), "uploaded_model.pkl")
                with open(model_path, "wb") as f:
                    f.write(uploaded_model.getbuffer())
                
                # Load the model
                try:
                    st.session_state.rag_bot = HybridRAGBot.load(
                        filepath=model_path,
                        auth_token=hf_token
                    )
                    st.session_state.processed = True
                    st.session_state.model_path = model_path
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {str(e)}")

# Tab 2: Query Documents
with tab2:
    st.header("Query Documents")
    
    if not st.session_state.processed:
        st.warning("⚠️ Please process documents or load a model first.")
    else:
        # Query input
        query = st.text_input("Enter your question about the documents")
        
        if st.button("Submit Question", key="submit_query") and query:
            with st.spinner("Generating answer..."):
                result = st.session_state.rag_bot.answer_query(query)
                
                # Add to history
                st.session_state.history.append(result)
            
            # Display answer
            st.markdown("### Answer")
            st.markdown(f"{result['answer']}")
            
            # Display sources
            st.markdown("### Sources")
            for source in result['sources']:
                with st.container():
                    st.markdown(f"""
                    <div class="source-item">
                        <strong>Document:</strong> {source['source']}<br>
                        <strong>Page:</strong> {source['page']}<br>
                        <strong>Relevance Score:</strong> {source['score']:.4f}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Query history
        if st.session_state.history:
            with st.expander("Query History", expanded=False):
                for i, item in enumerate(reversed(st.session_state.history)):
                    st.markdown(f"**Query {len(st.session_state.history) - i}:** {item['query']}")
                    st.markdown(f"**Answer:** {item['answer']}")
                    st.markdown("**Sources:**")
                    for source in item['sources']:
                        st.markdown(f"- {source['source']} (Page {source['page']}) - Score: {source['score']:.4f}")
                    st.markdown("---")

# Tab 3: System Status
with tab3:
    st.header("System Status")
    
    if st.session_state.rag_bot:
        # Display detailed information about the system
        st.subheader("RAG System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Configuration")
            st.markdown(f"**PDF Directory:** {st.session_state.pdf_directory}")
            st.markdown(f"**Model Name:** {model_name}")
            st.markdown(f"**Embedding Model:** {embedding_model}")
            st.markdown(f"**Chunk Size:** {chunk_size}")
            st.markdown(f"**Chunk Overlap:** {chunk_overlap}")
            st.markdown(f"**Top K Retrieval:** {top_k}")
        
        with col2:
            st.markdown("### Processing Statistics")
            if st.session_state.processed:
                st.markdown(f"**Total Chunks:** {len(st.session_state.rag_bot.chunks)}")
                
                # Count unique documents
                unique_docs = len(set(chunk.source for chunk in st.session_state.rag_bot.chunks))
                st.markdown(f"**Unique Documents:** {unique_docs}")
                
                # Count chunks per document
                doc_counts = {}
                for chunk in st.session_state.rag_bot.chunks:
                    doc_counts[chunk.source] = doc_counts.get(chunk.source, 0) + 1
                
                if doc_counts:
                    st.markdown("### Document Breakdown")
                    for doc, count in doc_counts.items():
                        st.markdown(f"- **{doc}:** {count} chunks")
        
        # Save model button
        if st.session_state.processed:
            st.subheader("Save Model")
            col1, col2 = st.columns(2)
            
            with col1:
                save_path = st.text_input("Save Path", value="hybrid_rag_model.pkl")
            
            with col2:
                if st.button("Save Model", key="save_model"):
                    with st.spinner("Saving model..."):
                        try:
                            st.session_state.rag_bot.save(save_path)
                            st.success(f"✅ Model saved to {save_path}")
                        except Exception as e:
                            st.error(f"❌ Error saving model: {str(e)}")
            
            # Download model button
            if st.session_state.model_path and os.path.exists(st.session_state.model_path):
                with open(st.session_state.model_path, "rb") as file:
                    st.download_button(
                        label="Download Model",
                        data=file,
                        file_name="hybrid_rag_model.pkl",
                        mime="application/octet-stream"
                    )
    else:
        st.warning("⚠️ RAG System not initialized. Please process documents or load a model first.")

# Footer
st.markdown("---")
st.markdown("Hybrid ML-LLM RAG System for Large Technical PDFs")
