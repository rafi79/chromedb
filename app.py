import streamlit as st
import os
import time
import pandas as pd
import numpy as np
import gc
import pickle
import tempfile
import plotly.express as px
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("streamlit_app")

# Import the RAG system
try:
    from hybrid_rag_system import (
        HybridRAGBot,
        AdvancedPDFProcessor,
        HybridRetriever,
        TFIDFRetriever,
        DenseRetriever,
        LLMGenerator,
        DocumentChunk,
        PDF_DIRECTORY,
        MODEL_NAME,
        EMBEDDING_MODEL_NAME,
        HF_TOKEN,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        SENTENCE_TRANSFORMERS_AVAILABLE,
        FAISS_AVAILABLE,
        TRANSFORMERS_AVAILABLE
    )
    SYSTEM_IMPORTED = True
except ImportError as e:
    logger.error(f"Error importing hybrid_rag_system: {str(e)}")
    SYSTEM_IMPORTED = False

# Page configuration
st.set_page_config(
    page_title="Hybrid ML-LLM RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F7FF;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E3A8A;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #ECFDF5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #047857;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFFBEB;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #D97706;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FEF2F2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #DC2626;
        margin-bottom: 1rem;
    }
    .source-box {
        background-color: #F3F4F6;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #6B7280;
    }
    .answer-box {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
        line-height: 1.6;
        border: 1px solid #E5E7EB;
    }
    .metric-card {
        background-color: #F9FAFB;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F3F4F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E3A8A;
        color: white;
    }
    .upload-box {
        border: 2px dashed #CBD5E1;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-card {
        background-color: #F0F7FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #BFDBFE;
    }
    .system-status {
        font-weight: bold;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        display: inline-block;
    }
    .status-ready {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .status-notready {
        background-color: #FEE2E2;
        color: #991B1B;
    }
    .status-processing {
        background-color: #FEF3C7;
        color: #92400E;
    }
</style>
""", unsafe_allow_html=True)

# Check if system is imported
if not SYSTEM_IMPORTED:
    st.error("""
    Failed to import the Hybrid RAG System. Please check that the file hybrid_rag_system.py 
    exists in the same directory as this app.py file and that all dependencies are installed.
    
    Error details are available in the logs.
    """)
    st.stop()

# Initialize session state variables
if 'rag_bot' not in st.session_state:
    st.session_state.rag_bot = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'selected_index' not in st.session_state:
    st.session_state.selected_index = 0
if 'pdf_files' not in st.session_state:
    st.session_state.pdf_files = []
if 'query_time' not in st.session_state:
    st.session_state.query_time = 0
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0
if 'num_chunks' not in st.session_state:
    st.session_state.num_chunks = 0
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'dependencies_checked' not in st.session_state:
    st.session_state.dependencies_checked = False
if 'available_dependencies' not in st.session_state:
    st.session_state.available_dependencies = {
        "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
        "faiss": FAISS_AVAILABLE,
        "transformers": TRANSFORMERS_AVAILABLE
    }

# Main header
st.markdown("<h1 class='main-header'>📚 Hybrid ML-LLM RAG System</h1>", unsafe_allow_html=True)

# App description
st.markdown("""
<div class="info-box">
    <p>This application provides an interface for the Hybrid ML-LLM RAG (Retrieval-Augmented Generation) System, 
    which combines traditional ML approaches with large language models to provide accurate answers from your PDF documents.</p>
    <b>Key features:</b>
    <ul>
        <li>Handles large technical PDFs (500+ pages)</li>
        <li>Uses hybrid retrieval (TF-IDF + dense embeddings)</li>
        <li>Optimized chunking for technical content</li>
        <li>Memory-efficient batch processing</li>
        <li>Uses Gemma 3 for generation</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.image("https://raw.githubusercontent.com/huggingface/awesome-huggingface/main/logo.svg", width=230)
    st.markdown("## Configuration")
    
    # Input for HF Token
    st.markdown("### Hugging Face Token")
    st.markdown("Your token is used to access Hugging Face models like Gemma 3.")
    hf_token = st.text_input("Token", value=HF_TOKEN, type="password")
    
    # Input for PDF directory
    st.markdown("### PDF Directory")
    pdf_dir = st.text_input("Directory Path", value=PDF_DIRECTORY)
    
    # Upload PDFs option
    st.markdown("### Upload PDFs")
    uploaded_files = st.file_uploader("Upload PDF files", type=['pdf'], accept_multiple_files=True)
    
    # Process uploaded files
    if uploaded_files:
        pdf_upload_dir = os.path.join(os.getcwd(), "uploaded_pdfs")
        
        # Create directory if it doesn't exist
        if not os.path.exists(pdf_upload_dir):
            os.makedirs(pdf_upload_dir)
        
        pdf_dir = pdf_upload_dir
        
        # Save the uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(pdf_upload_dir, uploaded_file.name)
            
            # Only save if the file doesn't exist already
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.sidebar.success(f"Saved: {uploaded_file.name}")
    
    # Advanced settings expander
    with st.expander("Advanced Settings"):
        # Model settings
        st.markdown("#### Model Settings")
        model_name = st.text_input("LLM Model", value=MODEL_NAME, help="Hugging Face model name for text generation")
        embedding_model = st.text_input("Embedding Model", value=EMBEDDING_MODEL_NAME, help="Model for document embeddings")
        
        # Chunking settings
        st.markdown("#### Chunking Settings")
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=CHUNK_SIZE, step=50, 
                              help="Size of document chunks for processing")
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=CHUNK_OVERLAP, step=10, 
                                help="Overlap between document chunks")
        
        # Retrieval settings
        st.markdown("#### Retrieval Settings")
        top_k = st.slider("Number of Chunks for Context", min_value=1, max_value=20, value=5, 
                         help="Number of most relevant chunks to use for context")
        tfidf_weight = st.slider("TF-IDF Weight", min_value=0.0, max_value=1.0, value=0.3, step=0.05, 
                               help="Weight for TF-IDF retrieval (vs. dense retrieval)")
        
        # Generation settings
        st.markdown("#### Generation Settings")
        max_new_tokens = st.slider("Max Tokens", min_value=100, max_value=1000, value=512, step=50, 
                                 help="Maximum number of tokens to generate")
        
        # Show dependency status
        st.markdown("#### System Dependencies")
        dependencies = st.session_state.available_dependencies
        
        st.markdown("Dense embeddings: " + 
                   ("✅ Available" if dependencies["sentence_transformers"] else "❌ Not available"))
        st.markdown("FAISS indexing: " + 
                   ("✅ Available" if dependencies["faiss"] else "❌ Not available"))
        st.markdown("LLM generation: " + 
                   ("✅ Available" if dependencies["transformers"] else "❌ Not available"))
    
    # Initialize button
    if st.button("Initialize and Process Documents", type="primary", disabled=st.session_state.processing):
        with st.spinner("Initializing system..."):
            try:
                # Set processing flag
                st.session_state.processing = True
                st.session_state.error_message = None
                
                # Check if directory exists
                if not os.path.exists(pdf_dir):
                    try:
                        os.makedirs(pdf_dir, exist_ok=True)
                        st.warning(f"Created directory {pdf_dir} as it did not exist")
                    except Exception as e:
                        st.session_state.error_message = f"Error creating directory: {str(e)}"
                        st.session_state.processing = False
                        st.error(st.session_state.error_message)
                        st.stop()
                
                # Check if there are PDF files
                pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
                if not pdf_files:
                    st.session_state.error_message = f"No PDF files found in {pdf_dir}"
                    st.session_state.processing = False
                    st.error(st.session_state.error_message)
                    st.stop()
                
                # Start timer
                start_time = time.time()
                
                # Initialize RAG bot with the specified settings
                st.session_state.rag_bot = HybridRAGBot(
                    pdf_directory=pdf_dir,
                    model_name=model_name,
                    embedding_model=embedding_model,
                    auth_token=hf_token,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k
                )
                
                # Configure hybrid retriever weights if available
                if hasattr(st.session_state.rag_bot, 'retriever') and hasattr(st.session_state.rag_bot.retriever, 'tfidf_weight'):
                    st.session_state.rag_bot.retriever.tfidf_weight = tfidf_weight
                    st.session_state.rag_bot.retriever.dense_weight = 1.0 - tfidf_weight
                
                # Process documents
                st.session_state.rag_bot.process_documents()
                
                # Update processing time
                processing_time = time.time() - start_time
                st.session_state.processing_time = processing_time
                
                # Update session state
                st.session_state.processed = True
                st.session_state.pdf_files = pdf_files
                
                # Update number of chunks
                if hasattr(st.session_state.rag_bot, 'chunks'):
                    st.session_state.num_chunks = len(st.session_state.rag_bot.chunks)
                
            except Exception as e:
                st.session_state.error_message = f"Error initializing system: {str(e)}"
                st.error(st.session_state.error_message)
                logger.error(f"Initialization error: {str(e)}", exc_info=True)
            finally:
                st.session_state.processing = False
    
    # Load model button
    with st.expander("Load Saved Model"):
        uploaded_model = st.file_uploader("Upload saved model (.pkl)", type=["pkl"])
        
        if uploaded_model is not None:
            if st.button("Load Model", key="load_model_button"):
                with st.spinner("Loading model..."):
                    try:
                        # Save the uploaded model to a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                            tmp_file.write(uploaded_model.getbuffer())
                            model_path = tmp_file.name
                        
                        # Load the model
                        st.session_state.rag_bot = HybridRAGBot.load(
                            model_path, 
                            pdf_directory=pdf_dir,
                            auth_token=hf_token
                        )
                        
                        # Update session state
                        st.session_state.processed = True
                        
                        # Update number of chunks
                        if hasattr(st.session_state.rag_bot, 'chunks'):
                            st.session_state.num_chunks = len(st.session_state.rag_bot.chunks)
                        
                        # Try to get PDF files
                        try:
                            pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
                            st.session_state.pdf_files = pdf_files
                        except:
                            # Not critical if this fails
                            pass
                        
                        st.success("Model loaded successfully!")
                        
                        # Clean up the temporary file
                        os.unlink(model_path)
                        
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
                        logger.error(f"Model loading error: {str(e)}", exc_info=True)
    
    # Save model button (only if processed)
    if st.session_state.processed and st.session_state.rag_bot:
        if st.button("Save Current Model"):
            with st.spinner("Saving model..."):
                try:
                    # Generate a filename
                    save_path = f"hybrid_rag_model_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
                    
                    # Save the model
                    st.session_state.rag_bot.save(save_path)
                    
                    # Provide download link
                    with open(save_path, "rb") as file:
                        st.download_button(
                            label="Download Saved Model",
                            data=file,
                            file_name=save_path,
                            mime="application/octet-stream"
                        )
                    
                    st.success(f"Model saved as {save_path}")
                    
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
    
    # Display system status
    st.markdown("### System Status")
    if st.session_state.processed:
        st.markdown('<span class="system-status status-ready">✅ System Ready</span>', unsafe_allow_html=True)
    elif st.session_state.processing:
        st.markdown('<span class="system-status status-processing">⏳ Processing...</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="system-status status-notready">❌ Not Initialized</span>', unsafe_allow_html=True)
    
    # Show error message if any
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
    
    # Display processing metrics if available
    if st.session_state.processed:
        st.markdown("### System Metrics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Documents", len(st.session_state.pdf_files) if st.session_state.pdf_files else 0)
            st.metric("Total Chunks", st.session_state.num_chunks)
        
        with col2:
            st.metric("Processing Time", f"{st.session_state.processing_time:.1f}s")
            st.metric("Avg. Query Time", f"{st.session_state.query_time:.2f}s" if st.session_state.query_time > 0 else "N/A")
    
    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application provides an interface for a hybrid RAG system that combines traditional ML techniques with large language models to provide accurate answers from your documents.
    
    Built with Streamlit & Hugging Face.
    """)

# Main content area - Create tabs
tabs = st.tabs(["Ask Questions", "Document Explorer", "Query History", "System Info"])

# Ask Questions Tab
with tabs[0]:
    if not st.session_state.processed:
        st.markdown(
            '<div class
