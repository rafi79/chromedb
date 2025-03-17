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
        st.markdown('<div class="warning-box">Please initialize the system using the sidebar controls first.</div>', unsafe_allow_html=True)
    else:
        st.markdown("<h2 class='sub-header'>Ask Questions About Your Documents</h2>", unsafe_allow_html=True)
        
        # Query input
        query = st.text_area("Enter your question", height=100, placeholder="What would you like to know about the documents?")
        
        col1, col2 = st.columns([1, 5])
        
        # Submit button
        with col1:
            submit_button = st.button("Submit", type="primary", disabled=not query or len(query.strip()) < 3)
        
        # Query settings
        with col2:
            max_context = st.slider("Context chunks", min_value=1, max_value=20, value=st.session_state.rag_bot.top_k if st.session_state.rag_bot else 5, 
                                 help="Number of document chunks to retrieve for context")
        
        # Process query when button is clicked
        if submit_button and query:
            with st.spinner("Generating answer..."):
                try:
                    # Record start time
                    start_time = time.time()
                    
                    # Update top_k if changed
                    if st.session_state.rag_bot.top_k != max_context:
                        st.session_state.rag_bot.top_k = max_context
                    
                    # Generate answer
                    result = st.session_state.rag_bot.answer_query(query)
                    
                    # Calculate query time
                    query_time = time.time() - start_time
                    
                    # Update average query time
                    if st.session_state.query_time == 0:
                        st.session_state.query_time = query_time
                    else:
                        st.session_state.query_time = (st.session_state.query_time + query_time) / 2
                    
                    # Save to history
                    st.session_state.query_history.append({
                        "query": query,
                        "answer": result["answer"],
                        "sources": result["sources"],
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "query_time": query_time
                    })
                    
                    # Display success message with time
                    st.markdown(
                        f'<div class="success-box">Answer generated in {query_time:.2f} seconds</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Display answer
                    st.markdown("<h3>Answer:</h3>", unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="answer-box">{result["answer"].replace(chr(10), "<br>")}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Display sources
                    st.markdown("<h3>Sources:</h3>", unsafe_allow_html=True)
                    
                    for i, source in enumerate(result["sources"]):
                        st.markdown(
                            f'<div class="source-box"><b>{source["source"]}</b> (Page {source["page"]})<br>'
                            f'Relevance: {source["score"]:.4f}</div>',
                            unsafe_allow_html=True
                        )
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.error(f"Query processing error: {str(e)}", exc_info=True)

# Document Explorer Tab
with tabs[1]:
    if not st.session_state.processed:
        st.markdown('<div class="warning-box">Please initialize the system using the sidebar controls first.</div>', unsafe_allow_html=True)
    else:
        st.markdown("<h2 class='sub-header'>Document Explorer</h2>", unsafe_allow_html=True)
        
        # Display list of documents
        st.markdown(f"### {len(st.session_state.pdf_files)} Document{'s' if len(st.session_state.pdf_files) != 1 else ''}")
        
        # Create a table of documents
        if st.session_state.pdf_files:
            # Create a dataframe for the documents table
            docs_data = []
            for pdf_file in st.session_state.pdf_files:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024) if os.path.exists(pdf_path) else 0
                
                # Count chunks for this document if available
                doc_chunks = 0
                if hasattr(st.session_state.rag_bot, 'chunks'):
                    doc_chunks = sum(1 for chunk in st.session_state.rag_bot.chunks if chunk.source == pdf_file)
                
                docs_data.append({
                    "Filename": pdf_file,
                    "Size (MB)": f"{file_size_mb:.2f}",
                    "Chunks": doc_chunks
                })
            
            # Convert to dataframe and display
            docs_df = pd.DataFrame(docs_data)
            st.dataframe(docs_df, use_container_width=True)
            
            # Visualize chunk distribution
            if hasattr(st.session_state.rag_bot, 'chunks') and st.session_state.rag_bot.chunks:
                st.markdown("### Chunk Distribution")
                
                # Count chunks per document
                chunk_counts = {}
                for chunk in st.session_state.rag_bot.chunks:
                    if chunk.source in chunk_counts:
                        chunk_counts[chunk.source] += 1
                    else:
                        chunk_counts[chunk.source] = 1
                
                # Convert to dataframe for plotting
                chunk_df = pd.DataFrame({
                    'Document': list(chunk_counts.keys()),
                    'Chunks': list(chunk_counts.values())
                })
                
                # Plot horizontal bar chart
                fig = px.bar(
                    chunk_df,
                    x='Chunks',
                    y='Document',
                    orientation='h',
                    title='Number of Chunks per Document',
                    color='Chunks',
                    color_continuous_scale='Blues'
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Number of Chunks',
                    yaxis_title='Document',
                    height=max(300, 50 * len(chunk_counts)),
                    yaxis={'categoryorder': 'total ascending'}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Chunk length distribution
                st.markdown("### Chunk Length Distribution")
                
                # Get length of each chunk's text
                chunk_lengths = [len(chunk.text) for chunk in st.session_state.rag_bot.chunks]
                
                # Create histogram
                fig = px.histogram(
                    x=chunk_lengths,
                    nbins=20,
                    title='Distribution of Chunk Lengths',
                    labels={'x': 'Chunk Length (characters)'},
                    color_discrete_sequence=['#1E3A8A']
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title='Chunk Length (characters)',
                    yaxis_title='Count',
                    bargap=0.1
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show chunk statistics
                st.markdown("### Chunk Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Chunks", len(st.session_state.rag_bot.chunks))
                
                with col2:
                    st.metric("Avg. Chunk Length", f"{np.mean(chunk_lengths):.0f} chars")
                
                with col3:
                    st.metric("Min Chunk Length", f"{min(chunk_lengths)} chars")
                
                with col4:
                    st.metric("Max Chunk Length", f"{max(chunk_lengths)} chars")
                
                # Show chunk explorer
                st.markdown("### Chunk Explorer")
                
                # Allow user to select a document to explore
                selected_doc = st.selectbox(
                    "Select Document", 
                    options=list(chunk_counts.keys()),
                    index=0
                )
                
                # Get chunks for the selected document
                doc_chunks = [chunk for chunk in st.session_state.rag_bot.chunks if chunk.source == selected_doc]
                
                # Display pagination controls
                chunks_per_page = 5
                total_pages = (len(doc_chunks) + chunks_per_page - 1) // chunks_per_page
                
                col1, col2, col3 = st.columns([1, 3, 1])
                with col1:
                    if st.button("← Previous", disabled=st.session_state.selected_index == 0):
                        st.session_state.selected_index = max(0, st.session_state.selected_index - 1)
                        st.experimental_rerun()
                
                with col2:
                    st.markdown(f"<div style='text-align: center'>Page {st.session_state.selected_index + 1} of {total_pages}</div>", unsafe_allow_html=True)
                
                with col3:
                    if st.button("Next →", disabled=st.session_state.selected_index >= total_pages - 1):
                        st.session_state.selected_index = min(total_pages - 1, st.session_state.selected_index + 1)
                        st.experimental_rerun()
                
                # Display chunks for the current page
                start_idx = st.session_state.selected_index * chunks_per_page
                end_idx = min(start_idx + chunks_per_page, len(doc_chunks))
                
                # Display each chunk
                for i, chunk in enumerate(doc_chunks[start_idx:end_idx]):
                    with st.expander(f"Chunk {chunk.chunk_id} (Page {chunk.page})"):
                        st.text_area(
                            "Chunk Content",
                            value=chunk.text,
                            height=200,
                            disabled=True
                        )
                        
                        # Show chunk metadata if available
                        if hasattr(chunk, 'embedding') and chunk.embedding is not None:
                            st.markdown(f"Embedding dimensions: {len(chunk.embedding)}")
        else:
            st.info("No documents available. Please initialize the system with PDF files.")

# Query History Tab
with tabs[2]:
    st.markdown("<h2 class='sub-header'>Query History</h2>", unsafe_allow_html=True)
    
    if not st.session_state.query_history:
        st.markdown(
            '<div class="info-box">No queries have been made yet.</div>', 
            unsafe_allow_html=True
        )
    else:
        # Option to clear history
        if st.button("Clear History"):
            st.session_state.query_history = []
            st.experimental_rerun()
        
        # Export history to CSV
        if st.button("Export History to CSV"):
            # Prepare data for CSV
            history_data = []
            for item in st.session_state.query_history:
                # Format sources as string
                sources_str = "; ".join([
                    f"{s['source']} (Page {s['page']}, Score: {s['score']:.4f})" 
                    for s in item["sources"]
                ])
                
                history_data.append({
                    "Timestamp": item["timestamp"],
                    "Query": item["query"],
                    "Answer": item["answer"],
                    "Sources": sources_str,
                    "Query Time (s)": f"{item['query_time']:.2f}"
                })
            
            # Convert to DataFrame and then to CSV
            history_df = pd.DataFrame(history_data)
            csv = history_df.to_csv(index=False)
            
            # Provide download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="query_history.csv",
                mime="text/csv"
            )
        
        # Display history entries
        for i, item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"**Q: {item['query']}** ({item['timestamp']})"):
                st.markdown(f"**Query Time:** {item['query_time']:.2f} seconds")
                
                # Display answer
                st.markdown("<h4>Answer:</h4>", unsafe_allow_html=True)
                st.markdown(
                    f'<div class="answer-box">{item["answer"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True
                )
                
                # Display sources
                st.markdown("<h4>Sources:</h4>", unsafe_allow_html=True)
                for source in item["sources"]:
                    st.markdown(
                        f'<div class="source-box"><b>{source["source"]}</b> (Page {source["page"]})<br>'
                        f'Relevance: {source["score"]:.4f}</div>',
                        unsafe_allow_html=True
                    )

# System Info Tab
with tabs[3]:
    st.markdown("<h2 class='sub-header'>System Information</h2>", unsafe_allow_html=True)
    
    # Overview section
    st.markdown("### System Overview")
    st.markdown("""
    <div class="info-box">
        This Hybrid ML-LLM RAG System combines traditional machine learning approaches with large language models
        to effectively process and query large technical documents.
    </div>
    """, unsafe_allow_html=True)
    
    # Architecture diagram
    st.markdown("### System Architecture")
    
    # Using columns to organize the display
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a simple architecture diagram using st.graphviz_chart
        try:
            import graphviz
            
            # Create the graph
            graph = graphviz.Digraph()
            graph.attr(rankdir='TB')
            
            # Add nodes
            graph.node('docs', 'PDF Documents', style='filled', fillcolor='lightyellow')
            graph.node('processor', 'Advanced PDF Processor', style='filled', fillcolor='lightblue')
            graph.node('chunks', 'Document Chunks', style='filled', fillcolor='lightblue')
            graph.node('embedder', 'Embeddings Generation', style='filled', fillcolor='lightblue')
            graph.node('tfidf', 'TF-IDF Vectorizer', style='filled', fillcolor='lightblue')
            graph.node('dense', 'Dense Vectorizer', style='filled', fillcolor='lightblue')
            graph.node('hybrid', 'Hybrid Retriever', style='filled', fillcolor='lightblue')
            graph.node('llm', 'Gemma 3 LLM', style='filled', fillcolor='lightgreen')
            graph.node('query', 'User Query', style='filled', fillcolor='lightyellow')
            graph.node('answer', 'Generated Answer', style='filled', fillcolor='lightgreen')
            
            # Add edges
            graph.edge('docs', 'processor')
            graph.edge('processor', 'chunks')
            graph.edge('chunks', 'embedder')
            graph.edge('chunks', 'tfidf')
            graph.edge('embedder', 'dense')
            graph.edge('tfidf', 'hybrid')
            graph.edge('dense', 'hybrid')
            graph.edge('query', 'hybrid')
            graph.edge('hybrid', 'llm')
            graph.edge('query', 'llm')
            graph.edge('llm', 'answer')
            
            # Display the graph
            st.graphviz_chart(graph)
            
        except ImportError:
            # Fallback if graphviz is not available
            st.info("To see the architecture diagram, please install the graphviz library.")
            
            # Show text representation instead
            st.code("""
            PDF Documents → Advanced PDF Processor → Document Chunks
            Document Chunks → TF-IDF Vectorizer → Hybrid Retriever
            Document Chunks → Embeddings Generation → Dense Vectorizer → Hybrid Retriever
            User Query → Hybrid Retriever → Gemma 3 LLM → Generated Answer
            User Query → Gemma 3 LLM
            """)
    
    with col2:
        # Component descriptions
        st.markdown("#### Key Components")
        st.markdown("""
        - **PDF Processor**: Advanced chunking optimized for technical content
        - **Hybrid Retriever**: Combines TF-IDF and dense embeddings
        - **Gemma 3 LLM**: Generates human-like responses
        """)
    
    # System details as an expander
    with st.expander("System Details"):
        st.markdown("#### Model Configuration")
        
        if st.session_state.processed and st.session_state.rag_bot:
            # Get configuration from the current rag_bot instance
            model_info = {
                "LLM Model": st.session_state.rag_bot.generator.model_name,
                "Embedding Model": getattr(st.session_state.rag_bot.retriever.dense_retriever, 'model_name', EMBEDDING_MODEL_NAME),
                "Chunk Size": st.session_state.rag_bot.pdf_processor.chunk_size,
                "Chunk Overlap": st.session_state.rag_bot.pdf_processor.chunk_overlap,
                "TF-IDF Weight": getattr(st.session_state.rag_bot.retriever, 'tfidf_weight', 0.3),
                "Dense Weight": getattr(st.session_state.rag_bot.retriever, 'dense_weight', 0.7),
                "Top-K Results": st.session_state.rag_bot.top_k
            }
        else:
            # Default configuration from constants
            model_info = {
                "LLM Model": MODEL_NAME,
                "Embedding Model": EMBEDDING_MODEL_NAME,
                "Chunk Size": CHUNK_SIZE,
                "Chunk Overlap": CHUNK_OVERLAP,
                "TF-IDF Weight": 0.3,
                "Dense Weight": 0.7,
                "Top-K Results": 5
            }
        
        # Convert to dataframe and display
        model_df = pd.DataFrame(list(model_info.items()), columns=["Parameter", "Value"])
        st.table(model_df)
        
        # Dependencies
        st.markdown("#### Dependencies")
        
        dep_data = [
            {"Dependency": "sentence-transformers", "Status": "Available" if SENTENCE_TRANSFORMERS_AVAILABLE else "Not Available", "Purpose": "Dense embeddings generation"},
            {"Dependency": "FAISS", "Status": "Available" if FAISS_AVAILABLE else "Not Available", "Purpose": "Vector similarity search"},
            {"Dependency": "Transformers", "Status": "Available" if TRANSFORMERS_AVAILABLE else "Not Available", "Purpose": "LLM inference"}
        ]
        
        dep_df = pd.DataFrame(dep_data)
        st.table(dep_df)
    
    # Performance metrics
    st.markdown("### Performance Metrics")
    
    if st.session_state.processed and hasattr(st.session_state, 'query_history') and st.session_state.query_history:
        # Extract query times from history
        query_times = [item['query_time'] for item in st.session_state.query_history]
        
        # Create a dataframe for plotting
        metrics_df = pd.DataFrame({
            'Query': [f"Q{i+1}" for i in range(len(query_times))],
            'Time (seconds)': query_times
        })
        
        # Plot query times
        fig = px.line(
            metrics_df, 
            x='Query', 
            y='Time (seconds)', 
            title='Query Response Time',
            markers=True,
            line_shape='linear',
            color_discrete_sequence=['#1E3A8A']
        )
        
        fig.update_layout(
            xaxis_title='Query',
            yaxis_title='Time (seconds)',
            yaxis=dict(range=[0, max(query_times) * 1.1])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Query time statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Query Time", f"{np.mean(query_times):.2f}s")
        
        with col2:
            st.metric("Fastest Query", f"{min(query_times):.2f}s")
        
        with col3:
            st.metric("Slowest Query", f"{max(query_times):.2f}s")
    else:
        st.info("Query performance metrics will be shown after you've made some queries.")
    
    # Usage tips
    st.markdown("### Usage Tips")
    
    tip_col1, tip_col2 = st.columns(2)
    
    with tip_col1:
        st.markdown("""
        <div class="feature-card">
            <h4>Effective Querying</h4>
            <ul>
                <li>Be specific in your questions</li>
                <li>Refer to document titles when known</li>
                <li>Ask follow-up questions for clarification</li>
                <li>Adjust the context chunks for complex queries</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with tip_col2:
        st.markdown("""
        <div class="feature-card">
            <h4>System Management</h4>
            <ul>
                <li>Save your model after processing large document sets</li>
                <li>Increase chunk overlap for technical content</li>
                <li>Adjust TF-IDF weight based on your document types</li>
                <li>Use regular expressions in queries for precise matching</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
