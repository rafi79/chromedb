import streamlit as st
import os
import sys
import time
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

# Import the RAG system
# Assuming the RAG code is in a file called hybrid_rag.py in the same directory
# If it's in the paste.txt file, save that content to hybrid_rag.py first
import importlib.util
try:
    # Try to import if already saved as a module
    from hybrid_rag import HybridRAGBot, DocumentChunk, AdvancedPDFProcessor, HybridRetriever, LLMGenerator
except ImportError:
    # If not saved yet, inform the user
    st.error("Please save the RAG system code as 'hybrid_rag.py' in the same directory as this app.")
    st.stop()

# Constants from the original code
PDF_DIRECTORY = "pdfs"
HF_TOKEN = "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"  # Default token
MODEL_NAME = "google/gemma-3-1b-it"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Streamlit app title and description
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
        color: #4527A0;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #5E35B1;
        margin-top: 0;
    }
    .section-header {
        font-size: 1.8rem;
        color: #673AB7;
        margin-top: 1rem;
    }
    .highlight {
        background-color: #EDE7F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF8E1;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div {
        background-color: #673AB7;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">Hybrid ML-LLM RAG System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Intelligent Document Question Answering</p>', unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.markdown("## Configuration")

# PDF Directory
pdf_dir = st.sidebar.text_input("PDF Directory", value=PDF_DIRECTORY)

# HF Token
hf_token = st.sidebar.text_input("Hugging Face Token", value=HF_TOKEN, type="password")

# Advanced settings with expander
with st.sidebar.expander("Advanced Settings", expanded=False):
    model_name = st.text_input("LLM Model", value=MODEL_NAME)
    embedding_model = st.text_input("Embedding Model", value=EMBEDDING_MODEL_NAME)
    chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=CHUNK_SIZE, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=CHUNK_OVERLAP, step=50)
    top_k = st.slider("Top K Results", min_value=1, max_value=10, value=5, step=1)

# Model loading/saving section
st.sidebar.markdown("## Model Management")

# File uploader for model
uploaded_model = st.sidebar.file_uploader("Upload Saved Model", type=["pkl"])

# Save model filename
save_filename = st.sidebar.text_input("Save Model Filename", value="hybrid_rag_model.pkl")

# Initialize session state for storing the RAG bot
if 'rag_bot' not in st.session_state:
    st.session_state.rag_bot = None
    st.session_state.model_loaded = False
    st.session_state.document_stats = None
    st.session_state.is_processing = False
    st.session_state.query_history = []

# Main app tabs
tabs = st.tabs(["Setup & Process", "Query Documents", "Document Explorer", "Query History"])

# Setup tab
with tabs[0]:
    st.markdown('<p class="section-header">Setup & Process Documents</p>', unsafe_allow_html=True)
    
    # PDF directory setup
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("### 1. PDF Directory Setup")
    st.write(f"Current PDF directory: `{pdf_dir}`")
    
    # Check if directory exists
    if not os.path.exists(pdf_dir):
        try:
            os.makedirs(pdf_dir)
            st.success(f"Created directory: {pdf_dir}")
        except Exception as e:
            st.error(f"Error creating directory: {str(e)}")
    
    # Count PDFs in directory
    pdf_files = []
    pdf_stats = {"Total Files": 0, "Total Size (MB)": 0}
    
    if os.path.exists(pdf_dir):
        for root, _, files in os.walk(pdf_dir):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    pdf_files.append({"File": file, "Size (MB)": f"{file_size:.2f}"})
                    pdf_stats["Total Files"] += 1
                    pdf_stats["Total Size (MB)"] += file_size
    
    if pdf_files:
        st.write(f"Found {pdf_stats['Total Files']} PDF files (Total: {pdf_stats['Total Size (MB)']:.2f} MB)")
        st.dataframe(pd.DataFrame(pdf_files), use_container_width=True)
    else:
        st.warning(f"No PDF files found in {pdf_dir}. Please add PDFs to this directory.")
    
    # Upload PDFs directly through Streamlit
    uploaded_pdfs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_pdfs:
        for uploaded_pdf in uploaded_pdfs:
            try:
                # Save the uploaded file to the PDF directory
                file_path = os.path.join(pdf_dir, uploaded_pdf.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_pdf.getbuffer())
                st.success(f"Saved {uploaded_pdf.name} to {pdf_dir}")
                
                # Add to the list if not already there
                if not any(pdf["File"] == uploaded_pdf.name for pdf in pdf_files):
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    pdf_files.append({"File": uploaded_pdf.name, "Size (MB)": f"{file_size:.2f}"})
                    pdf_stats["Total Files"] += 1
                    pdf_stats["Total Size (MB)"] += file_size
                
                # Update the dataframe
                st.dataframe(pd.DataFrame(pdf_files), use_container_width=True)
            except Exception as e:
                st.error(f"Error saving {uploaded_pdf.name}: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model initialization
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("### 2. Model Setup")
    
    # Load existing model if uploaded
    if uploaded_model is not None:
        try:
            # Save the uploaded model to a temporary file
            with open(save_filename, "wb") as f:
                f.write(uploaded_model.getbuffer())
            
            # Load the model
            rag_bot = HybridRAGBot.load(save_filename, pdf_directory=pdf_dir, auth_token=hf_token)
            
            st.session_state.rag_bot = rag_bot
            st.session_state.model_loaded = True
            
            # Get document stats
            st.session_state.document_stats = {
                "Total Chunks": len(rag_bot.chunks),
                "Unique Documents": len(set(chunk.source for chunk in rag_bot.chunks)),
                "Processed": rag_bot.processed
            }
            
            st.success(f"Successfully loaded model with {st.session_state.document_stats['Total Chunks']} chunks from {st.session_state.document_stats['Unique Documents']} documents")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
    
    # Initialize new model
    col1, col2 = st.columns(2)
    
    with col1:
        initialize_button = st.button("Initialize New Model", use_container_width=True)
    
    with col2:
        if st.session_state.rag_bot is not None:
            save_button = st.button("Save Current Model", use_container_width=True)
            if save_button:
                try:
                    if st.session_state.rag_bot.save(save_filename):
                        st.success(f"Model saved to {save_filename}")
                    else:
                        st.error("Error saving model")
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
    
    if initialize_button:
        try:
            with st.spinner("Initializing new model..."):
                # Initialize the RAG bot
                rag_bot = HybridRAGBot(
                    pdf_directory=pdf_dir,
                    model_name=model_name,
                    embedding_model=embedding_model,
                    auth_token=hf_token,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k
                )
                
                st.session_state.rag_bot = rag_bot
                st.session_state.model_loaded = True
                st.success("Model initialized successfully")
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process documents
    st.markdown('<div class="highlight">', unsafe_allow_html=True)
    st.markdown("### 3. Process Documents")
    
    if st.session_state.rag_bot is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            process_button = st.button("Process Documents", use_container_width=True, 
                                      disabled=st.session_state.is_processing)
        
        with col2:
            force_reprocess = st.checkbox("Force Reprocessing", 
                                         help="Process documents even if they have been processed before")
        
        if process_button:
            st.session_state.is_processing = True
            
            try:
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process the documents
                status_text.text("Processing documents...")
                
                # Since we can't modify the original code to report progress, 
                # we'll use a simple animation to indicate processing
                start_time = time.time()
                
                # We need to run the processing in the main thread since it's not safe to access TensorFlow/PyTorch models across threads
                st.session_state.rag_bot.process_documents(force_reprocess=force_reprocess)
                
                # Get document stats
                st.session_state.document_stats = {
                    "Total Chunks": len(st.session_state.rag_bot.chunks),
                    "Unique Documents": len(set(chunk.source for chunk in st.session_state.rag_bot.chunks)),
                    "Processed": st.session_state.rag_bot.processed
                }
                
                elapsed_time = time.time() - start_time
                
                # Update progress and status
                progress_bar.progress(100)
                status_text.text(f"Processing completed in {elapsed_time:.2f} seconds")
                
                # Display results
                st.success(f"Successfully processed {st.session_state.document_stats['Unique Documents']} documents into {st.session_state.document_stats['Total Chunks']} chunks")
                
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
            
            finally:
                st.session_state.is_processing = False
    else:
        st.warning("Please initialize or load a model first")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Document stats
    if st.session_state.document_stats:
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### Document Statistics")
        
        # Format the stats
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Documents", st.session_state.document_stats["Unique Documents"])
        col2.metric("Total Chunks", st.session_state.document_stats["Total Chunks"])
        col3.metric("Processed", "Yes" if st.session_state.document_stats["Processed"] else "No")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Query tab
with tabs[1]:
    st.markdown('<p class="section-header">Query Documents</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.warning("Please load or initialize a model first in the Setup tab")
    elif not st.session_state.document_stats or not st.session_state.document_stats.get("Processed", False):
        st.warning("Please process documents first in the Setup tab")
    else:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        # Query input
        query = st.text_area("Enter your question about the documents", 
                            height=100, 
                            placeholder="What is quantum entanglement?")
        
        # Query button
        if st.button("Ask Question", use_container_width=True):
            if not query:
                st.warning("Please enter a question")
            else:
                try:
                    with st.spinner("Generating answer..."):
                        start_time = time.time()
                        
                        # Get answer from RAG bot
                        result = st.session_state.rag_bot.answer_query(query)
                        
                        elapsed_time = time.time() - start_time
                        
                        # Add to query history
                        query_item = {
                            "query": query,
                            "answer": result["answer"],
                            "sources": result["sources"],
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "time_taken": f"{elapsed_time:.2f} sec"
                        }
                        st.session_state.query_history.append(query_item)
                        
                        # Display answer
                        st.markdown("### Answer")
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(result["answer"])
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display sources
                        st.markdown("### Sources")
                        for i, source in enumerate(result["sources"]):
                            with st.expander(f"Source {i+1}: {source['source']} (Page {source['page']})"):
                                # Find the full chunk text
                                chunk_text = "Source text not available"
                                for chunk in st.session_state.rag_bot.chunks:
                                    if (chunk.source == source["source"] and 
                                        chunk.page == source["page"] and 
                                        chunk.chunk_id == source["chunk_id"]):
                                        chunk_text = chunk.text
                                        break
                                
                                st.markdown(f"**Score:** {source['score']:.4f}")
                                st.markdown(f"**Text:**\n{chunk_text}")
                        
                        st.success(f"Query processed in {elapsed_time:.2f} seconds")
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Document Explorer tab
with tabs[2]:
    st.markdown('<p class="section-header">Document Explorer</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_loaded or not st.session_state.document_stats:
        st.warning("Please load or initialize a model first in the Setup tab")
    else:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        
        # Get unique documents
        if st.session_state.rag_bot and st.session_state.rag_bot.chunks:
            unique_docs = sorted(set(chunk.source for chunk in st.session_state.rag_bot.chunks))
            
            # Document selector
            selected_doc = st.selectbox("Select Document", unique_docs)
            
            if selected_doc:
                # Get pages in the document
                doc_pages = sorted(set(chunk.page for chunk in st.session_state.rag_bot.chunks 
                                    if chunk.source == selected_doc))
                
                # Page selector
                selected_page = st.selectbox("Select Page", doc_pages)
                
                if selected_page:
                    # Get chunks for this page
                    page_chunks = [chunk for chunk in st.session_state.rag_bot.chunks 
                                  if chunk.source == selected_doc and chunk.page == selected_page]
                    
                    # Display chunks
                    st.markdown(f"### Document: {selected_doc}, Page: {selected_page}")
                    st.write(f"Found {len(page_chunks)} chunks on this page")
                    
                    for i, chunk in enumerate(page_chunks):
                        with st.expander(f"Chunk {i+1} (ID: {chunk.chunk_id})"):
                            st.markdown(chunk.text)
        else:
            st.warning("No document chunks available. Please process documents first.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Query History tab
with tabs[3]:
    st.markdown('<p class="section-header">Query History</p>', unsafe_allow_html=True)
    
    if st.session_state.query_history:
        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        
        # Display history in reverse order (newest first)
        for i, item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"Query {len(st.session_state.query_history) - i}: {item['query']} ({item['timestamp']})"):
                st.markdown(f"**Time taken:** {item['time_taken']}")
                st.markdown("**Answer:**")
                st.markdown(item['answer'])
                
                st.markdown("**Sources:**")
                sources_df = pd.DataFrame([
                    {"Document": s['source'], "Page": s['page'], "Score": f"{s['score']:.4f}"}
                    for s in item['sources']
                ])
                st.dataframe(sources_df, use_container_width=True)
        
        # Clear history button
        if st.button("Clear Query History"):
            st.session_state.query_history = []
            st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No queries yet. Ask questions in the 'Query Documents' tab to build history.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Hybrid ML-LLM RAG System for Large Technical PDFs</p>
    <p>Features: Hybrid Retrieval (TF-IDF + Dense Embeddings), Technical Content Chunking, Gemma 3 Generation</p>
</div>
""", unsafe_allow_html=True)
