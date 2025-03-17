import os
import time
import pickle
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import tempfile
import base64
from typing import List, Dict, Tuple, Optional, Any

# Import the RAG Bot (assume hybrid_rag.py is in the same directory)
from hybrid_rag import HybridRAGBot, DocumentChunk, logger

# Configure Streamlit page
st.set_page_config(
    page_title="Hybrid RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 600;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 500;
        color: #0D47A1;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stProgress .st-bo {
        background-color: #1E88E5;
    }
    .source-title {
        font-weight: 600;
        color: #1E88E5;
    }
    .chunk-text {
        font-size: 0.9rem;
        max-height: 200px;
        overflow-y: auto;
    }
    .sidebar .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag_bot" not in st.session_state:
    st.session_state.rag_bot = None
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "model_saved" not in st.session_state:
    st.session_state.model_saved = False

# Sidebar
st.sidebar.markdown("<div class='main-header'>Hybrid RAG</div>", unsafe_allow_html=True)
st.sidebar.markdown("A robust Retrieval-Augmented Generation system for large technical PDFs")

# PDF Upload section
st.sidebar.markdown("<div class='sub-header'>📄 Document Upload</div>", unsafe_allow_html=True)
uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

# Configuration options
st.sidebar.markdown("<div class='sub-header'>⚙️ Configuration</div>", unsafe_allow_html=True)
chunk_size = st.sidebar.slider("Chunk Size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk Overlap", 50, 500, 200, 10)
top_k = st.sidebar.slider("Number of chunks to retrieve", 1, 10, 5, 1)

# Model weights
st.sidebar.markdown("<div class='sub-header'>🧠 Retrieval Weights</div>", unsafe_allow_html=True)
tfidf_weight = st.sidebar.slider("TF-IDF Weight", 0.0, 1.0, 0.3, 0.1)
dense_weight = st.sidebar.slider("Dense Embeddings Weight", 0.0, 1.0, 0.7, 0.1)

# Hugging Face Token
st.sidebar.markdown("<div class='sub-header'>🔑 API Configuration</div>", unsafe_allow_html=True)
hf_token = st.sidebar.text_input("Hugging Face Token", 
                              value="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",
                              type="password")

# Optional: Model selection (if implementing multiple model support)
model_options = {
    "Gemma 3 (1B)": "google/gemma-3-1b-it", 
    "Mistral (7B)": "mistralai/Mistral-7B-Instruct-v0.2",
    "Flan T5 (Base)": "google/flan-t5-base"
}
selected_model = st.sidebar.selectbox("LLM Model", list(model_options.keys()))
model_name = model_options[selected_model]

# Main content area
st.markdown("<div class='main-header'>Hybrid ML-LLM RAG System</div>", unsafe_allow_html=True)
st.markdown("This system combines TF-IDF and dense vector embeddings with a language model to answer questions about your technical documents.")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Upload & Process", "Query System", "Analysis", "Settings"])

with tab1:
    st.markdown("<div class='sub-header'>Document Processing</div>", unsafe_allow_html=True)
    
    # Setup temporary directory for PDFs
    pdf_dir = os.path.join(tempfile.gettempdir(), "hybrid_rag_pdfs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    # Function to process files
    def process_files():
        if not uploaded_files:
            st.error("Please upload at least one PDF file first.")
            return
            
        # Save files to temp directory
        for uploaded_file in uploaded_files:
            file_path = os.path.join(pdf_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Status updates through a callback
        def status_callback(message, progress):
            status_text.text(message)
            progress_bar.progress(progress)
        
        try:
            status_callback("Initializing RAG Bot...", 0.1)
            
            # Initialize RAG Bot
            st.session_state.rag_bot = HybridRAGBot(
                pdf_directory=pdf_dir,
                model_name=model_name,
                auth_token=hf_token,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                top_k=top_k
            )
            
            # Override retriever weights
            st.session_state.rag_bot.retriever.tfidf_weight = tfidf_weight
            st.session_state.rag_bot.retriever.dense_weight = dense_weight
            
            status_callback("Processing documents...", 0.3)
            
            # Process documents
            st.session_state.rag_bot.process_documents()
            
            status_callback("Processing complete!", 1.0)
            st.session_state.processing_complete = True
            
            # Show summary
            st.success(f"Successfully processed {len(uploaded_files)} PDF files")
            st.info(f"Created {len(st.session_state.rag_bot.chunks)} chunks from the documents")
            
            # Display file summary
            file_stats = {}
            for chunk in st.session_state.rag_bot.chunks:
                if chunk.source not in file_stats:
                    file_stats[chunk.source] = {"chunks": 0, "pages": set()}
                file_stats[chunk.source]["chunks"] += 1
                file_stats[chunk.source]["pages"].add(chunk.page)
            
            # Create a DataFrame for file stats
            stats_data = []
            for filename, stats in file_stats.items():
                stats_data.append({
                    "Filename": filename,
                    "Chunks": stats["chunks"],
                    "Pages": len(stats["pages"])
                })
            
            stats_df = pd.DataFrame(stats_data)
            st.dataframe(stats_df)
            
            # Create a visualization of chunks per file
            fig = px.bar(stats_df, x="Filename", y="Chunks", title="Number of Chunks per Document")
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            status_callback("Error occurred during processing.", 0)
    
    # Save/Load Model
    col1, col2 = st.columns(2)
    
    def save_model():
        if st.session_state.rag_bot is None or not st.session_state.processing_complete:
            st.error("Please process documents first before saving the model.")
            return
        
        try:
            # Save model to file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                st.session_state.rag_bot.save(tmp_file.name)
                
                # Create download link
                with open(tmp_file.name, 'rb') as f:
                    model_bytes = f.read()
                
                b64 = base64.b64encode(model_bytes).decode()
                href = f'<a href="data:file/pickle;base64,{b64}" download="hybrid_rag_model.pkl">Download Trained Model</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.session_state.model_saved = True
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
    
    def load_model():
        uploaded_model = st.file_uploader("Upload a saved model (.pkl file)", type=['pkl'])
        if uploaded_model:
            try:
                # Save the uploaded model to a temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                    tmp_file.write(uploaded_model.getbuffer())
                    model_path = tmp_file.name
                
                # Load the model
                st.session_state.rag_bot = HybridRAGBot.load(model_path, auth_token=hf_token)
                st.session_state.processing_complete = True
                st.success("Model loaded successfully!")
                
                # Show model statistics
                st.info(f"Model contains {len(st.session_state.rag_bot.chunks)} chunks from {len(set([chunk.source for chunk in st.session_state.rag_bot.chunks]))} documents")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
    
    # Process and save/load buttons
    with col1:
        if st.button("Process Documents", key="process_btn", use_container_width=True):
            process_files()
    
    with col2:
        if st.button("Save Model", key="save_btn", disabled=(not st.session_state.processing_complete), use_container_width=True):
            save_model()
            
    if not st.session_state.processing_complete:
        with st.expander("Load Existing Model"):
            load_model()

with tab2:
    st.markdown("<div class='sub-header'>Ask Questions</div>", unsafe_allow_html=True)
    
    if not st.session_state.processing_complete:
        st.warning("Please process documents first in the 'Upload & Process' tab.")
    else:
        # Query input
        query = st.text_input("Ask a question about your documents:", key="query_input", 
                             placeholder="e.g., What is quantum entanglement?")
        
        debug_mode = st.checkbox("Show detailed information about retrieved passages")
        
        if st.button("Submit Question", key="query_btn"):
            if not query:
                st.error("Please enter a question.")
            else:
                with st.spinner("Generating answer..."):
                    try:
                        # Time the query
                        start_time = time.time()
                        result = st.session_state.rag_bot.answer_query(query)
                        query_time = time.time() - start_time
                        
                        # Store in history
                        st.session_state.query_history.append({
                            "query": query,
                            "answer": result["answer"],
                            "sources": result["sources"],
                            "time": query_time
                        })
                        
                        # Display the answer
                        st.markdown("### Answer")
                        st.markdown(result["answer"])
                        st.info(f"Query processed in {query_time:.2f} seconds")
                        
                        # Display sources
                        st.markdown("### Sources")
                        
                        if debug_mode:
                            for i, source in enumerate(result["sources"]):
                                with st.expander(f"Source {i+1}: {source['source']} (Page {source['page']}) - Score: {source['score']:.4f}"):
                                    # Find the chunk text
                                    chunk_text = ""
                                    for chunk in st.session_state.rag_bot.chunks:
                                        if (chunk.source == source['source'] and 
                                            chunk.page == source['page'] and 
                                            chunk.chunk_id == source['chunk_id']):
                                            chunk_text = chunk.text
                                            break
                                    
                                    st.markdown(f"<div class='chunk-text'>{chunk_text}</div>", unsafe_allow_html=True)
                        else:
                            # Simple source display
                            source_text = "<ul>"
                            for source in result["sources"][:3]:  # Show first 3 sources
                                source_text += f"<li>{source['source']} (Page {source['page']})</li>"
                            source_text += "</ul>"
                            st.markdown(source_text, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Error generating answer: {str(e)}")
        
        # Display query history
        if st.session_state.query_history:
            st.markdown("### Query History")
            for i, hist_item in enumerate(reversed(st.session_state.query_history[-5:])):  # Show last 5 queries
                with st.expander(f"Q: {hist_item['query']}"):
                    st.markdown(f"**Answer:** {hist_item['answer']}")
                    st.markdown("**Sources:**")
                    for source in hist_item['sources'][:3]:
                        st.markdown(f"- {source['source']} (Page {source['page']})")
                    st.text(f"Time: {hist_item['time']:.2f} seconds")

with tab3:
    st.markdown("<div class='sub-header'>System Analysis</div>", unsafe_allow_html=True)
    
    if not st.session_state.processing_complete:
        st.warning("Please process documents first in the 'Upload & Process' tab.")
    else:
        # Create subtabs for different analyses
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["Document Analysis", "Retrieval Performance", "Query Analysis"])
        
        with analysis_tab1:
            st.markdown("### Document Statistics")
            
            # Get document statistics
            doc_sources = [chunk.source for chunk in st.session_state.rag_bot.chunks]
            doc_pages = [chunk.page for chunk in st.session_state.rag_bot.chunks]
            
            # Count chunks per document
            doc_counts = pd.Series(doc_sources).value_counts().reset_index()
            doc_counts.columns = ["Document", "Number of Chunks"]
            
            # Plot document distribution
            fig1 = px.bar(doc_counts, x="Document", y="Number of Chunks", 
                         title="Distribution of Chunks Across Documents")
            st.plotly_chart(fig1, use_container_width=True)
            
            # Count chunks per page
            page_data = pd.DataFrame({
                "Document": doc_sources,
                "Page": doc_pages
            })
            page_counts = page_data.groupby(["Document", "Page"]).size().reset_index()
            page_counts.columns = ["Document", "Page", "Number of Chunks"]
            
            # Plot page distribution for the document with most pages
            top_doc = doc_counts.iloc[0]["Document"]
            doc_page_data = page_counts[page_counts["Document"] == top_doc]
            
            fig2 = px.bar(doc_page_data, x="Page", y="Number of Chunks", 
                         title=f"Chunks Per Page for {top_doc}")
            st.plotly_chart(fig2, use_container_width=True)
            
            # Display avg chunks per document
            st.metric("Average Chunks per Document", 
                     f"{len(st.session_state.rag_bot.chunks) / len(doc_counts):.1f}")
        
        with analysis_tab2:
            st.markdown("### Retrieval Analysis")
            
            # Compare different retrieval methods
            st.markdown("#### Retrieval Comparison")
            
            test_query = st.text_input("Enter a test query:", "What is quantum entanglement?", 
                                      key="test_query_input")
            
            if st.button("Compare Retrieval Methods", key="compare_btn"):
                with st.spinner("Comparing retrieval methods..."):
                    try:
                        # Get results from TF-IDF
                        tfidf_results = st.session_state.rag_bot.retriever.tfidf_retriever.query(test_query, top_k=3)
                        
                        # Get results from Dense
                        dense_results = st.session_state.rag_bot.retriever.dense_retriever.query(test_query, top_k=3)
                        
                        # Get results from Hybrid
                        hybrid_results = st.session_state.rag_bot.retriever.query(test_query, top_k=3)
                        
                        # Display comparison
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**TF-IDF Results**")
                            for i, (chunk, score) in enumerate(tfidf_results):
                                st.markdown(f"{i+1}. {chunk.source} (Page {chunk.page})")
                                st.progress(float(score))
                        
                        with col2:
                            st.markdown("**Dense Embedding Results**")
                            for i, (chunk, score) in enumerate(dense_results):
                                st.markdown(f"{i+1}. {chunk.source} (Page {chunk.page})")
                                st.progress(float(score))
                        
                        with col3:
                            st.markdown("**Hybrid Results**")
                            for i, (chunk, score) in enumerate(hybrid_results):
                                st.markdown(f"{i+1}. {chunk.source} (Page {chunk.page})")
                                st.progress(float(score))
                        
                        # Calculate overlap
                        tfidf_ids = set([(chunk.source, chunk.page) for chunk, _ in tfidf_results])
                        dense_ids = set([(chunk.source, chunk.page) for chunk, _ in dense_results])
                        hybrid_ids = set([(chunk.source, chunk.page) for chunk, _ in hybrid_results])
                        
                        # Overlap metrics
                        tfidf_dense_overlap = len(tfidf_ids.intersection(dense_ids))
                        tfidf_hybrid_overlap = len(tfidf_ids.intersection(hybrid_ids))
                        dense_hybrid_overlap = len(dense_ids.intersection(hybrid_ids))
                        
                        # Display metrics
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        metrics_col1.metric("TF-IDF/Dense Overlap", f"{tfidf_dense_overlap}/3")
                        metrics_col2.metric("TF-IDF/Hybrid Overlap", f"{tfidf_hybrid_overlap}/3")
                        metrics_col3.metric("Dense/Hybrid Overlap", f"{dense_hybrid_overlap}/3")
                    
                    except Exception as e:
                        st.error(f"Error comparing retrieval methods: {str(e)}")
        
        with analysis_tab3:
            st.markdown("### Query Analysis")
            
            if not st.session_state.query_history:
                st.info("No queries have been made yet. Ask some questions in the Query tab.")
            else:
                # Analyze query performance
                query_data = pd.DataFrame(st.session_state.query_history)
                
                # Query response time
                fig3 = px.line(query_data, y="time", title="Query Response Time", 
                              labels={"index": "Query Number", "time": "Time (seconds)"})
                st.plotly_chart(fig3, use_container_width=True)
                
                # Display metrics
                avg_time = query_data["time"].mean()
                st.metric("Average Response Time", f"{avg_time:.2f} seconds")
                
                # Export results
                if st.button("Export Query History", key="export_btn"):
                    # Create a dataframe with simplified sources
                    export_data = []
                    for item in st.session_state.query_history:
                        export_data.append({
                            "Query": item["query"],
                            "Answer": item["answer"],
                            "Sources": ", ".join([f"{s['source']} (Page {s['page']})" for s in item["sources"][:3]]),
                            "Time (seconds)": item["time"]
                        })
                    
                    export_df = pd.DataFrame(export_data)
                    
                    # Convert to CSV string
                    csv = export_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="query_history.csv">Download Query History CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)

with tab4:
    st.markdown("<div class='sub-header'>Advanced Settings</div>", unsafe_allow_html=True)
    
    # Advanced settings for retrieval and processing
    st.markdown("### Retrieval Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### TF-IDF Settings")
        use_svd = st.checkbox("Use SVD for dimensionality reduction", value=True)
        n_components = st.slider("SVD Components", 50, 500, 100, 10, 
                                disabled=not use_svd)
        min_df = st.slider("Minimum Document Frequency", 1, 10, 2, 1)
        max_df = st.slider("Maximum Document Frequency", 0.5, 1.0, 0.95, 0.05)
    
    with col2:
        st.markdown("#### Dense Retriever Settings")
        embedding_models = {
            "MiniLM (Small, Fast)": "all-MiniLM-L6-v2",
            "MPNet (Medium, Balanced)": "all-mpnet-base-v2",
            "MiniLM (Multilingual)": "paraphrase-multilingual-MiniLM-L12-v2"
        }
        embedding_model = st.selectbox("Embedding Model", list(embedding_models.keys()))
        batch_size = st.slider("Embedding Batch Size", 16, 128, 64, 8)
    
    st.markdown("### Processing Settings")
    
    max_workers = st.slider("Parallel Processing Workers", 1, 8, 4, 1)
    
    if st.button("Apply Settings", key="apply_settings"):
        st.session_state.new_settings = {
            "use_svd": use_svd,
            "n_components": n_components,
            "min_df": min_df,
            "max_df": max_df,
            "embedding_model": embedding_models[embedding_model],
            "batch_size": batch_size,
            "max_workers": max_workers
        }
        st.success("Settings saved. They will be applied when you process documents again.")

# Footer
st.markdown("---")
st.markdown("Hybrid ML-LLM RAG System | Built for processing large technical PDFs")


# Run the application with: streamlit run streamlit_app.py
