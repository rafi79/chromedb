import streamlit as st
import os
import time
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tempfile
import threading
import queue
import io

# Import from your existing system - adjust import paths as needed
# You might need to place this script in the same directory as your main code
# or adjust the import paths accordingly
try:
    from hybrid_rag_system import (
        EnhancedHybridRAGBot, HF_TOKEN, MODEL_NAME, EMBEDDING_MODEL_NAME, 
        CHUNK_SIZE, CHUNK_OVERLAP
    )
except ImportError:
    # Fallback imports for demonstration
    st.error("Failed to import from hybrid_rag_system. Using placeholder values.")
    HF_TOKEN = "your_default_token"
    MODEL_NAME = "google/gemma-3-1b-it"
    EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    # You'll need to add code for EnhancedHybridRAGBot class here or adjust imports

# Set page config
st.set_page_config(
    page_title="Enhanced RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .citation {
        background-color: #e0f7fa;
        border-left: 3px solid #00acc1;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        gap: 10px;
    }
    .metric-box {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        flex: 1;
        text-align: center;
    }
    .answer-container {
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        margin-top: 20px;
    }
    .point {
        margin-bottom: 10px;
    }
    .references {
        margin-top: 20px;
        padding-top: 10px;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "rag_bot" not in st.session_state:
    st.session_state.rag_bot = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = []
if "temp_dir" not in st.session_state:
    st.session_state.temp_dir = None
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None
if "processing_message" not in st.session_state:
    st.session_state.processing_message = ""
if "processing_progress" not in st.session_state:
    st.session_state.processing_progress = 0
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = None

# Background processing function
def process_documents_thread(temp_dir, status_queue, progress_queue):
    try:
        # Create RAG bot
        rag_bot = EnhancedHybridRAGBot(
            pdf_directory=temp_dir,
            auth_token=st.session_state.hf_token,
            chunk_size=st.session_state.chunk_size,
            chunk_overlap=st.session_state.chunk_overlap,
            top_k=st.session_state.top_k
        )
        
        # Process documents
        status_queue.put("Extracting text from PDFs...")
        progress_queue.put(0.2)
        
        rag_bot.process_documents()
        progress_queue.put(0.8)
        
        # Calculate document statistics
        doc_stats = {}
        doc_stats["total_chunks"] = len(rag_bot.chunks)
        
        # Count documents and pages
        sources = {}
        total_pages = 0
        for chunk in rag_bot.chunks:
            if chunk.source not in sources:
                sources[chunk.source] = set()
            sources[chunk.source].add(chunk.page)
            total_pages = max(total_pages, chunk.page)
        
        doc_stats["total_documents"] = len(sources)
        doc_stats["total_pages"] = sum([len(pages) for pages in sources.values()])
        
        # Calculate average chunks per document
        if doc_stats["total_documents"] > 0:
            doc_stats["avg_chunks_per_doc"] = doc_stats["total_chunks"] / doc_stats["total_documents"]
        else:
            doc_stats["avg_chunks_per_doc"] = 0
            
        # Get document titles
        doc_stats["document_titles"] = []
        for chunk in rag_bot.chunks:
            if hasattr(chunk, 'title'):
                doc_stats["document_titles"].append(chunk.title)
        doc_stats["document_titles"] = list(set(doc_stats["document_titles"]))
        
        # Save to session state
        status_queue.put("Finalizing...")
        progress_queue.put(0.9)
        
        status_queue.put("complete")
        progress_queue.put(1.0)
        
        # Return the RAG bot and stats
        return rag_bot, doc_stats
        
    except Exception as e:
        status_queue.put(f"Error: {str(e)}")
        return None, None

def format_answer_html(result):
    """Format the answer with citation highlights in HTML."""
    answer = result["answer"]
    
    # Split by points if the answer is in a point-by-point format
    if "\n1. " in answer:
        parts = answer.split("\n\n")
        formatted_parts = []
        
        for part in parts:
            if part.startswith("1. ") or part.strip().startswith("1. "):
                # This is the points section
                points = part.split("\n")
                formatted_points = []
                
                for point in points:
                    if point.strip():
                        # Highlight citations
                        highlighted_point = point
                        if "[" in point and "]" in point:
                            for i in range(1, 10):  # Assuming up to 9 citations
                                citation = f"[{i}]"
                                if citation in highlighted_point:
                                    highlighted_point = highlighted_point.replace(
                                        citation, 
                                        f"<span style='background-color: #e0f7fa; padding: 2px 4px; border-radius: 3px;'>{citation}</span>"
                                    )
                        
                        formatted_points.append(f"<div class='point'>{highlighted_point}</div>")
                
                formatted_parts.append("".join(formatted_points))
            elif part.startswith("References:"):
                # References section
                formatted_parts.append(f"<div class='references'>{part}</div>")
            else:
                # Introduction or conclusion
                formatted_parts.append(f"<p>{part}</p>")
        
        return "".join(formatted_parts)
    else:
        # Not in a clear point format, do basic formatting
        formatted = answer.replace("\n\n", "</p><p>").replace("\n", "<br>")
        return f"<p>{formatted}</p>"

def save_answer_as_pdf(result):
    """Save the answer with citations as a PDF file."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='Point',
            parent=styles['Normal'],
            spaceAfter=6,
            leftIndent=20,
            firstLineIndent=-20
        ))
        styles.add(ParagraphStyle(
            name='Citation',
            parent=styles['Normal'],
            backColor=colors.lightblue,
            borderPadding=5,
            borderWidth=0,
            borderRadius=5,
            spaceAfter=6
        ))
        
        # Build document
        elements = []
        
        # Add title
        elements.append(Paragraph(f"Query: {result['query']}", styles['Heading1']))
        elements.append(Spacer(1, 12))
        
        # Add answer
        answer_text = result['answer']
        
        # Split by lines to format points
        if "\n1. " in answer_text:
            # Add introduction
            intro_end = answer_text.find("\n\n1.")
            if intro_end == -1:
                intro_end = answer_text.find("\n1.")
            
            if intro_end != -1:
                intro = answer_text[:intro_end]
                elements.append(Paragraph(intro, styles['Normal']))
                elements.append(Spacer(1, 6))
            
            # Add points
            lines = answer_text.split("\n")
            for line in lines:
                if line.strip():
                    if line.strip().startswith(tuple([f"{i}. " for i in range(1, 10)])):
                        elements.append(Paragraph(line, styles['Point']))
                    elif line.startswith("References:"):
                        elements.append(Spacer(1, 12))
                        elements.append(Paragraph(line, styles['Heading2']))
                    elif "[" in line and "]" in line and "References:" in answer_text and line not in answer_text[:answer_text.find("References:")]:
                        # This is a reference line
                        elements.append(Paragraph(line, styles['Normal']))
                    else:
                        if not any(line.startswith(f"{i}.") for i in range(1, 10)):
                            elements.append(Paragraph(line, styles['Normal']))
        else:
            elements.append(Paragraph(answer_text, styles['Normal']))
        
        # Add sources
        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Sources:", styles['Heading2']))
        
        for source in result['sources']:
            if 'title' in source and 'author' in source and source['author'] != "Unknown":
                source_text = f"{source['title']} by {source['author']} (Page {source['page']})"
            elif 'title' in source:
                source_text = f"{source['title']} (Page {source['page']})"
            else:
                source_text = f"{source['source']} (Page {source['page']})"
                
            elements.append(Paragraph(source_text, styles['Normal']))
        
        # Build and return PDF
        doc.build(elements)
        buffer.seek(0)
        return buffer
        
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

# Sidebar for settings
with st.sidebar:
    st.title("📚 Enhanced RAG System")
    st.subheader("Settings")
    
    # Model settings
    st.session_state.hf_token = st.text_input("Hugging Face Token", value=HF_TOKEN, type="password")
    model_options = ["google/gemma-3-1b-it", "google/gemma-3-2b", "google/gemma-3-8b-it"]
    st.session_state.model = st.selectbox("LLM Model", model_options, index=0)
    
    # Processing settings
    st.session_state.chunk_size = st.slider("Chunk Size", 500, 2000, CHUNK_SIZE, 100)
    st.session_state.chunk_overlap = st.slider("Chunk Overlap", 0, 500, CHUNK_OVERLAP, 50)
    st.session_state.top_k = st.slider("Top K Results", 3, 10, 5, 1)
    
    st.markdown("---")
    
    # Status and stats display
    if st.session_state.processed:
        st.success("✅ Documents processed")
        
        if st.session_state.doc_stats:
            st.subheader("Document Statistics")
            stats = st.session_state.doc_stats
            
            st.metric("Total Documents", stats["total_documents"])
            st.metric("Total Pages", stats["total_pages"])
            st.metric("Total Chunks", stats["total_chunks"])
            st.metric("Avg. Chunks/Document", f"{stats['avg_chunks_per_doc']:.1f}")
            
            if "document_titles" in stats and stats["document_titles"]:
                with st.expander("Document Titles"):
                    for title in stats["document_titles"]:
                        st.write(f"• {title}")
    
    st.markdown("---")
    st.markdown("Developed by Your Name")
    st.markdown("Version 1.0.0")

# Main content
st.title("Enhanced RAG System with Citations")
st.markdown("""
This application uses a hybrid machine learning approach to answer questions based on your PDF documents.
Upload your PDFs, process them, and ask questions to get cited answers from your documents.
""")

# File uploader
uploaded_files = st.file_uploader("Upload PDF Documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    # Create a temporary directory to store the uploaded PDFs
    if not st.session_state.temp_dir:
        st.session_state.temp_dir = tempfile.mkdtemp()
    
    # Save uploaded files to temp directory
    st.session_state.pdf_files = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.pdf_files.append(file_path)
    
    # Display uploaded files
    if st.session_state.pdf_files:
        st.success(f"📁 {len(st.session_state.pdf_files)} PDF files uploaded")
        
        # Process button
        if not st.session_state.processed:
            if st.button("Process Documents"):
                # Create progress elements
                status_text = st.empty()
                progress_bar = st.progress(0)
                
                # Set up queues for communication
                status_queue = queue.Queue()
                progress_queue = queue.Queue()
                
                # Start background thread
                thread = threading.Thread(
                    target=lambda: setattr(
                        st.session_state, 
                        'processing_result', 
                        process_documents_thread(
                            st.session_state.temp_dir, 
                            status_queue, 
                            progress_queue
                        )
                    )
                )
                thread.start()
                
                # Monitor progress
                while thread.is_alive():
                    # Check for status updates
                    try:
                        while not status_queue.empty():
                            status = status_queue.get_nowait()
                            if status == "complete":
                                st.session_state.processing_status = "complete"
                            else:
                                status_text.text(f"Processing: {status}")
                    except queue.Empty:
                        pass
                    
                    # Check for progress updates
                    try:
                        while not progress_queue.empty():
                            progress = progress_queue.get_nowait()
                            progress_bar.progress(progress)
                    except queue.Empty:
                        pass
                    
                    time.sleep(0.1)
                
                # Processing complete
                thread.join()
                
                if st.session_state.processing_status == "complete":
                    rag_bot, doc_stats = st.session_state.processing_result
                    if rag_bot:
                        st.session_state.rag_bot = rag_bot
                        st.session_state.doc_stats = doc_stats
                        st.session_state.processed = True
                        
                        # Clear progress elements
                        status_text.empty()
                        progress_bar.empty()
                        
                        # Display success and reload page
                        st.success("✅ Documents processed successfully!")
                        st.experimental_rerun()
                    else:
                        status_text.error("❌ Processing failed. Please check logs.")
                        progress_bar.empty()

# Main query interface (if documents are processed)
if st.session_state.processed and st.session_state.rag_bot:
    st.markdown("---")
    st.subheader("Ask Questions About Your Documents")
    
    # Query input
    query = st.text_input("Enter your question:", placeholder="What is quantum entanglement?")
    
    # Query button
    if query and st.button("Ask"):
        with st.spinner("Generating answer..."):
            # Get answer from RAG bot
            result = st.session_state.rag_bot.answer_query(query)
            
            # Add to query history
            st.session_state.query_history.append(result)
        
        # Display answer
        st.markdown("### Answer")
        st.markdown(f"<div class='answer-container'>{format_answer_html(result)}</div>", unsafe_allow_html=True)
        
        # Display sources
        st.markdown("### Sources")
        source_cols = st.columns(2)
        
        with source_cols[0]:
            st.markdown("**Top Sources:**")
            for i, source in enumerate(result['sources'][:5]):
                if 'title' in source and 'author' in source and source['author'] != "Unknown":
                    st.markdown(f"<div class='citation'>{i+1}. {source['title']} by {source['author']} (Page {source['page']})</div>", unsafe_allow_html=True)
                elif 'title' in source:
                    st.markdown(f"<div class='citation'>{i+1}. {source['title']} (Page {source['page']})</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='citation'>{i+1}. {source['source']} (Page {source['page']})</div>", unsafe_allow_html=True)
        
        with source_cols[1]:
            if 'sources' in result and result['sources']:
                # Create a simple chart of source relevance
                source_names = [s.get('title', s['source']) for s in result['sources'][:5]]
                source_scores = [s['score'] for s in result['sources'][:5]]
                
                # Create a DataFrame for the chart
                chart_data = pd.DataFrame({
                    'Source': source_names,
                    'Relevance': source_scores
                })
                
                # Create bar chart
                fig, ax = plt.subplots(figsize=(5, 3))
                sns.barplot(x='Relevance', y='Source', data=chart_data, ax=ax, palette='viridis')
                ax.set_title('Source Relevance Scores')
                ax.set_xlim(0, 1)
                st.pyplot(fig)
        
        # Export options
        st.markdown("### Export")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as PDF
            pdf_buffer = save_answer_as_pdf(result)
            if pdf_buffer:
                st.download_button(
                    label="Download as PDF",
                    data=pdf_buffer,
                    file_name=f"answer_{query[:20].replace(' ', '_')}.pdf",
                    mime="application/pdf"
                )
        
        with col2:
            # Export as text
            text_content = f"Query: {result['query']}\n\nAnswer:\n{result['answer']}\n\nSources:\n"
            for i, source in enumerate(result['sources']):
                if 'title' in source and 'author' in source and source['author'] != "Unknown":
                    text_content += f"{i+1}. {source['title']} by {source['author']} (Page {source['page']})\n"
                elif 'title' in source:
                    text_content += f"{i+1}. {source['title']} (Page {source['page']})\n"
                else:
                    text_content += f"{i+1}. {source['source']} (Page {source['page']})\n"
            
            st.download_button(
                label="Download as Text",
                data=text_content,
                file_name=f"answer_{query[:20].replace(' ', '_')}.txt",
                mime="text/plain"
            )

# Query history
if st.session_state.query_history:
    st.markdown("---")
    with st.expander("Query History"):
        for i, result in enumerate(st.session_state.query_history):
            st.markdown(f"**Q{i+1}: {result['query']}**")
            st.markdown(f"<div style='background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin-bottom: 10px;'>{result['answer'][:200]}...</div>", unsafe_allow_html=True)
            st.markdown("---")

# Run the app with: streamlit run app.py
