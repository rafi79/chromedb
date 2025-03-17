# Minimal Streamlit app for RAG system - avoiding session state issues
import streamlit as st
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import io

# Set page config as the first Streamlit command
st.set_page_config(
    page_title="Simple RAG System",
    page_icon="📚",
    layout="wide"
)

# Define a minimal RAG bot class (if import fails)
class SimpleRAGBot:
    def __init__(self, pdf_directory="./pdfs"):
        self.pdf_directory = pdf_directory
        self.chunks = []
        self.processed = False
    
    def process_documents(self):
        """Process documents in the directory."""
        # Find all PDF files
        pdf_files = []
        for root, _, files in os.walk(self.pdf_directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        # Create simple chunks (in a real system, this would extract text)
        self.chunks = []
        for pdf_path in pdf_files:
            filename = os.path.basename(pdf_path)
            self.chunks.append({
                "text": f"Content from {filename}",
                "source": filename,
                "page": 1,
                "title": os.path.splitext(filename)[0]
            })
        
        self.processed = True
        return len(pdf_files), len(self.chunks)
    
    def answer_query(self, query):
        """Answer a query based on document content."""
        if not self.processed:
            return {
                "query": query,
                "answer": "Please process documents first.",
                "sources": []
            }
        
        # Create a dummy answer with citations
        sources = self.chunks[:3]  # Use up to 3 chunks as sources
        
        answer = f"Here's information about '{query}':\n\n"
        answer += "1. First key point relevant to your question. [1]\n"
        answer += "2. Additional information from the documents. [2]\n"
        answer += "3. Further context and details. [1, 3]\n\n"
        answer += "References:\n"
        
        for i, source in enumerate(sources):
            answer += f"[{i+1}] {source['title']} (Page {source['page']})\n"
        
        formatted_sources = []
        for i, source in enumerate(sources):
            formatted_sources.append({
                "source": source["source"],
                "page": source["page"],
                "title": source["title"],
                "score": 0.9 - (i * 0.1)
            })
        
        return {
            "query": query,
            "answer": answer,
            "sources": formatted_sources
        }

# Try to import the actual RAG system (if available)
try:
    # Attempt to import the RAG system
    from hybrid_rag_system import EnhancedHybridRAGBot
    RAG_AVAILABLE = True
except ImportError:
    # Fall back to the simple implementation
    EnhancedHybridRAGBot = SimpleRAGBot
    RAG_AVAILABLE = False

# Main app title
st.title("Simple RAG System with Citations")
st.markdown("Upload PDFs, process them, and ask questions to get cited answers.")

# Create two columns: settings and main content
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Settings")
    
    # Configuration options
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
    top_k = st.slider("Top K Results", 3, 10, 5, 1)
    
    st.markdown("---")
    
    # Show version info
    st.caption("RAG System v1.0")
    if RAG_AVAILABLE:
        st.success("Using EnhancedHybridRAGBot")
    else:
        st.warning("Using SimpleRAGBot (fallback)")

with col2:
    # File upload and processing
    st.subheader("1. Upload Documents")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    # Initialize variables
    temp_dir = None
    rag_bot = None
    total_docs = 0
    total_chunks = 0
    
    # Process files if uploaded
    if uploaded_files:
        # Create a temporary directory for the files
        temp_dir = tempfile.mkdtemp()
        
        # Save the uploaded files
        pdf_files = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            pdf_files.append(file_path)
        
        st.success(f"Uploaded {len(pdf_files)} PDF documents")
        
        # Process button
        st.subheader("2. Process Documents")
        if st.button("Process Documents"):
            try:
                with st.spinner("Processing documents..."):
                    # Initialize RAG bot
                    rag_bot = EnhancedHybridRAGBot(
                        pdf_directory=temp_dir,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        top_k=top_k
                    )
                    
                    # Process documents
                    total_docs, total_chunks = rag_bot.process_documents()
                    
                st.success(f"✅ Processed {total_docs} documents into {total_chunks} chunks")
            except Exception as e:
                st.error(f"Error processing documents: {str(e)}")
                st.info("Creating a simple RAG bot with basic functionality")
                rag_bot = SimpleRAGBot(pdf_directory=temp_dir)
                total_docs, total_chunks = rag_bot.process_documents()
    
    # Query interface (only show if we have a RAG bot)
    if rag_bot and rag_bot.processed:
        st.markdown("---")
        st.subheader("3. Ask Questions")
        query = st.text_input("Enter your question:", placeholder="What is quantum entanglement?")
        
        if query and st.button("Ask"):
            with st.spinner("Generating answer..."):
                result = rag_bot.answer_query(query)
            
            # Display the answer
            st.markdown("### Answer")
            st.markdown(f"**Query:** {result['query']}")
            st.markdown(result['answer'])
            
            # Display sources
            st.markdown("### Sources")
            for i, source in enumerate(result['sources']):
                title = source.get('title', source['source'])
                if 'author' in source and source['author'] != "Unknown":
                    st.markdown(f"{i+1}. {title} by {source['author']} (Page {source['page']}) - Score: {source['score']:.2f}")
                else:
                    st.markdown(f"{i+1}. {title} (Page {source['page']}) - Score: {source['score']:.2f}")
            
            # Create a visualization of source relevance
            if len(result['sources']) > 0:
                try:
                    st.markdown("### Source Relevance")
                    source_names = [s.get('title', s['source']) for s in result['sources'][:5]]
                    source_scores = [s['score'] for s in result['sources'][:5]]
                    
                    chart_data = pd.DataFrame({
                        'Source': source_names,
                        'Relevance': source_scores
                    })
                    
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.barplot(x='Relevance', y='Source', data=chart_data, ax=ax)
                    ax.set_xlim(0, 1)
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not create visualization: {str(e)}")
