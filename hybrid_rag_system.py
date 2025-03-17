"""
Hybrid ML-LLM RAG System for Large Technical PDFs
- Handles 8-15 large PDFs (500+ pages each)
- Uses hybrid retrieval (TF-IDF + dense embeddings)
- Optimized chunking for technical content
- Memory-efficient batch processing
- Uses Gemma 3 for generation
"""
import os
import re
import time
import math
import numpy as np
import pandas as pd
import pickle
import tempfile
from typing import List, Dict, Tuple, Optional, Union, Generator, Any
from dataclasses import dataclass
from pathlib import Path
import gc
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
PDF_DIRECTORY = "./pdfs"
HF_TOKEN = "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"  # Default token, should be replaced
MODEL_NAME = "google/gemma-3-1b-it"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Lighter model for embeddings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_WORKERS = 4
BATCH_SIZE = 64

# PDF processing
try:
    import PyPDF2
except ImportError:
    logger.warning("PyPDF2 not available. PDF processing will not work.")

# ML and embeddings
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available. TF-IDF retrieval will not work.")
    SKLEARN_AVAILABLE = False

# Try to import sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence_transformers not available. Will use TF-IDF only for embeddings.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Try to import faiss for vector storage
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available. Will use numpy for vector similarity.")
    FAISS_AVAILABLE = False

# Try to import torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Will affect model performance.")
    TORCH_AVAILABLE = False

# Hugging Face integration
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from huggingface_hub import login
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers not available. Will not be able to use Gemma 3 LLM.")
    TRANSFORMERS_AVAILABLE = False


@dataclass
class DocumentChunk:
    """Class to represent a chunk of text from a PDF document."""
    text: str
    source: str
    page: int
    chunk_id: int
    embedding: Optional[np.ndarray] = None


class AdvancedPDFProcessor:
    """Process PDFs efficiently with optimized chunking for technical content."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF with page numbers, optimized for speed and memory."""
        page_texts = []
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Process pages
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and len(text.strip()) > 20:  # Skip mostly empty pages
                        page_texts.append((text, page_num + 1))
                    
                    # Force garbage collection every 50 pages
                    if page_num % 50 == 0:
                        gc.collect()
                        
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
        
        return page_texts
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks optimized for technical/scientific content."""
        # For very short text, return as is
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try to split by sections first (common in technical papers)
        section_pattern = r'(?:\n\s*|\s{4,})((?:[0-9]+\.|[IVXLCDM]+\.|[A-Z][A-Za-z\s]+:))'
        formula_pattern = r'(?:\$\$.*?\$\$|\$.*?\$)'
        
        # Preserve formulas and equations
        formulas = re.findall(formula_pattern, text)
        for i, formula in enumerate(formulas):
            text = text.replace(formula, f"[FORMULA_{i}]")
        
        sections = re.split(section_pattern, text)
        
        if len(sections) > 1:
            # Process each section
            chunks = []
            current_section = ""
            
            # Recombine the split parts
            for i in range(0, len(sections) - 1, 2):
                if i + 1 < len(sections):
                    section_title = sections[i]
                    section_content = sections[i + 1] if i + 1 < len(sections) else ""
                    
                    # Combine section title with content
                    section_text = section_title + section_content
                    
                    # If section is small, add to current chunk
                    if len(current_section) + len(section_text) <= self.chunk_size:
                        current_section += section_text
                    else:
                        # Save current section if not empty
                        if current_section:
                            chunks.append(current_section.strip())
                        
                        # Handle large sections by sub-chunking
                        if len(section_text) > self.chunk_size:
                            sub_chunks = self._create_sub_chunks(section_text)
                            chunks.extend(sub_chunks)
                            current_section = ""
                        else:
                            current_section = section_text
            
            # Add the last section if not empty
            if current_section:
                chunks.append(current_section.strip())
        else:
            # No clear sections, use sliding window approach
            chunks = self._create_sub_chunks(text)
        
        # Restore formulas
        for i, formula in enumerate(formulas):
            for j in range(len(chunks)):
                chunks[j] = chunks[j].replace(f"[FORMULA_{i}]", formula)
                
        return chunks
    
    def _create_sub_chunks(self, text: str) -> List[str]:
        """Create chunks using sliding window approach, optimized for coherence."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a semantic boundary
            if end < len(text):
                # First try to find paragraph break
                paragraph_match = re.search(r'\n\s*\n', text[max(0, end-200):min(end+200, len(text))])
                if paragraph_match and end-200 + paragraph_match.start() > start:
                    end = end-200 + paragraph_match.start()
                else:
                    # Then try sentence boundary
                    sentence_match = re.search(r'(?<=[.!?])\s+(?=[A-Z0-9])', text[max(0, end-100):min(end+100, len(text))])
                    if sentence_match and end-100 + sentence_match.start() > start:
                        end = end-100 + sentence_match.start()
            
            chunk = text[start:end].strip()
            if chunk:  # Skip empty chunks
                chunks.append(chunk)
            
            # Slide the window with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end - self.chunk_overlap)
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[DocumentChunk]:
        """Process a single PDF into document chunks."""
        logger.info(f"Processing PDF: {pdf_path}")
        document_chunks = []
        chunk_id = 0
        
        try:
            page_texts = self.extract_text_from_pdf(pdf_path)
            filename = os.path.basename(pdf_path)
            
            for page_text, page_num in page_texts:
                chunks = self.chunk_text(page_text)
                
                for text in chunks:
                    if text.strip():  # Skip empty chunks
                        document_chunks.append(DocumentChunk(
                            text=text,
                            source=filename,
                            page=page_num,
                            chunk_id=chunk_id,
                            embedding=None
                        ))
                        chunk_id += 1
        
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
        
        logger.info(f"Created {len(document_chunks)} chunks from {pdf_path}")
        return document_chunks
    
    def process_directory(self, directory: str) -> List[DocumentChunk]:
        """Process all PDFs in a directory."""
        all_chunks = []
        
        # Check if directory exists
        if not os.path.exists(directory):
            logger.error(f"Directory {directory} does not exist")
            return all_chunks
        
        # Find all PDF files
        pdf_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        # Process each PDF file
        for pdf_path in pdf_files:
            pdf_chunks = self.process_pdf(pdf_path)
            all_chunks.extend(pdf_chunks)
            
            # Garbage collection to prevent memory issues
            gc.collect()
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


class TFIDFRetriever:
    """TF-IDF based retrieval system."""
    
    def __init__(self, use_svd: bool = True, n_components: int = 100):
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn is not available. TFIDFRetriever cannot be used.")
            self.fitted = False
            return
            
        self.vectorizer = TfidfVectorizer(
            strip_accents='unicode',
            lowercase=True,
            analyzer='word',
            stop_words='english',
            max_df=0.95,
            min_df=2,
            ngram_range=(1, 2)
        )
        self.use_svd = use_svd
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components) if use_svd else None
        self.document_vectors = None
        self.chunks = []
        self.fitted = False
    
    def fit(self, chunks: List[DocumentChunk]):
        """Train the TF-IDF model on document chunks."""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn is not available. Cannot fit TF-IDF model.")
            return
            
        if not chunks:
            logger.warning("No chunks provided for TF-IDF training.")
            return
        
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        
        # Fit the TF-IDF vectorizer
        try:
            logger.info("Fitting TF-IDF vectorizer...")
            self.document_vectors = self.vectorizer.fit_transform(texts)
            
            # Apply SVD if enabled
            if self.use_svd:
                logger.info(f"Applying SVD to reduce dimensions to {self.n_components}...")
                self.document_vectors = self.svd.fit_transform(self.document_vectors)
            
            self.fitted = True
            logger.info(f"TF-IDF model fitted with {len(texts)} documents")
            
        except Exception as e:
            logger.error(f"Error training TF-IDF model: {str(e)}")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve the most relevant chunks for a query."""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn is not available. Cannot query TF-IDF model.")
            return []
            
        if not self.fitted:
            logger.error("TF-IDF model not fitted yet.")
            return []
        
        try:
            # Create query vector
            query_vector = self.vectorizer.transform([query_text])
            
            # Apply SVD if enabled
            if self.use_svd:
                query_vector = self.svd.transform(query_vector)
            
            # Calculate similarities
            if self.use_svd:
                # For SVD-transformed vectors, use numpy directly
                similarities = np.dot(query_vector, self.document_vectors.T)[0]
                # Normalize if needed
                document_norms = np.linalg.norm(self.document_vectors, axis=1)
                query_norm = np.linalg.norm(query_vector)
                if query_norm > 0 and np.all(document_norms > 0):
                    similarities = similarities / (document_norms * query_norm)
            else:
                # For sparse matrices
                similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            
            # Get top-k results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append((self.chunks[idx], float(similarities[idx])))
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying TF-IDF model: {str(e)}")
            return []


class DenseRetriever:
    """Dense vector retrieval system using sentence transformers or similar."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.embeddings = None
        self.fitted = False
        
        # Initialize model if available
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading sentence transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.error(f"Error loading sentence transformer model: {str(e)}")
                self.model = None
        else:
            logger.warning("SentenceTransformer not available. Using fallback for dense retrieval.")
    
    def _fallback_embedding(self, texts: List[str]) -> np.ndarray:
        """Fallback embedding method using TF-IDF and SVD."""
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn is not available. Cannot create fallback embeddings.")
            return np.zeros((len(texts), 100))  # Return dummy embeddings
            
        logger.info("Using TF-IDF + SVD as fallback for embeddings")
        vectorizer = TfidfVectorizer(max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Apply SVD for dimensionality reduction
        svd = TruncatedSVD(n_components=min(384, tfidf_matrix.shape[1] - 1))
        return svd.fit_transform(tfidf_matrix)
    
    def fit(self, chunks: List[DocumentChunk]):
        """Create and store embeddings for document chunks."""
        if not chunks:
            logger.warning("No chunks provided for dense embeddings.")
            return
        
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        
        try:
            # Generate embeddings
            if self.model is not None:
                logger.info("Generating embeddings with sentence transformers...")
                
                # Process in batches to save memory
                all_embeddings = []
                batch_size = BATCH_SIZE
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = self.model.encode(batch_texts, show_progress_bar=False)
                    all_embeddings.append(batch_embeddings)
                    
                    # Force garbage collection
                    if i % (batch_size * 5) == 0:
                        gc.collect()
                
                self.embeddings = np.vstack(all_embeddings)
            else:
                # Use fallback method
                self.embeddings = self._fallback_embedding(texts)
            
            # Create FAISS index if available
            if FAISS_AVAILABLE:
                logger.info("Creating FAISS index...")
                dimension = self.embeddings.shape[1]
                self.index = faiss.IndexFlatIP(dimension)
                
                # Normalize vectors for cosine similarity
                normalized_embeddings = self.embeddings.copy()
                faiss.normalize_L2(normalized_embeddings)
                self.index.add(normalized_embeddings)
            
            self.fitted = True
            logger.info(f"Dense embeddings created for {len(texts)} chunks")
            
        except Exception as e:
            logger.error(f"Error creating dense embeddings: {str(e)}")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve the most relevant chunks using dense embeddings."""
        if not self.fitted:
            logger.error("Dense retriever not fitted yet.")
            return []
        
        try:
            # Create query embedding
            if self.model is not None:
                query_embedding = self.model.encode([query_text])[0]
            else:
                # Use fallback method (this is simplified, in practice we need the same pipeline)
                query_embedding = self._fallback_embedding([query_text])[0]
            
            # Search for similar chunks
            if FAISS_AVAILABLE and self.index is not None:
                # Normalize query vector for cosine similarity
                query_embedding_norm = query_embedding.copy()
                faiss.normalize_L2(np.array([query_embedding_norm]).astype('float32'))
                
                # Search using FAISS
                scores, indices = self.index.search(np.array([query_embedding_norm]).astype('float32'), k=top_k)
                
                results = []
                for i, idx in enumerate(indices[0]):
                    if idx < len(self.chunks) and scores[0][i] > 0:
                        results.append((self.chunks[idx], float(scores[0][i])))
                
                return results
            else:
                # Fallback to numpy
                similarities = np.dot(self.embeddings, query_embedding)
                
                # Normalize for cosine similarity
                norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
                similarities = similarities / norms
                
                # Get top-k results
                top_indices = similarities.argsort()[-top_k:][::-1]
                
                results = []
                for idx in top_indices:
                    results.append((self.chunks[idx], float(similarities[idx])))
                
                return results
                
        except Exception as e:
            logger.error(f"Error querying dense retriever: {str(e)}")
            return []


class HybridRetriever:
    """Hybrid retrieval system combining TF-IDF and dense embeddings."""
    
    def __init__(self, 
                 tfidf_weight: float = 0.3, 
                 dense_weight: float = 0.7,
                 model_name: str = EMBEDDING_MODEL_NAME):
        self.tfidf_retriever = TFIDFRetriever(use_svd=True)
        self.dense_retriever = DenseRetriever(model_name=model_name)
        self.tfidf_weight = tfidf_weight
        self.dense_weight = dense_weight
        self.chunks = []
        self.fitted = False
    
    def fit(self, chunks: List[DocumentChunk]):
        """Train both retrieval models."""
        if not chunks:
            logger.warning("No chunks provided for training.")
            return
        
        self.chunks = chunks
        
        # Train TF-IDF retriever
        logger.info("Training TF-IDF retriever...")
        self.tfidf_retriever.fit(chunks)
        
        # Train dense retriever
        logger.info("Training dense retriever...")
        self.dense_retriever.fit(chunks)
        
        self.fitted = True
        logger.info("Hybrid retriever trained successfully")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve the most relevant chunks using both methods."""
        if not self.fitted:
            logger.error("Hybrid retriever not fitted yet.")
            return []
        
        try:
            # Get more results than needed from each retriever
            search_k = min(top_k * 2, len(self.chunks))
            
            tfidf_results = self.tfidf_retriever.query(query_text, search_k)
            dense_results = self.dense_retriever.query(query_text, search_k)
            
            # Combine results with weights
            combined_scores = {}
            
            for chunk, score in tfidf_results:
                chunk_id = (chunk.source, chunk.page, chunk.chunk_id)
                combined_scores[chunk_id] = self.tfidf_weight * score
            
            for chunk, score in dense_results:
                chunk_id = (chunk.source, chunk.page, chunk.chunk_id)
                if chunk_id in combined_scores:
                    combined_scores[chunk_id] += self.dense_weight * score
                else:
                    combined_scores[chunk_id] = self.dense_weight * score
            
            # Create a mapping from IDs to chunks
            chunk_map = {}
            for chunk in self.chunks:
                chunk_id = (chunk.source, chunk.page, chunk.chunk_id)
                chunk_map[chunk_id] = chunk
            
            # Get top-k results
            top_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
            
            results = []
            for chunk_id in top_ids:
                if chunk_id in chunk_map:  # Safeguard against missing chunks
                    results.append((chunk_map[chunk_id], combined_scores[chunk_id]))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid query: {str(e)}")
            
            # Fallback to TF-IDF if hybrid fails
            logger.info("Falling back to TF-IDF retriever...")
            return self.tfidf_retriever.query(query_text, top_k)


class LLMGenerator:
    """Generate responses using HuggingFace model with robust fallback."""
    
    def __init__(self, model_name: str = MODEL_NAME, auth_token: Optional[str] = None):
        self.model_name = model_name
        self.auth_token = auth_token
        self.tokenizer = None
        self.model = None
        self.initialized = False
        
        # Try to initialize
        try:
            if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
                logger.warning("Transformers or PyTorch not available. Will use fallback text summarization mode.")
                return

            # Login to HuggingFace if token provided
            if self.auth_token:
                login(token=self.auth_token)
            
            # Load tokenizer and model
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Use half precision and device mapping for memory efficiency
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            self.initialized = True
            logger.info(f"Model {self.model_name} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model {self.model_name}: {str(e)}")
            logger.info("Will use fallback text summarization mode")
    
    def generate(self, query: str, context_chunks: List[Tuple[DocumentChunk, float]], 
                 max_new_tokens: int = 512) -> str:
        """Generate a response based on the query and context chunks."""
        # Format chunks with source information
        context_text = ""
        for i, (chunk, score) in enumerate(context_chunks):
            context_part = f"[Document {i+1}] {chunk.source} (Page {chunk.page}):\n{chunk.text}\n\n"
            context_text += context_part
        
        if self.initialized:
            try:
                # Create prompt
                prompt = f"""
You are a specialized AI research assistant with expertise in analyzing technical documents.
Use only the information from the provided documents to answer the question.
If the information to answer the question is not in the documents, say "I don't have enough information in the provided documents to answer this question."
Be precise and cite specific documents and page numbers when providing information.

DOCUMENTS:
{context_text}

QUESTION:
{query}

ANSWER:
"""
                
                # Generate response
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = inputs.to("cuda")
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs.input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.1
                    )
                
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract only the answer part
                answer_start = full_response.find("ANSWER:")
                if answer_start != -1:
                    answer = full_response[answer_start + 7:].strip()
                else:
                    answer = full_response.split(query)[-1].strip()
                
                return answer
                
            except Exception as e:
                logger.error(f"Error generating response with LLM: {str(e)}")
                logger.info("Falling back to text summarization mode")
                return self._summarize_context(query, context_text)
        else:
            return self._summarize_context(query, context_text)
    
    def _summarize_context(self, query: str, context_text: str) -> str:
        """Simple extractive summarization as a fallback."""
        try:
            # Extract sentences from context
            import re
            sentences = re.split(r'(?<=[.!?])\s+', context_text)
            
            # Score sentences based on word overlap with query
            query_words = set(query.lower().split())
            sentence_scores = []
            
            for sentence in sentences:
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                    
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words.intersection(sentence_words))
                score = overlap / len(sentence_words) if sentence_words else 0
                sentence_scores.append((sentence, score))
            
            # Get top sentences
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:5]
            
            # Create a summary
            summary = " ".join([s[0] for s in top_sentences])
            
            answer = f"Based on the provided documents, I found the following information about {query}:\n\n{summary}"
            return answer
            
        except Exception as e:
            logger.error(f"Error in fallback summarization: {str(e)}")
            return f"I found relevant information in the documents, but couldn't generate a detailed response. Here's a short excerpt:\n\n{context_text[:500]}..."


class HybridRAGBot:
    """Complete RAG system combining ML and LLM components for large PDFs."""
    
    def __init__(self, 
                 pdf_directory: str = PDF_DIRECTORY,
                 model_name: str = MODEL_NAME,
                 embedding_model: str = EMBEDDING_MODEL_NAME,
                 auth_token: Optional[str] = HF_TOKEN,
                 chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP,
                 top_k: int = 5):
        self.pdf_directory = pdf_directory
        self.top_k = top_k
        
        # Initialize components
        logger.info("Initializing PDF processor...")
        self.pdf_processor = AdvancedPDFProcessor(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        logger.info("Initializing hybrid retriever...")
        self.retriever = HybridRetriever(
            tfidf_weight=0.3,
            dense_weight=0.7,
            model_name=embedding_model
        )
        
        logger.info("Initializing LLM generator...")
        self.generator = LLMGenerator(
            model_name=model_name,
            auth_token=auth_token
        )
        
        # Storage for document chunks
        self.chunks = []
        self.processed = False
    
    def process_documents(self, force_reprocess: bool = False):
        """Process all PDFs in the directory and create embeddings."""
        if self.processed and not force_reprocess:
            logger.info("Documents already processed. Use force_reprocess=True to reprocess.")
            return
        
        # Process PDFs
        start_time = time.time()
        logger.info(f"Processing documents from {self.pdf_directory}...")
        
        self.chunks = self.pdf_processor.process_directory(self.
