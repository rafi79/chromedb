#!/usr/bin/env python
"""
Quick start script for the Hybrid RAG System.
This script helps set up and run the system with minimal configuration.
"""

import os
import argparse
import subprocess
import sys

def check_dependencies():
    """Check if required packages are installed and install if needed."""
    try:
        import streamlit
        import numpy
        import pandas
        import PyPDF2
        import sklearn
        import plotly
        print("✅ Core dependencies are installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        install = input("Would you like to install required packages? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install", 
                                   "streamlit", "numpy", "pandas", "PyPDF2", 
                                   "scikit-learn", "plotly", "tqdm"])
            print("✅ Core dependencies installed")
        else:
            print("❌ Please install required packages and try again")
            sys.exit(1)
    
    # Check optional dependencies
    missing_optional = []
    try:
        import sentence_transformers
    except ImportError:
        missing_optional.append("sentence-transformers")
    
    try:
        import faiss
    except ImportError:
        missing_optional.append("faiss-cpu")
    
    try:
        import transformers
        import huggingface_hub
    except ImportError:
        missing_optional.append("transformers")
        missing_optional.append("huggingface-hub")
    
    if missing_optional:
        print(f"⚠️ Missing optional dependencies: {', '.join(missing_optional)}")
        print("ℹ️ The system will use fallback methods for missing features")
        install = input("Would you like to install optional packages for enhanced features? (y/n): ")
        if install.lower() == 'y':
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_optional)
            print("✅ Optional dependencies installed")

def setup_directories(pdf_dir):
    """Create necessary directories if they don't exist."""
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"✅ Created PDF directory: {pdf_dir}")
    else:
        print(f"✅ PDF directory exists: {pdf_dir}")
    
    # Count PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    if pdf_files:
        print(f"ℹ️ Found {len(pdf_files)} PDF files in {pdf_dir}")
    else:
        print(f"⚠️ No PDF files found in {pdf_dir}. Please add PDFs before processing.")

def set_huggingface_token():
    """Set up Hugging Face token if needed."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        print("ℹ️ No Hugging Face token found in environment variables")
        token = input("Enter your Hugging Face token (press Enter to skip): ")
        if token:
            os.environ['HF_TOKEN'] = token
            print("✅ Hugging Face token set")
        else:
            print("⚠️ No token provided. Some features may be limited.")
    else:
        print("✅ Hugging Face token found in environment variables")

def main():
    """Run the main script."""
    parser = argparse.ArgumentParser(description="Hybrid RAG System Quick Start")
    parser.add_argument("--pdf-dir", default="./pdfs", help="Directory for PDF files")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    args = parser.parse_args()
    
    print("🚀 Hybrid ML-LLM RAG System Quick Start")
    print("---------------------------------------")
    
    if not args.skip_checks:
        check_dependencies()
    
    setup_directories(args.pdf_dir)
    set_huggingface_token()
    
    print("\n✨ Starting Streamlit application...")
    subprocess.run(["streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
