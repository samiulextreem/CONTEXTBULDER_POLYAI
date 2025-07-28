#!/usr/bin/env python3
"""
Main script for creating AI context from PDF documents.

This script combines the functionality of the original txt_extractor.py and 
vector_db_builder.py into a single, streamlined file. It extracts text from 
PDFs, manages incremental processing, and generates a combined text file 
ready for AI context.

Includes OCR fallback for non-text-readable PDFs.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
import sys
import re
import logging
from typing import List, Dict, Any

import pypdf
import fitz  # PyMuPDF

# OCR-related imports
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
    logger.info("OCR libraries loaded successfully.")
except ImportError:
    OCR_AVAILABLE = False
    logger.warning("OCR libraries (pytesseract, pdf2image) not found. OCR functionality will be disabled.")
    logger.warning("To enable OCR, install the required libraries and Tesseract OCR engine.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Core Classes and Functions ---

class PDFTextExtractor:
    """
    A class to extract text from PDF documents.
    """
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file using multiple methods.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        text = ""
        
        # Method 1: Try PyMuPDF first (often better for text-based PDFs)
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                page_text = page.get_text("text")
                if page_text and page_text.strip():
                    text += f"\n\n--- Page {page_num + 1} ---\n\n"
                    text += page_text
            doc.close()
            
            if text.strip():
                logger.info(f"Successfully extracted text using PyMuPDF from {pdf_path}")
                return text.strip()
                
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        
        # Method 2: Try pypdf as a secondary fallback for text-based PDFs
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += f"\n\n--- Page {page_num + 1} ---\n\n"
                        text += page_text
                
                if text.strip():
                    logger.info(f"Successfully extracted text using pypdf from {pdf_path}")
                    return text.strip()
                    
        except Exception as e:
            logger.warning(f"pypdf extraction failed for {pdf_path}: {e}")

        # Method 3: OCR Fallback for scanned/image-based PDFs
        if not text.strip() and OCR_AVAILABLE:
            logger.info(f"No text extracted with standard methods. Attempting OCR for {pdf_path}...")
            try:
                images = convert_from_path(pdf_path, dpi=300)
                ocr_text = ""
                for i, image in enumerate(images):
                    ocr_text += f"\n\n--- Page {i + 1} (OCR) ---\n\n"
                    try:
                        ocr_text += pytesseract.image_to_string(image, lang='eng')
                    except pytesseract.TesseractNotFoundError:
                        logger.error("Tesseract is not installed or not in your PATH.")
                        logger.error("OCR functionality will not work until Tesseract is installed.")
                        # Disable OCR for the rest of the run
                        global OCR_AVAILABLE
                        OCR_AVAILABLE = False
                        return "" # Return empty string as we can't proceed
                    except Exception as ocr_error:
                        logger.error(f"OCR failed on page {i+1} for {pdf_path}: {ocr_error}")
                
                if ocr_text.strip():
                    logger.info(f"Successfully extracted text using OCR from {pdf_path}")
                    return ocr_text.strip()

            except Exception as e:
                logger.error(f"Failed to perform OCR on {pdf_path}: {e}")
        
        # If all methods failed
        if not text.strip():
            logger.error(f"All text extraction methods failed for {pdf_path}")
        
        return text.strip()

# --- Core Functions from txt_extractor.py ---

def get_file_hash(file_path: Path) -> str:
    """Get a hash of the file to detect content changes"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_processed_files(tracking_file: Path) -> dict:
    """Load the tracking file that contains info about processed files"""
    if not tracking_file.exists():
        return {}
    
    try:
        with open(tracking_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def save_processed_files(tracking_file: Path, processed_files: dict):
    """Save the tracking file with processed files info"""
    with open(tracking_file, 'w', encoding='utf-8') as f:
        json.dump(processed_files, f, indent=2, ensure_ascii=False)

def should_process_file(file_path: Path, processed_files: dict) -> bool:
    """Check if a file should be processed (new or modified)"""
    file_key = str(file_path.name)
    
    if file_key not in processed_files:
        return True  # New file
    
    # Check if file has been modified
    current_mtime = file_path.stat().st_mtime
    current_hash = get_file_hash(file_path)
    
    stored_info = processed_files[file_key]
    
    # Process if modification time or hash changed
    if (stored_info.get('mtime') != current_mtime or 
        stored_info.get('hash') != current_hash):
        return True
    
    return False

def clean_text_for_api(text: str) -> str:
    """Clean extracted text for better API processing"""
    lines = text.split('\n')
    cleaned_lines = [ ' '.join(line.split()) for line in lines if line.strip() ]
    cleaned_text = '\n'.join(cleaned_lines)
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.replace('--- ', '')
    return cleaned_text.strip()

# --- Main Application Logic ---

def process_documents(input_folder: str, output_folder: str):
    """
    Main function to process documents in a folder.
    
    Args:
        input_folder: Folder containing PDF files.
        output_folder: Folder to save exported text files.
    """
    print("=" * 80)
    print("PolyAI CONTEXT BUILDER")
    print("=" * 80)
    
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    tracking_file = output_path / "processed_files.json"
    processed_files = load_processed_files(tracking_file)
    
    print("🔧 Initializing text extractor...")
    extractor = PDFTextExtractor()
    
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ ERROR: Input folder not found: {input_folder}")
        return
    
    all_files = list(input_path.glob("*.pdf"))
    
    if not all_files:
        print(f"✅ No PDF files found in {input_folder}")
        return
    
    files_to_process = [f for f in all_files if should_process_file(f, processed_files)]
    
    print(f"📁 Found {len(all_files)} total PDF files.")
    print(f"📁 Skipping {len(all_files) - len(files_to_process)} already processed files.")
    print(f"🆕 Processing {len(files_to_process)} new or modified files.")
    
    if not files_to_process:
        print("✅ All documents are up to date!")
        return
    
    print()
    
    exported_files = []
    total_text_length = 0
    
    for file_path in files_to_process:
        print(f"📄 Processing: {file_path.name}")
        try:
            text = extractor.extract_text_from_pdf(str(file_path))
            
            if not text:
                print(f"   ⚠️  No text extracted from {file_path.name}")
                status = 'no_text'
                text_len = 0
            else:
                cleaned_text = clean_text_for_api(text)
                text_len = len(cleaned_text)
                exported_files.append({
                    'source': file_path.name,
                    'text_content': cleaned_text,
                    'text_length': text_len,
                })
                total_text_length += text_len
                status = 'success'
                print(f"   ✅ Extracted: {text_len} characters")

            processed_files[file_path.name] = {
                'mtime': file_path.stat().st_mtime,
                'hash': get_file_hash(file_path),
                'processed_at': datetime.now().isoformat(),
                'text_length': text_len,
                'status': status
            }
        except Exception as e:
            print(f"   ❌ ERROR processing {file_path.name}: {e}")
            processed_files[file_path.name] = {
                'mtime': file_path.stat().st_mtime,
                'hash': get_file_hash(file_path),
                'processed_at': datetime.now().isoformat(),
                'text_length': 0,
                'status': 'error',
                'error': str(e)
            }
    
    save_processed_files(tracking_file, processed_files)
    
    if exported_files:
        combined_file = output_path / "all_documents_combined.txt"
        with open(combined_file, 'a', encoding='utf-8') as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"SESSION: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"FILES PROCESSED: {len(exported_files)}\n")
            f.write("="*80 + "\n\n")
            
            for i, file_info in enumerate(exported_files, 1):
                f.write(f"\n--- START: {file_info['source']} ---\n\n")
                f.write(file_info['text_content'])
                f.write(f"\n\n--- END: {file_info['source']} ---\n")
    
    print("\n" + "="*80)
    print("✅ CONTEXT CREATION COMPLETE!")
    print("="*80)
    print(f"📊 New files processed: {len(exported_files)}")
    print(f"📊 Total text from new files: {total_text_length:,} characters")
    print(f"📁 Output folder: {output_path.resolve()}")
    if exported_files:
        print(f"📄 Combined file 'all_documents_combined.txt' has been updated.")
    print("="*80)


if __name__ == "__main__":
    # Command-line argument support
    if len(sys.argv) >= 2:
        input_folder = sys.argv[1]
    else:
        # Fallback to a default if no argument is provided.
        # The make_context.bat script should always provide an argument.
        input_folder = "pdfs"
        Path(input_folder).mkdir(exist_ok=True)
    
    # In this script, input and output folders are the same.
    output_folder = input_folder

    process_documents(input_folder, output_folder) 