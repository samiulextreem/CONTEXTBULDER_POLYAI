#!/usr/bin/env python3
"""
Enhanced Vector Database Builder for PDF Documents and Images

This script extracts comprehensive text and visual data from PDF files and standalone images,
creates embeddings, and builds a FAISS vector database for use with Large Language Models (LLMs).

Features:
- Multi-method PDF text extraction (PyMuPDF, pypdf, Enhanced OCR)
- Standalone image processing with comprehensive OCR
- AI-powered image description using BLIP vision model
- Enhanced OCR with multiple configurations and preprocessing
- Comprehensive metadata extraction and analysis
- Multimodal content indexing (text + images)
- Advanced image analysis (color, complexity, text density)
"""

import os
import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pickle
import json
import re
from collections import defaultdict

import pypdf
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    import cv2
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PDFVectorDatabase:
    """
    A class to build and query a vector database from PDF documents.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 512, chunk_overlap: int = 50, 
                 process_images: bool = True):
        """
        Initialize the PDF Vector Database.
        
        Args:
            model_name: Name of the sentence transformer model to use for embeddings
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            process_images: Whether to extract and describe images from PDFs
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.process_images = process_images
        
        # Load the embedding model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize vision model if requested and available
        self.vision_processor = None
        self.vision_model = None
        if self.process_images and VISION_AVAILABLE:
            try:
                logger.info("Loading vision model for image description...")
                self.vision_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                logger.info("Vision model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load vision model: {e}")
                self.process_images = False
                self.vision_processor = None
                self.vision_model = None
        elif self.process_images and not VISION_AVAILABLE:
            logger.warning("Vision processing requested but transformers not available. Install with: pip install transformers torch")
            self.process_images = False
        
        # Initialize FAISS index
        self.index = None
        self.metadata = []  # Store metadata for each chunk
        
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing image data and metadata
        """
        images_data = []
        
        if not self.process_images:
            return images_data
        
        try:
            # Method 1: Extract images using PyMuPDF
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Convert to PIL Image
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("ppm")
                            pil_image = Image.open(io.BytesIO(img_data))
                        else:  # CMYK
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            img_data = pix1.tobytes("ppm")
                            pil_image = Image.open(io.BytesIO(img_data))
                            pix1 = None
                        
                        pix = None
                        
                        # Only process reasonably sized images
                        if pil_image.width > 50 and pil_image.height > 50:
                            images_data.append({
                                'image': pil_image,
                                'page_num': page_num + 1,
                                'img_index': img_index,
                                'width': pil_image.width,
                                'height': pil_image.height,
                                'source': 'embedded'
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error extracting image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            logger.warning(f"Error extracting images from {pdf_path}: {e}")
        
        # Method 2: If no embedded images found, try converting PDF pages to images
        if not images_data and OCR_AVAILABLE:
            try:
                logger.info(f"No embedded images found, converting PDF pages to images for {pdf_path}")
                page_images = convert_from_path(pdf_path, dpi=150, first_page=1, last_page=min(5, 10))  # Limit to first 5 pages
                
                for page_num, page_image in enumerate(page_images):
                    images_data.append({
                        'image': page_image,
                        'page_num': page_num + 1,
                        'img_index': 0,
                        'width': page_image.width,
                        'height': page_image.height,
                        'source': 'page_render'
                    })
                    
            except Exception as e:
                logger.warning(f"Error converting PDF pages to images for {pdf_path}: {e}")
        
        logger.info(f"Extracted {len(images_data)} images from {pdf_path}")
        return images_data
    
    def describe_image(self, image: Image.Image) -> str:
        """
        Generate a text description of an image using a vision model.
        
        Args:
            image: PIL Image object
            
        Returns:
            Text description of the image
        """
        if not self.process_images or not self.vision_model or not self.vision_processor:
            return ""
        
        try:
            # Prepare the image
            inputs = self.vision_processor(image, return_tensors="pt")
            
            # Generate description
            with torch.no_grad():
                out = self.vision_model.generate(**inputs, max_length=150, num_beams=5)
            
            # Decode the description
            description = self.vision_processor.decode(out[0], skip_special_tokens=True)
            return description
            
        except Exception as e:
            logger.warning(f"Error generating image description: {e}")
            return ""

    def extract_text_from_pdf(self, pdf_path: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Extract text content and images from a PDF file using multiple methods.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted text content, list of image descriptions)
        """
        text = ""
        image_descriptions = []
        
        # Extract images first if image processing is enabled
        if self.process_images:
            logger.info(f"Extracting images from {pdf_path}")
            images_data = self.extract_images_from_pdf(pdf_path)
            
            for img_data in images_data:
                description = self.describe_image(img_data['image'])
                if description:
                    image_desc = {
                        'description': description,
                        'page_num': img_data['page_num'],
                        'img_index': img_data['img_index'],
                        'source': img_data['source'],
                        'dimensions': f"{img_data['width']}x{img_data['height']}"
                    }
                    image_descriptions.append(image_desc)
                    logger.info(f"Image on page {img_data['page_num']}: {description[:100]}...")
        
        # Method 1: Try PyMuPDF first (better than pypdf)
        try:
            logger.info(f"Trying PyMuPDF extraction for {pdf_path}")
            doc = fitz.open(pdf_path)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                try:
                    page_text = page.get_text()
                    if page_text.strip():
                        text += f"\n\n--- Page {page_num + 1} ---\n\n"
                        text += page_text
                except AttributeError:
                    # Fallback if get_text() is not available
                    page_text = page.get_text("text")
                    if page_text.strip():
                        text += f"\n\n--- Page {page_num + 1} ---\n\n"
                        text += page_text
                    
                    # Add image descriptions for this page
                    page_images = [img for img in image_descriptions if img['page_num'] == page_num + 1]
                    if page_images:
                        text += f"\n\n[IMAGES ON PAGE {page_num + 1}]:\n"
                        for img in page_images:
                            text += f"- Image {img['img_index'] + 1} ({img['source']}): {img['description']}\n"
                        
            doc.close()
            
            if text.strip():
                logger.info(f"Successfully extracted text using PyMuPDF from {pdf_path}")
                return text.strip(), image_descriptions
                
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
        
        # Method 2: Try pypdf as fallback
        try:
            logger.info(f"Trying pypdf extraction for {pdf_path}")
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += f"\n\n--- Page {page_num + 1} ---\n\n"
                            text += page_text
                            
                            # Add image descriptions for this page
                            page_images = [img for img in image_descriptions if img['page_num'] == page_num + 1]
                            if page_images:
                                text += f"\n\n[IMAGES ON PAGE {page_num + 1}]:\n"
                                for img in page_images:
                                    text += f"- Image {img['img_index'] + 1} ({img['source']}): {img['description']}\n"
                                    
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1} of {pdf_path}: {e}")
                        continue
                
                if text.strip():
                    logger.info(f"Successfully extracted text using pypdf from {pdf_path}")
                    return text.strip(), image_descriptions
                    
        except Exception as e:
            logger.warning(f"pypdf extraction failed for {pdf_path}: {e}")
        
        # Method 3: Try OCR if available and other methods failed
        if OCR_AVAILABLE and not text.strip():
            try:
                logger.info(f"Trying enhanced OCR extraction for {pdf_path}")
                # Convert PDF to images
                images = convert_from_path(pdf_path, dpi=300)  # Higher DPI for better OCR
                
                for page_num, image in enumerate(images):
                    try:
                        # Use enhanced OCR extraction
                        page_text = self.extract_text_with_enhanced_ocr(image, f"{Path(pdf_path).name} page {page_num + 1}")
                        
                        # Also try table extraction for structured data
                        table_text = self.extract_table_data(image, f"{Path(pdf_path).name} page {page_num + 1}")
                        
                        combined_text = ""
                        if page_text and page_text.strip():
                            combined_text = page_text.strip()
                        
                        # Add table data if different/additional
                        if table_text and table_text.strip():
                            if combined_text:
                                combined_text += f"\n\n[STRUCTURED TABLE DATA]:\n{table_text.strip()}"
                            else:
                                combined_text = table_text.strip()
                        
                        if combined_text:
                            text += f"\n\n--- Page {page_num + 1} (OCR) ---\n\n"
                            text += combined_text
                            
                            # Add image descriptions for this page
                            page_images = [img for img in image_descriptions if img['page_num'] == page_num + 1]
                            if page_images:
                                text += f"\n\n[IMAGES ON PAGE {page_num + 1}]:\n"
                                for img in page_images:
                                    text += f"- Image {img['img_index'] + 1} ({img['source']}): {img['description']}\n"
                                    
                    except Exception as e:
                        logger.warning(f"OCR failed for page {page_num + 1} of {pdf_path}: {e}")
                        continue
                
                if text.strip():
                    logger.info(f"Successfully extracted text using OCR from {pdf_path}")
                    return text.strip(), image_descriptions
                    
            except Exception as e:
                logger.warning(f"OCR extraction failed for {pdf_path}: {e}")
        
        # If all text extraction methods failed but we have image descriptions, use those
        if not text.strip() and image_descriptions:
            logger.info(f"No text extracted, but found {len(image_descriptions)} image descriptions")
            text = "--- Document contains images only ---\n\n"
            for img in image_descriptions:
                text += f"Page {img['page_num']}, Image {img['img_index'] + 1}: {img['description']}\n"
            return text, image_descriptions
        
        # If all methods failed
        if not text.strip():
            logger.error(f"All extraction methods failed for {pdf_path}")
            return "", image_descriptions
        
        return text.strip(), image_descriptions
    
    def chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text content to chunk
            filename: Name of the source file
            
        Returns:
            List of dictionaries containing chunk text and metadata
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at sentence or word boundaries
            if end < len(text):
                # Look for sentence ending
                last_period = chunk_text.rfind('.')
                last_exclamation = chunk_text.rfind('!')
                last_question = chunk_text.rfind('?')
                sentence_end = max(last_period, last_exclamation, last_question)
                
                if sentence_end > len(chunk_text) * 0.7:  # If sentence ending is in last 30%
                    chunk_text = chunk_text[:sentence_end + 1]
                    end = start + sentence_end + 1
                else:
                    # Look for word boundary
                    last_space = chunk_text.rfind(' ')
                    if last_space > len(chunk_text) * 0.8:  # If space is in last 20%
                        chunk_text = chunk_text[:last_space]
                        end = start + last_space
            
            # Clean the chunk
            chunk_text = chunk_text.strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append({
                    'text': chunk_text,
                    'filename': filename,
                    'chunk_id': chunk_id,
                    'start_char': start,
                    'end_char': end,
                    'length': len(chunk_text)
                })
                chunk_id += 1
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(text):
                break
                
        return chunks
    
    def build_database(self, pdf_folder: str, save_path: str = "vector_db"):
        """
        Build the vector database from PDF files in a folder.
        
        Args:
            pdf_folder: Path to folder containing PDF files
            save_path: Path to save the vector database
        """
        pdf_folder_path = Path(pdf_folder)
        if not pdf_folder_path.exists():
            raise FileNotFoundError(f"PDF folder not found: {pdf_folder_path}")
        
        # Find all PDF files and standalone image files
        pdf_files = list(pdf_folder_path.glob("*.pdf"))
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
            image_files.extend(pdf_folder_path.glob(ext))
        
        total_files = len(pdf_files) + len(image_files)
        if total_files == 0:
            raise ValueError(f"No PDF or image files found in {pdf_folder_path}")
        
        logger.info(f"Found {len(pdf_files)} PDF files and {len(image_files)} image files")
        
        # DEBUG: Track processing statistics
        processing_stats = {
            'total_files': total_files,
            'processed_files': 0,
            'skipped_files': 0,
            'skipped_reasons': [],
            'total_text_extracted': 0,
            'total_chunks_created': 0,
            'files_with_text': [],
            'files_without_text': []
        }
        
        # First, analyze all PDFs and images
        print("\nAnalyzing PDF and image files...")
        analyses = self.analyze_pdf_folder(pdf_folder_path)
        
        print("\nFile Analysis Results:")
        print("=" * 70)
        for analysis in analyses:
            print(f"\nFile: {analysis['filename']}")
            print(f"  Type: {analysis.get('file_type', 'pdf')}")
            print(f"  Size: {analysis['file_size_mb']} MB")
            if analysis.get('dimensions'):
                print(f"  Dimensions: {analysis['dimensions']}")
            if analysis.get('format'):
                print(f"  Format: {analysis['format']}")
            if analysis.get('mode'):
                print(f"  Color Mode: {analysis['mode']}")
            print(f"  Pages: {analysis['total_pages']}")
            print(f"  Text extractable: {'✓' if analysis['text_extractable'] else '✗'}")
            print(f"  Likely scanned: {'✓' if analysis['likely_scanned'] else '✗'}")
            print(f"  Images found: {analysis.get('image_count', 0)}")
            print(f"  Has images: {'✓' if analysis.get('has_images', False) else '✗'}")
            if analysis['extraction_method']:
                print(f"  Best method: {analysis['extraction_method']}")
            if analysis.get('estimated_text_length'):
                print(f"  Estimated text length: {analysis['estimated_text_length']} chars")
            if analysis.get('text_density'):
                print(f"  Text density: {analysis['text_density']:.2f} chars/megapixel")
            if analysis.get('content_richness'):
                print(f"  Content richness: {analysis['content_richness']}")
            if analysis['sample_text']:
                print(f"  Sample text: {analysis['sample_text'][:150]}...")
            if analysis.get('image_descriptions'):
                print(f"  Sample image descriptions:")
                for img_desc in analysis['image_descriptions']:
                    print(f"    Page {img_desc['page']}: {img_desc['description'][:100]}...")
            if analysis.get('error'):
                print(f"  Error: {analysis['error']}")
        
        print("\n" + "=" * 70)
        
        all_chunks = []
        
        # Process each PDF file
        for pdf_path in tqdm(pdf_files, desc="Processing PDF files"):
            logger.info(f"Processing PDF: {pdf_path.name}")
            
            # DEBUG: Track this file
            file_debug = {
                'filename': pdf_path.name,
                'extraction_successful': False,
                'text_length': 0,
                'chunks_created': 0,
                'skip_reason': None
            }
            
            try:
                # Extract text and images
                text, image_descriptions = self.extract_text_from_pdf(str(pdf_path))
                
                # DEBUG: Check extraction results
                if not text:
                    file_debug['skip_reason'] = 'No text extracted'
                    processing_stats['skipped_files'] += 1
                    processing_stats['skipped_reasons'].append(f"{pdf_path.name}: No text extracted")
                    processing_stats['files_without_text'].append(pdf_path.name)
                    logger.warning(f"No text extracted from {pdf_path.name}")
                    
                    # Print debug info even for failed extraction
                    print(f"\n{'=' * 80}")
                    print(f"DEBUG: FAILED EXTRACTION FROM: {pdf_path.name}")
                    print(f"{'=' * 80}")
                    print(f"Text extracted: {bool(text)}")
                    print(f"Text length: {len(text) if text else 0} characters")
                    print(f"Image descriptions: {len(image_descriptions)} images")
                    print(f"Skip reason: {file_debug['skip_reason']}")
                    print(f"{'=' * 80}\n")
                    continue
                
                # DEBUG: Track successful extraction
                file_debug['extraction_successful'] = True
                file_debug['text_length'] = len(text)
                processing_stats['total_text_extracted'] += len(text)
                processing_stats['files_with_text'].append(pdf_path.name)
                
                # Print extracted text in terminal
                print(f"\n{'=' * 80}")
                print(f"EXTRACTED TEXT FROM: {pdf_path.name}")
                print(f"{'=' * 80}")
                print(f"Text length: {len(text)} characters")
                print(f"Image descriptions: {len(image_descriptions)} images")
                print(f"{'=' * 80}")
                print(text)
                print(f"{'=' * 80}")
                print(f"END OF TEXT FROM: {pdf_path.name}")
                print(f"{'=' * 80}\n")
                
                # Create chunks
                chunks = self.chunk_text(text, pdf_path.name)
                
                # DEBUG: Check chunking results
                if not chunks:
                    file_debug['skip_reason'] = 'No chunks created from text'
                    processing_stats['skipped_files'] += 1
                    processing_stats['skipped_reasons'].append(f"{pdf_path.name}: No chunks created")
                    logger.warning(f"No chunks created from {pdf_path.name} despite having text")
                    
                    # Print debug info for chunking failure
                    print(f"\n{'=' * 80}")
                    print(f"DEBUG: CHUNKING FAILED FOR: {pdf_path.name}")
                    print(f"{'=' * 80}")
                    print(f"Original text length: {len(text)} characters")
                    print(f"Chunks created: {len(chunks)}")
                    print(f"Chunk size setting: {self.chunk_size}")
                    print(f"Chunk overlap setting: {self.chunk_overlap}")
                    print(f"First 200 chars of text: {text[:200]}...")
                    print(f"{'=' * 80}\n")
                    continue
                
                # DEBUG: Track successful chunking
                file_debug['chunks_created'] = len(chunks)
                processing_stats['total_chunks_created'] += len(chunks)
                processing_stats['processed_files'] += 1
                
                # Add image metadata to chunks
                for chunk in chunks:
                    chunk['image_count'] = len(image_descriptions)
                    chunk['has_images'] = len(image_descriptions) > 0
                    chunk['file_type'] = 'pdf'
                
                all_chunks.extend(chunks)
                
                # DEBUG: Print chunk information
                print(f"\n{'=' * 80}")
                print(f"CHUNKING RESULTS FOR: {pdf_path.name}")
                print(f"{'=' * 80}")
                print(f"Original text length: {len(text)} characters")
                print(f"Chunks created: {len(chunks)}")
                print(f"Average chunk size: {sum(len(chunk['text']) for chunk in chunks) / len(chunks):.1f} characters")
                print(f"Chunk size range: {min(len(chunk['text']) for chunk in chunks)} - {max(len(chunk['text']) for chunk in chunks)}")
                print(f"{'=' * 80}\n")
                
                logger.info(f"Created {len(chunks)} chunks from {pdf_path.name}")
                if image_descriptions:
                    logger.info(f"Found {len(image_descriptions)} images with descriptions")
                    
            except Exception as e:
                file_debug['skip_reason'] = f'Exception during processing: {str(e)}'
                processing_stats['skipped_files'] += 1
                processing_stats['skipped_reasons'].append(f"{pdf_path.name}: Exception - {str(e)}")
                logger.error(f"Exception processing {pdf_path.name}: {e}")
                continue
        
        # Process each standalone image file
        for image_path in tqdm(image_files, desc="Processing image files"):
            logger.info(f"Processing image: {image_path.name}")
            
            # DEBUG: Track this file
            file_debug = {
                'filename': image_path.name,
                'extraction_successful': False,
                'text_length': 0,
                'chunks_created': 0,
                'skip_reason': None
            }
            
            try:
                # Extract text and descriptions from image
                text, image_descriptions = self.extract_text_from_image(str(image_path))
                
                # DEBUG: Check extraction results
                if not text:
                    file_debug['skip_reason'] = 'No text extracted'
                    processing_stats['skipped_files'] += 1
                    processing_stats['skipped_reasons'].append(f"{image_path.name}: No text extracted")
                    processing_stats['files_without_text'].append(image_path.name)
                    logger.warning(f"No text extracted from {image_path.name}")
                    
                    # Print debug info even for failed extraction
                    print(f"\n{'=' * 80}")
                    print(f"DEBUG: FAILED EXTRACTION FROM IMAGE: {image_path.name}")
                    print(f"{'=' * 80}")
                    print(f"Text extracted: {bool(text)}")
                    print(f"Text length: {len(text) if text else 0} characters")
                    print(f"Image descriptions: {len(image_descriptions)} images")
                    print(f"Skip reason: {file_debug['skip_reason']}")
                    print(f"{'=' * 80}\n")
                    continue
                
                # DEBUG: Track successful extraction
                file_debug['extraction_successful'] = True
                file_debug['text_length'] = len(text)
                processing_stats['total_text_extracted'] += len(text)
                processing_stats['files_with_text'].append(image_path.name)
                
                # Print extracted text in terminal
                print(f"\n{'=' * 80}")
                print(f"EXTRACTED TEXT FROM IMAGE: {image_path.name}")
                print(f"{'=' * 80}")
                print(f"Text length: {len(text)} characters")
                print(f"Image descriptions: {len(image_descriptions)} images")
                print(f"{'=' * 80}")
                print(text)
                print(f"{'=' * 80}")
                print(f"END OF TEXT FROM IMAGE: {image_path.name}")
                print(f"{'=' * 80}\n")
                
                # Create chunks
                chunks = self.chunk_text(text, image_path.name)
                
                # DEBUG: Check chunking results
                if not chunks:
                    file_debug['skip_reason'] = 'No chunks created from text'
                    processing_stats['skipped_files'] += 1
                    processing_stats['skipped_reasons'].append(f"{image_path.name}: No chunks created")
                    logger.warning(f"No chunks created from {image_path.name} despite having text")
                    
                    # Print debug info for chunking failure
                    print(f"\n{'=' * 80}")
                    print(f"DEBUG: CHUNKING FAILED FOR IMAGE: {image_path.name}")
                    print(f"{'=' * 80}")
                    print(f"Original text length: {len(text)} characters")
                    print(f"Chunks created: {len(chunks)}")
                    print(f"Chunk size setting: {self.chunk_size}")
                    print(f"Chunk overlap setting: {self.chunk_overlap}")
                    print(f"First 200 chars of text: {text[:200]}...")
                    print(f"{'=' * 80}\n")
                    continue
                
                # DEBUG: Track successful chunking
                file_debug['chunks_created'] = len(chunks)
                processing_stats['total_chunks_created'] += len(chunks)
                processing_stats['processed_files'] += 1
                
                # Add image metadata to chunks
                for chunk in chunks:
                    chunk['image_count'] = len(image_descriptions)
                    chunk['has_images'] = len(image_descriptions) > 0
                    chunk['file_type'] = 'image'
                
                all_chunks.extend(chunks)
                
                # DEBUG: Print chunk information
                print(f"\n{'=' * 80}")
                print(f"CHUNKING RESULTS FOR IMAGE: {image_path.name}")
                print(f"{'=' * 80}")
                print(f"Original text length: {len(text)} characters")
                print(f"Chunks created: {len(chunks)}")
                print(f"Average chunk size: {sum(len(chunk['text']) for chunk in chunks) / len(chunks):.1f} characters")
                print(f"Chunk size range: {min(len(chunk['text']) for chunk in chunks)} - {max(len(chunk['text']) for chunk in chunks)}")
                print(f"{'=' * 80}\n")
                
                logger.info(f"Created {len(chunks)} chunks from {image_path.name}")
                if image_descriptions:
                    logger.info(f"Generated image description")
                    
            except Exception as e:
                file_debug['skip_reason'] = f'Exception during processing: {str(e)}'
                processing_stats['skipped_files'] += 1
                processing_stats['skipped_reasons'].append(f"{image_path.name}: Exception - {str(e)}")
                logger.error(f"Exception processing {image_path.name}: {e}")
                continue
        
        # DEBUG: Print comprehensive processing summary
        print(f"\n{'=' * 80}")
        print(f"PROCESSING SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total files found: {processing_stats['total_files']}")
        print(f"Files successfully processed: {processing_stats['processed_files']}")
        print(f"Files skipped: {processing_stats['skipped_files']}")
        print(f"Total text extracted: {processing_stats['total_text_extracted']} characters")
        print(f"Total chunks created: {processing_stats['total_chunks_created']}")
        print(f"")
        print(f"Files WITH text extracted: {len(processing_stats['files_with_text'])}")
        for filename in processing_stats['files_with_text']:
            print(f"  ✓ {filename}")
        print(f"")
        print(f"Files WITHOUT text extracted: {len(processing_stats['files_without_text'])}")
        for filename in processing_stats['files_without_text']:
            print(f"  ✗ {filename}")
        print(f"")
        if processing_stats['skipped_reasons']:
            print(f"Skip reasons:")
            for reason in processing_stats['skipped_reasons']:
                print(f"  - {reason}")
        print(f"{'=' * 80}\n")
        
        if not all_chunks:
            error_msg = f"No text chunks were created from the PDF or image files.\n"
            error_msg += f"Files processed: {processing_stats['processed_files']}/{processing_stats['total_files']}\n"
            error_msg += f"Files skipped: {processing_stats['skipped_files']}\n"
            error_msg += f"Skip reasons: {processing_stats['skipped_reasons']}\n"
            error_msg += f"This could indicate:\n"
            error_msg += f"- Files are corrupted, encrypted, or unreadable\n"
            error_msg += f"- Files contain no extractable text content\n"
            error_msg += f"- OCR libraries are missing or not working\n"
            error_msg += f"- Text extraction methods are failing\n"
            error_msg += f"- Chunking process is filtering out all text\n"
            raise ValueError(error_msg)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        # Create embeddings
        logger.info("Creating embeddings...")
        texts = [chunk['text'] for chunk in all_chunks]
        
        # DEBUG: Check embedding input
        print(f"\n{'=' * 80}")
        print(f"EMBEDDING CREATION")
        print(f"{'=' * 80}")
        print(f"Number of chunks to embed: {len(all_chunks)}")
        print(f"Total text length for embedding: {sum(len(text) for text in texts)} characters")
        print(f"Average text length per chunk: {sum(len(text) for text in texts) / len(texts):.1f} characters")
        print(f"Embedding model: {self.model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"{'=' * 80}\n")
        
        # Create embeddings in batches to avoid memory issues
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        logger.info("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Store metadata
        self.metadata = all_chunks
        
        # Save the database
        self.save_database(save_path)
        
        logger.info(f"Vector database built successfully with {len(all_chunks)} chunks")
        logger.info(f"Database saved to: {save_path}")
        
        # Show final summary
        print(f"\nSUCCESS: Vector database created with {len(all_chunks)} chunks")
        successful_files = list(set([chunk['filename'] for chunk in all_chunks]))
        print(f"Successfully processed {len(successful_files)} files: {', '.join(successful_files)}")
        
        # DEBUG: Print final database statistics
        print(f"\n{'=' * 80}")
        print(f"FINAL DATABASE STATISTICS")
        print(f"{'=' * 80}")
        print(f"Total chunks in database: {len(self.metadata)}")
        print(f"Total files represented: {len(set(chunk['filename'] for chunk in self.metadata))}")
        print(f"File breakdown:")
        file_chunk_counts = {}
        for chunk in self.metadata:
            filename = chunk['filename']
            file_chunk_counts[filename] = file_chunk_counts.get(filename, 0) + 1
        for filename, count in file_chunk_counts.items():
            print(f"  {filename}: {count} chunks")
        print(f"{'=' * 80}\n")
    
    def save_database(self, save_path: str):
        """
        Save the vector database to disk.
        
        Args:
            save_path: Path to save the database
        """
        save_path_obj = Path(save_path)
        save_path_obj.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(save_path_obj / "index.faiss"))
        
        # Save metadata
        with open(save_path_obj / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'embedding_dim': self.embedding_dim,
            'num_chunks': len(self.metadata),
            'process_images': self.process_images
        }
        
        with open(save_path_obj / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def load_database(self, load_path: str):
        """
        Load a previously saved vector database.
        
        Args:
            load_path: Path to the saved database
        """
        load_path_obj = Path(load_path)
        
        if not load_path_obj.exists():
            raise FileNotFoundError(f"Database path not found: {load_path_obj}")
        
        # Load configuration
        with open(load_path_obj / "config.json", "r") as f:
            config = json.load(f)
        
        # Verify model compatibility
        if config['model_name'] != self.model_name:
            logger.warning(f"Model mismatch: expected {self.model_name}, found {config['model_name']}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(load_path_obj / "index.faiss"))
        
        # Load metadata
        with open(load_path_obj / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Database loaded with {len(self.metadata)} chunks")
    
    def query(self, query_text: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Query the vector database.
        
        Args:
            query_text: Text to search for
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        if self.index is None:
            raise ValueError("No database loaded. Please build or load a database first.")
        
        # Create query embedding
        query_embedding = self.model.encode([query_text], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= score_threshold and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary containing database statistics
        """
        if not self.metadata:
            return {"error": "No database loaded"}
        
        # Count chunks per file
        file_counts = {}
        total_chars = 0
        images_count = 0
        
        for chunk in self.metadata:
            filename = chunk['filename']
            file_counts[filename] = file_counts.get(filename, 0) + 1
            total_chars += chunk['length']
            if chunk.get('has_images', False):
                images_count += chunk.get('image_count', 0)
        
        return {
            'total_chunks': len(self.metadata),
            'total_characters': total_chars,
            'average_chunk_size': total_chars / len(self.metadata) if self.metadata else 0,
            'files_processed': len(file_counts),
            'chunks_per_file': file_counts,
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name,
            'total_images_processed': images_count,
            'image_processing_enabled': self.process_images
        }
    
    def analyze_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Analyze a PDF file to determine its characteristics.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF analysis results
        """
        analysis = {
            'filename': Path(pdf_path).name,
            'file_size_mb': round(Path(pdf_path).stat().st_size / (1024 * 1024), 2),
            'total_pages': 0,
            'text_extractable': False,
            'likely_scanned': False,
            'extraction_method': None,
            'sample_text': "",
            'image_count': 0,
            'has_images': False,
            'image_descriptions': []
        }
        
        try:
            # Try PyMuPDF for analysis
            doc = fitz.open(pdf_path)
            analysis['total_pages'] = len(doc)
            
            # Check for images
            total_images = 0
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images(full=True)
                total_images += len(image_list)
            
            analysis['image_count'] = total_images
            analysis['has_images'] = total_images > 0
            
            # Check first few pages for text
            text_pages = 0
            total_text_length = 0
            
            for page_num in range(min(3, len(doc))):  # Check first 3 pages
                page = doc.load_page(page_num)
                try:
                    page_text = page.get_text()
                except AttributeError:
                    # Fallback if get_text() is not available
                    page_text = page.get_text("text")
                
                if page_text and len(page_text.strip()) > 50:  # Substantial text
                    text_pages += 1
                    total_text_length += len(page_text)
                    if not analysis['sample_text']:
                        analysis['sample_text'] = page_text[:200] + "..."
            
            doc.close()
            
            if text_pages > 0:
                analysis['text_extractable'] = True
                analysis['extraction_method'] = 'PyMuPDF'
                
                # Heuristic: if very little text per page, might be scanned
                avg_text_per_page = total_text_length / max(text_pages, 1)
                if avg_text_per_page < 100:
                    analysis['likely_scanned'] = True
            else:
                analysis['likely_scanned'] = True
            
            # If image processing is enabled, get sample image descriptions
            if self.process_images and analysis['has_images']:
                try:
                    sample_images = self.extract_images_from_pdf(pdf_path)[:2]  # Just first 2 images
                    for img_data in sample_images:
                        description = self.describe_image(img_data['image'])
                        if description:
                            analysis['image_descriptions'].append({
                                'page': img_data['page_num'],
                                'description': description[:100] + "..."
                            })
                except Exception as e:
                    logger.warning(f"Error analyzing images in {pdf_path}: {e}")
                
        except Exception as e:
            logger.warning(f"Error analyzing PDF {pdf_path}: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def analyze_pdf_folder(self, pdf_folder) -> List[Dict[str, Any]]:
        """
        Analyze all PDF and image files in a folder.
        
        Args:
            pdf_folder: Path to folder containing PDF and image files (str or Path)
            
        Returns:
            List of analysis results for each file
        """
        pdf_folder_path = Path(pdf_folder)
        pdf_files = list(pdf_folder_path.glob("*.pdf"))
        
        # Find image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
            image_files.extend(pdf_folder_path.glob(ext))
        
        analyses = []
        
        # Analyze PDF files
        for pdf_path in pdf_files:
            analysis = self.analyze_pdf(str(pdf_path))
            analyses.append(analysis)
        
        # Analyze image files
        for image_path in image_files:
            analysis = self.analyze_image(str(image_path))
            analyses.append(analysis)
        
        return analyses
    
    def extract_text_from_image(self, image_path: str) -> tuple[str, List[Dict[str, Any]]]:
        """
        Extract text content from a standalone image file using enhanced OCR and image description.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tuple of (extracted text content, list of image descriptions)
        """
        text = ""
        image_descriptions = []
        
        try:
            # Load the image
            image = Image.open(image_path)
            filename = Path(image_path).name
            
            # Generate comprehensive image description if vision processing is enabled
            if self.process_images:
                description = self.describe_image(image)
                if description:
                    image_desc = {
                        'description': description,
                        'page_num': 1,  # Standalone images are treated as single page
                        'img_index': 0,
                        'source': 'standalone_image',
                        'dimensions': f"{image.width}x{image.height}"
                    }
                    image_descriptions.append(image_desc)
                    logger.info(f"Standalone image {filename}: {description[:150]}...")
            
            # Try enhanced OCR if available
            if OCR_AVAILABLE:
                try:
                    logger.info(f"Trying enhanced OCR extraction for {filename}")
                    ocr_text = self.extract_text_with_enhanced_ocr(image, filename)
                    
                    # Also try table extraction for better structured data
                    table_text = self.extract_table_data(image, filename)
                    
                    combined_text = ""
                    if ocr_text and ocr_text.strip():
                        combined_text = ocr_text.strip()
                    
                    # If table extraction found different/additional content, include it
                    if table_text and table_text.strip():
                        if combined_text:
                            combined_text += f"\n\n[STRUCTURED TABLE DATA]:\n{table_text.strip()}"
                        else:
                            combined_text = table_text.strip()
                    
                    if combined_text:
                        text = f"--- {filename} (Enhanced OCR) ---\n\n{combined_text}"
                        
                        # Add image description if available
                        if image_descriptions:
                            text += f"\n\n[IMAGE DESCRIPTION]:\n"
                            for img in image_descriptions:
                                text += f"- {img['description']}\n"
                        
                        logger.info(f"Successfully extracted text data from {filename}")
                        return text, image_descriptions
                        
                except Exception as e:
                    logger.warning(f"Enhanced OCR failed for {filename}: {e}")
                    
                # Fallback to basic OCR
                try:
                    logger.info(f"Trying basic OCR extraction for {filename}")
                    basic_ocr_text = pytesseract.image_to_string(image, lang='eng')
                    
                    if basic_ocr_text and basic_ocr_text.strip():
                        text = f"--- {filename} (Basic OCR) ---\n\n{basic_ocr_text.strip()}"
                        
                        # Add image description
                        if image_descriptions:
                            text += f"\n\n[IMAGE DESCRIPTION]:\n"
                            for img in image_descriptions:
                                text += f"- {img['description']}\n"
                        
                        logger.info(f"Successfully extracted text using basic OCR from {filename}")
                        return text, image_descriptions
                        
                except Exception as e:
                    logger.warning(f"Basic OCR failed for {filename}: {e}")
            
            # If OCR failed but we have image description, use that with more details
            if image_descriptions:
                text = f"--- {filename} (Vision Analysis Only) ---\n\n"
                text += f"Image Analysis: {image_descriptions[0]['description']}\n\n"
                
                # Add technical details
                text += f"Technical Details:\n"
                text += f"- Dimensions: {image.width}x{image.height} pixels\n"
                text += f"- Format: {image.format}\n"
                text += f"- Color Mode: {image.mode}\n"
                text += f"- File Size: {Path(image_path).stat().st_size / 1024:.1f} KB\n"
                
                # Add technical details
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    text += f"- Color Channels: {img_array.shape[2]}\n"
                text += f"- Pixel Density: {image.width * image.height} total pixels\n"
                
                return text, image_descriptions
            
            # If all methods failed
            logger.warning(f"Could not extract any meaningful data from {filename}")
            if not OCR_AVAILABLE:
                logger.info("OCR libraries not available. Install pytesseract for image text extraction.")
            if not self.process_images:
                logger.info("Image processing disabled. Enable it to get image descriptions.")
            
            return "", image_descriptions
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return "", []
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a standalone image file to determine its characteristics comprehensively.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with comprehensive image analysis results
        """
        analysis = {
            'filename': Path(image_path).name,
            'file_size_mb': round(Path(image_path).stat().st_size / (1024 * 1024), 2),
            'file_type': 'image',
            'total_pages': 1,  # Images are single "page"
            'text_extractable': False,
            'likely_scanned': True,  # Images are always considered "scanned"
            'extraction_method': None,
            'sample_text': "",
            'image_count': 1,
            'has_images': True,
            'image_descriptions': [],
            'text_density': 0,
            'estimated_text_length': 0
        }
        
        try:
            # Load and analyze the image
            image = Image.open(image_path)
            analysis['dimensions'] = f"{image.width}x{image.height}"
            analysis['format'] = image.format
            analysis['mode'] = image.mode
            analysis['pixel_count'] = image.width * image.height
            
            # Try enhanced OCR to see if there's extractable text
            if OCR_AVAILABLE:
                try:
                    # Test with basic OCR first
                    sample_text = pytesseract.image_to_string(image, lang='eng')
                    if sample_text and len(sample_text.strip()) > 10:
                        analysis['text_extractable'] = True
                        analysis['extraction_method'] = 'Basic OCR'
                        analysis['sample_text'] = sample_text[:300] + "..." if len(sample_text) > 300 else sample_text
                        analysis['estimated_text_length'] = len(sample_text)
                        analysis['text_density'] = len(sample_text.strip()) / (image.width * image.height) * 1000000  # text per megapixel
                    
                    # Try enhanced OCR for better results
                    enhanced_text = self.extract_text_with_enhanced_ocr(image, Path(image_path).name)
                    if enhanced_text and len(enhanced_text.strip()) > len(sample_text.strip()):
                        analysis['extraction_method'] = 'Enhanced OCR'
                        analysis['sample_text'] = enhanced_text[:300] + "..." if len(enhanced_text) > 300 else enhanced_text
                        analysis['estimated_text_length'] = len(enhanced_text)
                        analysis['text_density'] = len(enhanced_text.strip()) / (image.width * image.height) * 1000000
                        
                except Exception as e:
                    logger.warning(f"OCR analysis failed for {image_path}: {e}")
            
            # Generate comprehensive image description if enabled
            if self.process_images:
                try:
                    description = self.describe_image(image)
                    if description:
                        analysis['image_descriptions'].append({
                            'page': 1,
                            'description': description[:200] + "..." if len(description) > 200 else description
                        })
                except Exception as e:
                    logger.warning(f"Error analyzing image description for {image_path}: {e}")
                    
            # Add image analysis
            img_array = np.array(image)
            
            # Color analysis
            if image.mode == 'RGB':
                analysis['color_channels'] = 3
            elif image.mode == 'L':
                analysis['color_channels'] = 1
            
            # Estimate image complexity (standard deviation of pixel values)
            if len(img_array.shape) >= 2:
                complexity = np.std(img_array.flatten())
                analysis['image_complexity'] = f"{complexity:.1f}"
                
                # High complexity might indicate rich content
                if complexity > 50:
                    analysis['content_richness'] = 'High'
                elif complexity > 20:
                    analysis['content_richness'] = 'Medium'
                else:
                    analysis['content_richness'] = 'Low'
                    
        except Exception as e:
            logger.warning(f"Error analyzing image {image_path}: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def preprocess_image_for_ocr(self, image: Image.Image) -> List[Image.Image]:
        """
        Preprocess image using the exact method from imageextraction.py.
        
        Args:
            image: PIL Image object
            
        Returns:
            List of preprocessed images to try
        """
        if not OCR_AVAILABLE:
            return [image]
        
        try:
            # Use the exact preprocessing method from imageextraction.py
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale - exact method from imageextraction.py
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast - exact parameters from imageextraction.py
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Denoise - exact method from imageextraction.py
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # Threshold to get binary image - exact method from imageextraction.py
            _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL
            preprocessed_image = Image.fromarray(thresh)
            
            # Return both original and preprocessed
            return [image, preprocessed_image]
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return [image]

    def extract_text_with_enhanced_ocr(self, image: Image.Image, filename: str = "") -> str:
        """
        Advanced OCR text extraction using the exact methods from imageextraction.py.
        
        Args:
            image: PIL Image object
            filename: Name of the file for logging
            
        Returns:
            Extracted text content
        """
        if not OCR_AVAILABLE:
            return ""
        
        try:
            # STEP 1: Try with preprocessing - exactly like imageextraction.py
            preprocessed_images = self.preprocess_image_for_ocr(image)
            
            for i, test_image in enumerate(preprocessed_images):
                logger.info(f"Trying OCR with image variant {i+1} for {filename}")
                
                # Use the exact same method as imageextraction.py extract_all_text()
                ocr_data = pytesseract.image_to_data(test_image, output_type=pytesseract.Output.DICT)
                
                text_regions = []
                full_text = []
                
                for j in range(len(ocr_data['text'])):
                    text = ocr_data['text'][j].strip()
                    if text and int(ocr_data['conf'][j]) > 30:  # Confidence threshold from imageextraction.py
                        # Store the text region info (same as imageextraction.py)
                        text_regions.append({
                            'text': text,
                            'confidence': float(ocr_data['conf'][j]),
                            'bbox': (ocr_data['left'][j], ocr_data['top'][j], 
                                    ocr_data['width'][j], ocr_data['height'][j]),
                            'block_type': self._classify_text_block(text)
                        })
                        full_text.append(text)
                
                # Join all text with spaces (exactly like imageextraction.py)
                extracted_text = ' '.join(full_text)
                
                if extracted_text.strip() and len(extracted_text.strip()) > 20:  # Minimum meaningful text
                    logger.info(f"imageextraction.py method extracted {len(extracted_text)} characters from {filename} (variant {i+1})")
                    logger.info(f"Found {len(text_regions)} text regions with confidence > 30")
                    return extracted_text
            
            # STEP 2: Try with different tesseract configurations - like imageextraction.py
            logger.info(f"Trying fallback OCR configurations for {filename}")
            
            # Use the tesseract config from imageextraction.py
            tesseract_configs = [
                r'--oem 3 --psm 6',  # Default from imageextraction.py
                r'--oem 3 --psm 4',  # Single column text
                r'--oem 3 --psm 11', # Sparse text
                r'--oem 3 --psm 8',  # Single word
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-+/\|'  # Table config from imageextraction.py
            ]
            
            for config in tesseract_configs:
                for test_image in preprocessed_images:
                    try:
                        fallback_text = pytesseract.image_to_string(test_image, lang='eng', config=config)
                        if fallback_text and len(fallback_text.strip()) > 20:
                            logger.info(f"Fallback OCR with config '{config}' extracted {len(fallback_text)} characters from {filename}")
                            return fallback_text.strip()
                    except Exception:
                        continue
            
            # STEP 3: Try the ImageDataExtractor approach directly
            logger.info(f"Trying ImageDataExtractor approach for {filename}")
            result = self._extract_using_imageextraction_class(image, filename)
            if result:
                return result
            
            return ""
            
        except Exception as e:
            logger.warning(f"imageextraction.py method failed for {filename}: {e}")
            return ""
    
    def _extract_using_imageextraction_class(self, image: Image.Image, filename: str) -> str:
        """
        Use the exact ImageDataExtractor class approach from imageextraction.py
        """
        try:
            # Convert PIL to OpenCV format exactly like imageextraction.py
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Create a temporary extractor object with the same methods
            extractor_data = self._create_imageextraction_data(opencv_image, image)
            
            # Use the exact preprocessing method from imageextraction.py
            processed_image = self._preprocess_image_exact(opencv_image)
            processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            
            # Extract text using the exact method from imageextraction.py
            return self._extract_all_text_exact(processed_pil, filename)
            
        except Exception as e:
            logger.warning(f"ImageDataExtractor approach failed for {filename}: {e}")
            return ""
    
    def _create_imageextraction_data(self, opencv_image: np.ndarray, pil_image: Image.Image) -> Dict:
        """Create data structure similar to imageextraction.py ImageDataExtractor"""
        return {
            'image': opencv_image,
            'pil_image': pil_image,
            'height': opencv_image.shape[0],
            'width': opencv_image.shape[1],
            'tesseract_config': r'--oem 3 --psm 6',
            'table_config': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-+/\|'
        }
    
    def _preprocess_image_exact(self, opencv_image: np.ndarray) -> np.ndarray:
        """Exact preprocessing method from imageextraction.py"""
        # Convert to grayscale - exact method from imageextraction.py
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast - exact parameters from imageextraction.py
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise - exact method from imageextraction.py
        denoised = cv2.fastNlMeansDenoising(enhanced)
        
        # Threshold to get binary image - exact method from imageextraction.py
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to BGR for consistency
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
    def _extract_all_text_exact(self, pil_image: Image.Image, filename: str) -> str:
        """Extract all text using the exact method from imageextraction.py"""
        try:
            # Get detailed OCR data - exact method from imageextraction.py
            ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            text_regions = []
            full_text = []
            
            for i in range(len(ocr_data['text'])):
                text = ocr_data['text'][i].strip()
                if text and int(ocr_data['conf'][i]) > 30:  # Confidence threshold from imageextraction.py
                    text_regions.append({
                        'text': text,
                        'confidence': float(ocr_data['conf'][i]),
                        'bbox': (ocr_data['left'][i], ocr_data['top'][i], 
                                ocr_data['width'][i], ocr_data['height'][i]),
                        'block_type': self._classify_text_block(text)
                    })
                    full_text.append(text)
            
            # Join all text - exact method from imageextraction.py
            extracted_text = ' '.join(full_text)
            
            if extracted_text.strip():
                logger.info(f"Exact imageextraction.py method extracted {len(extracted_text)} characters from {filename}")
                return extracted_text
            
            return ""
            
        except Exception as e:
            logger.warning(f"Exact text extraction failed for {filename}: {e}")
            return ""
    
    def _classify_text_block(self, text: str) -> str:
        """Classify text block type (from imageextraction.py)"""
        if re.match(r'^\d+$', text):
            return 'number'
        elif re.match(r'^\d+\.\d+$', text):
            return 'decimal'
        elif re.match(r'^[A-Za-z]+$', text):
            return 'word'
        elif re.match(r'^[A-Za-z0-9\s]+$', text):
            return 'mixed'
        else:
            return 'special'
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace while preserving structure
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Clean each line but preserve meaningful spaces
            cleaned_line = ' '.join(line.split())
            if cleaned_line:  # Only add non-empty lines
                cleaned_lines.append(cleaned_line)
        
        # Join lines back together
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive newlines (more than 2 consecutive)
        import re
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        return cleaned_text.strip()
    
    def extract_table_data(self, image: Image.Image, filename: str = "") -> str:
        """
        Extract table data using the exact methods from imageextraction.py.
        
        Args:
            image: PIL Image object
            filename: Name of the file for logging
            
        Returns:
            Formatted table text
        """
        if not OCR_AVAILABLE:
            return ""
        
        try:
            # STEP 1: Try with preprocessing like imageextraction.py
            preprocessed_images = self.preprocess_image_for_ocr(image)
            
            for i, test_image in enumerate(preprocessed_images):
                logger.info(f"Trying table extraction with image variant {i+1} for {filename}")
                
                # Convert to OpenCV format
                opencv_image = cv2.cvtColor(np.array(test_image), cv2.COLOR_RGB2BGR)
                
                # Detect table structure using the exact method from imageextraction.py
                table_regions = self._detect_table_structure_exact(opencv_image)
                
                if table_regions:
                    logger.info(f"Found {len(table_regions)} table regions in {filename} (variant {i+1})")
                    
                    # Extract text from each table region using imageextraction.py method
                    all_table_text = []
                    for j, (x, y, w, h) in enumerate(table_regions):
                        # Crop the table region
                        table_image = opencv_image[y:y+h, x:x+w]
                        table_text = self._extract_table_data_exact(table_image, f"{filename} table {j+1}")
                        if table_text:
                            all_table_text.append(f"=== Table {j+1} ===\n{table_text}")
                    
                    if all_table_text:
                        result = '\n\n'.join(all_table_text)
                        logger.info(f"Table extraction successful for {filename} (variant {i+1})")
                        return result
                
                # Try extracting from the whole image using imageextraction.py method
                whole_image_text = self._extract_table_data_exact(opencv_image, f"{filename} whole image")
                if whole_image_text and len(whole_image_text.strip()) > 20:
                    logger.info(f"Whole image table extraction successful for {filename} (variant {i+1})")
                    return whole_image_text
            
            # STEP 2: Try with different table-specific configurations
            logger.info(f"Trying table-specific OCR configurations for {filename}")
            
            # Table-specific configurations from imageextraction.py
            table_configs = [
                r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-+/\|',
                r'--oem 3 --psm 6 -c preserve_interword_spaces=1',
                r'--oem 3 --psm 4 -c preserve_interword_spaces=1',
                r'--oem 3 --psm 6',
                r'--oem 3 --psm 4'
            ]
            
            for config in table_configs:
                for test_image in preprocessed_images:
                    try:
                        table_text = pytesseract.image_to_string(test_image, lang='eng', config=config)
                        if table_text and len(table_text.strip()) > 20:
                            logger.info(f"Table OCR with config '{config}' extracted {len(table_text)} characters from {filename}")
                            return table_text.strip()
                    except Exception:
                        continue
            
            # STEP 3: Try the advanced structured table extraction
            logger.info(f"Trying advanced structured table extraction for {filename}")
            result = self._extract_structured_table_exact(image, filename)
            if result:
                return result
            
            return ""
            
        except Exception as e:
            logger.warning(f"Table extraction failed for {filename}: {e}")
            return ""
    
    def _extract_structured_table_exact(self, image: Image.Image, filename: str) -> str:
        """
        Extract structured table data using the exact method from imageextraction.py
        """
        try:
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Use the exact table extraction method from imageextraction.py
            table_data = self._extract_table_data_imageextraction_style(opencv_image)
            
            if table_data:
                # Format the table data exactly like imageextraction.py
                return self._format_table_data_exact(table_data)
                
            return ""
            
        except Exception as e:
            logger.warning(f"Structured table extraction failed for {filename}: {e}")
            return ""
    
    def _extract_table_data_imageextraction_style(self, opencv_image: np.ndarray) -> List[Dict]:
        """
        Extract table data using the exact method from imageextraction.py extract_table_data()
        """
        try:
            # Convert to PIL Image for pytesseract - exact method from imageextraction.py
            pil_table = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
            
            # Extract text with bounding boxes - exact method from imageextraction.py
            data = pytesseract.image_to_data(pil_table, output_type=pytesseract.Output.DICT)
            
            # Group text by rows based on y-coordinates - exact method from imageextraction.py
            rows = defaultdict(list)
            for i, text in enumerate(data['text']):
                if text.strip() and int(data['conf'][i]) > 30:  # Same confidence threshold
                    y_coord = data['top'][i]
                    rows[y_coord].append({
                        'text': text.strip(),
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'confidence': data['conf'][i]
                    })
            
            # Sort rows by y-coordinate - exact method from imageextraction.py
            sorted_rows = sorted(rows.items())
            
            # Convert to structured table - exact method from imageextraction.py
            table_data = []
            for row_idx, (y_coord, row_items) in enumerate(sorted_rows):
                # Sort items in row by x-coordinate
                row_items.sort(key=lambda x: x['x'])
                
                row_data = {
                    'row_index': row_idx,
                    'cells': [item['text'] for item in row_items],
                    'cell_count': len(row_items),
                    'y_position': y_coord
                }
                table_data.append(row_data)
            
            return table_data
            
        except Exception as e:
            logger.warning(f"ImageExtraction-style table extraction failed: {e}")
            return []
    
    def _format_table_data_exact(self, table_data: List[Dict]) -> str:
        """
        Format table data exactly like imageextraction.py would format it
        """
        try:
            if not table_data:
                return ""
            
            # Format as structured text
            formatted_lines = []
            for row in table_data:
                if row['cells']:
                    # Join cells with | separator for table-like format
                    row_text = ' | '.join(row['cells'])
                    formatted_lines.append(row_text)
            
            return '\n'.join(formatted_lines) if formatted_lines else ""
            
        except Exception as e:
            logger.warning(f"Table formatting failed: {e}")
            return ""
    
    def _detect_table_structure_exact(self, opencv_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect table structure using the exact method from imageextraction.py"""
        try:
            # Preprocess image - exact method from imageextraction.py
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Detect horizontal lines - exact parameters from imageextraction.py
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Detect vertical lines - exact parameters from imageextraction.py
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine lines - exact method from imageextraction.py
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find table contours - exact method from imageextraction.py
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            table_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Same threshold as imageextraction.py
                    x, y, w, h = cv2.boundingRect(contour)
                    table_regions.append((x, y, w, h))
            
            return table_regions
            
        except Exception as e:
            logger.warning(f"Table structure detection failed: {e}")
            return []
    
    def _extract_table_data_exact(self, opencv_image: np.ndarray, table_name: str) -> str:
        """Extract table data using the exact method from imageextraction.py"""
        try:
            # Convert to PIL Image for pytesseract - exact method from imageextraction.py
            pil_table = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
            
            # Extract text with bounding boxes - exact method from imageextraction.py
            data = pytesseract.image_to_data(pil_table, output_type=pytesseract.Output.DICT)
            
            # Group text by rows based on y-coordinates - exact method from imageextraction.py
            rows = defaultdict(list)
            for i, text in enumerate(data['text']):
                if text.strip() and int(data['conf'][i]) > 30:  # Same confidence threshold
                    y_coord = data['top'][i]
                    rows[y_coord].append({
                        'text': text.strip(),
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'confidence': data['conf'][i]
                    })
            
            # Sort rows by y-coordinate - exact method from imageextraction.py
            sorted_rows = sorted(rows.items())
            
            # Convert to structured table - exact method from imageextraction.py
            table_data = []
            for row_idx, (y_coord, row_items) in enumerate(sorted_rows):
                # Sort items in row by x-coordinate
                row_items.sort(key=lambda x: x['x'])
                
                row_data = {
                    'row_index': row_idx,
                    'cells': [item['text'] for item in row_items],
                    'cell_count': len(row_items),
                    'y_position': y_coord
                }
                table_data.append(row_data)
            
            # Format the table data as text
            if table_data:
                table_lines = []
                for row in table_data:
                    if row['cells']:
                        # Join cells with | separator for table-like format
                        row_text = ' | '.join(row['cells'])
                        table_lines.append(row_text)
                
                if table_lines:
                    return '\n'.join(table_lines)
            
            # Fallback to simple OCR if structured extraction fails
            return pytesseract.image_to_string(pil_table, lang='eng', config='--psm 6').strip()
                
        except Exception as e:
            logger.warning(f"Table data extraction failed for {table_name}: {e}")
            return ""
    
    def _detect_table_structure(self, gray_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect table structure using line detection (from imageextraction.py)"""
        try:
            # Threshold the image
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Detect horizontal lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            
            # Detect vertical lines
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            # Combine lines to find table mask
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find table contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            table_regions = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Filter small regions
                    x, y, w, h = cv2.boundingRect(contour)
                    table_regions.append((x, y, w, h))
            
            return table_regions
            
        except Exception as e:
            logger.warning(f"Table structure detection failed: {e}")
            return []
    
    def _extract_table_with_structure(self, table_image: Image.Image, table_name: str) -> str:
        """Extract table data with spatial structure preservation"""
        try:
            # Use pytesseract to get detailed data
            data = pytesseract.image_to_data(table_image, output_type=pytesseract.Output.DICT)
            
            # Group text by rows based on y-coordinates
            rows = defaultdict(list)
            
            for i, text in enumerate(data['text']):
                if text.strip() and int(data['conf'][i]) > 30:
                    y_coord = data['top'][i]
                    rows[y_coord].append({
                        'text': text.strip(),
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'confidence': data['conf'][i]
                    })
            
            # Sort rows by y-coordinate and process
            sorted_rows = sorted(rows.items())
            table_lines = []
            
            for row_idx, (y_coord, row_items) in enumerate(sorted_rows):
                # Sort items in row by x-coordinate
                row_items.sort(key=lambda x: x['x'])
                
                # Extract high confidence text
                row_cells = [item['text'] for item in row_items if item['confidence'] > 40]
                
                if row_cells:
                    # Join cells with | separator for table-like format
                    row_text = ' | '.join(row_cells)
                    table_lines.append(row_text)
            
            if table_lines:
                return '\n'.join(table_lines)
            else:
                # Fallback to simple OCR
                return pytesseract.image_to_string(table_image, lang='eng', config='--psm 6').strip()
                
        except Exception as e:
            logger.warning(f"Structured table extraction failed for {table_name}: {e}")
            return ""
    
    def _extract_structured_table_improved(self, image: Image.Image, filename: str = "") -> str:
        """Extract table using pytesseract's structured data output (improved from imageextraction.py)"""
        try:
            # Enhanced preprocessing for tables
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive threshold for better table line detection
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Morphological operations to enhance table structure
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL
            processed_image = Image.fromarray(cleaned)
            
            # Extract with table-optimized configuration
            table_config = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
            
            # Get structured data
            data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT, config=table_config)
            
            # Process the data to maintain table structure
            text_blocks = []
            for i, text in enumerate(data['text']):
                if text.strip() and int(data['conf'][i]) > 25:  # Lower threshold for tables
                    text_blocks.append({
                        'text': text.strip(),
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i],
                        'conf': data['conf'][i]
                    })
            
            if not text_blocks:
                return ""
            
            # Group into lines with better tolerance for table layouts
            lines = {}
            for block in text_blocks:
                y_key = block['y'] // 10 * 10  # Group by approximate line
                if y_key not in lines:
                    lines[y_key] = []
                lines[y_key].append(block)
            
            # Sort lines by y-coordinate and format
            formatted_lines = []
            for y_coord in sorted(lines.keys()):
                line_blocks = sorted(lines[y_coord], key=lambda b: b['x'])
                
                # Create line text with spacing
                line_parts = []
                prev_x = 0
                
                for block in line_blocks:
                    # Add spacing based on x-position gaps
                    x_gap = block['x'] - prev_x
                    if prev_x > 0 and x_gap > 50:  # Significant gap, likely new column
                        line_parts.append(' | ')
                    elif prev_x > 0:
                        line_parts.append(' ')
                    
                    line_parts.append(block['text'])
                    prev_x = block['x'] + block['w']
                
                line_text = ''.join(line_parts).strip()
                if line_text:
                    formatted_lines.append(line_text)
            
            result = '\n'.join(formatted_lines)
            
            if result.strip():
                logger.info(f"Improved table extraction successful for {filename} (lines: {len(formatted_lines)})")
                return result.strip()
            
            # Final fallback
            return pytesseract.image_to_string(processed_image, lang='eng', config=table_config).strip()
            
        except Exception as e:
            logger.warning(f"Improved table extraction failed for {filename}: {e}")
            return ""
    
    def _extract_structured_table(self, image: Image.Image, filename: str = "") -> str:
        """Extract table using pytesseract's structured data output."""
        try:
            # Get detailed OCR data
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            # Group text by rows and columns based on coordinates
            rows = {}
            for i, text in enumerate(data['text']):
                if text.strip() and data['conf'][i] > 20:  # Filter low confidence
                    top = data['top'][i]
                    left = data['left'][i]
                    
                    # Group by rows (within 15 pixels vertically)
                    row_key = top // 15 * 15
                    if row_key not in rows:
                        rows[row_key] = []
                    
                    rows[row_key].append({
                        'text': text.strip(),
                        'left': left,
                        'top': top,
                        'conf': data['conf'][i]
                    })
            
            # Format as table
            formatted_rows = []
            for row_top in sorted(rows.keys()):
                # Sort items in row by left position (columns)
                row_items = sorted(rows[row_top], key=lambda x: x['left'])
                
                # Create columns with proper spacing
                if len(row_items) > 1:  # Likely a table row
                    row_text = ' | '.join([item['text'] for item in row_items])
                    formatted_rows.append(row_text)
                elif len(row_items) == 1:  # Single item, might be header
                    formatted_rows.append(row_items[0]['text'])
            
            return '\n'.join(formatted_rows) if formatted_rows else ""
            
        except Exception as e:
            logger.warning(f"Structured table extraction failed: {e}")
            return ""
    
    def _extract_table_with_preprocessing(self, image: Image.Image, filename: str = "") -> str:
        """Extract table with image preprocessing for better detection."""
        try:
            import cv2
            
            # Convert to grayscale
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines to create table structure
            table_structure = cv2.add(horizontal_lines, vertical_lines)
            
            # If we found table structure, enhance the image
            if np.sum(table_structure) > 1000:  # Threshold for table detection
                # Enhance the original image based on detected structure
                enhanced = cv2.add(img_array, table_structure)
                enhanced_img = Image.fromarray(255 - enhanced)  # Invert for OCR
                
                # Use table-specific OCR config
                config = '--psm 6 -c preserve_interword_spaces=1'
                text = pytesseract.image_to_string(enhanced_img, config=config)
                
                if text.strip():
                    return text.strip()
            
            return ""
            
        except Exception as e:
            logger.warning(f"Preprocessed table extraction failed: {e}")
            return ""
    
    def _extract_table_with_layout_analysis(self, image: Image.Image, filename: str = "") -> str:
        """Extract table using layout analysis and cell detection."""
        try:
            import cv2
            
            # Convert to OpenCV format
            gray = image.convert('L')
            img_array = np.array(gray)
            
            # Detect contours that might be table cells
            _, thresh = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size to find potential table cells
            min_area = 100
            max_area = image.width * image.height * 0.1
            
            cell_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Filter by aspect ratio (avoid very thin lines)
                    if w > 20 and h > 10 and w/h < 10 and h/w < 10:
                        cell_contours.append((x, y, w, h))
            
            # If we found potential cells, extract text from each
            if len(cell_contours) > 3:  # Minimum cells for a table
                # Sort cells by position (top to bottom, left to right)
                cell_contours.sort(key=lambda cell: (cell[1], cell[0]))
                
                cell_texts = []
                for x, y, w, h in cell_contours:
                    # Extract cell region
                    cell_region = img_array[y:y+h, x:x+w]
                    cell_img = Image.fromarray(cell_region)
                    
                    # OCR on individual cell
                    cell_text = pytesseract.image_to_string(cell_img, config='--psm 8').strip()
                    if cell_text:
                        cell_texts.append(cell_text)
                
                # Group cells into rows based on y-coordinate
                if cell_texts:
                    # This is a simplified approach - you could improve row detection
                    return ' | '.join(cell_texts)
            
            return ""
            
        except Exception as e:
            logger.warning(f"Layout analysis table extraction failed: {e}")
            return ""
    

    
def main():
    """
    Main function to demonstrate usage.
    """
    # Configuration
    PDF_FOLDER = "databrick"
    DB_SAVE_PATH = "vector_database"
    
    # Initialize the vector database
    vector_db = PDFVectorDatabase(
        model_name="all-MiniLM-L6-v2",  # Fast and good quality
        chunk_size=512,
        chunk_overlap=50,
        process_images=True  # Enable image processing
    )
    
    try:
        # Build the database
        print("Building vector database from PDF files...")
        vector_db.build_database(PDF_FOLDER, DB_SAVE_PATH)
        
        # Show statistics
        stats = vector_db.get_stats()
        print("\nDatabase Statistics:")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Files processed: {stats['files_processed']}")
        print(f"Average chunk size: {stats['average_chunk_size']:.1f} characters")
        print(f"Embedding dimension: {stats['embedding_dimension']}")
        print(f"Images processed: {stats.get('total_images_processed', 0)}")
        print(f"Image processing: {'Enabled' if stats.get('image_processing_enabled', False) else 'Disabled'}")
        
        print("\nChunks per file:")
        for filename, count in stats['chunks_per_file'].items():
            print(f"  {filename}: {count} chunks")
        
        # Example queries
        print("\n" + "="*50)
        print("EXAMPLE QUERIES")
        print("="*50)
        
        test_queries = [
            "PhD application",
            "academic transcript",
            "research experience",
            "education background"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = vector_db.query(query, top_k=3)
            
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i} (Score: {result['similarity_score']:.3f}):")
                print(f"    File: {result['filename']}")
                print(f"    Text preview: {result['text'][:150]}...")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
