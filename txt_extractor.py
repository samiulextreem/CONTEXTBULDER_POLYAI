#!/usr/bin/env python3
"""
Export Text for API Upload

This script exports extracted text from your documents into clean text files
that can be easily uploaded to APIs like OpenAI, Anthropic, Google, etc.
Only processes new or modified files to avoid reprocessing.
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
import sys

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_db_builder import PDFVectorDatabase

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

def export_text_for_api(input_folder: str = "databrick", output_folder: str = "api_ready_texts"):
    """
    Extract text from documents and export as clean files for API upload
    Only processes new or modified files to avoid reprocessing.
    
    Args:
        input_folder: Folder containing PDF/image files
        output_folder: Folder to save exported text files
    """
    
    print("=" * 80)
    print("EXPORTING TEXT FOR API UPLOAD (INCREMENTAL)")
    print("=" * 80)
    
    # Create output directory
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Load tracking file
    tracking_file = output_path / "processed_files.json"
    processed_files = load_processed_files(tracking_file)
    
    # Initialize extractor
    print("🔧 Initializing text extractor...")
    extractor = PDFVectorDatabase(
        model_name="all-MiniLM-L6-v2",  # Only for chunking, not for API
        chunk_size=2000,  # Larger chunks for API upload
        chunk_overlap=100,
        process_images=True
    )
    
    # Find files to process
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"❌ ERROR: Input folder not found: {input_folder}")
        return
    
    # Find all files
    pdf_files = list(input_path.glob("*.pdf"))
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
        image_files.extend(input_path.glob(ext))
    
    all_files = pdf_files + image_files
    
    if not all_files:
        print(f"❌ ERROR: No PDF or image files found in {input_folder}")
        return
    
    # Filter to only new/modified files
    files_to_process = []
    for file_path in all_files:
        if should_process_file(file_path, processed_files):
            files_to_process.append(file_path)
    
    print(f"📁 Found {len(all_files)} total files")
    print(f"📁 Already processed: {len(all_files) - len(files_to_process)}")
    print(f"🆕 New/Modified files to process: {len(files_to_process)}")
    
    if not files_to_process:
        print("✅ No new files to process. All files are up to date!")
        return
    
    print(f"   PDFs: {len([f for f in files_to_process if f.suffix.lower() == '.pdf'])}")
    print(f"   Images: {len([f for f in files_to_process if f.suffix.lower() != '.pdf'])}")
    print()
    
    # Process each new/modified file
    exported_files = []
    total_text_length = 0
    
    for file_path in files_to_process:
        print(f"📄 Processing: {file_path.name}")
        
        try:
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text, image_descriptions = extractor.extract_text_from_pdf(str(file_path))
            else:
                text, image_descriptions = extractor.extract_text_from_image(str(file_path))
            
            if not text:
                print(f"   ⚠️  No text extracted from {file_path.name}")
                # Still mark as processed to avoid retrying
                processed_files[file_path.name] = {
                    'mtime': file_path.stat().st_mtime,
                    'hash': get_file_hash(file_path),
                    'processed_at': datetime.now().isoformat(),
                    'text_length': 0,
                    'status': 'no_text'
                }
                continue
            
            # Clean the text for API upload
            cleaned_text = clean_text_for_api(text)
            
            # Store extracted text for combined file
            exported_files.append({
                'source': file_path.name,
                'text_content': cleaned_text,
                'text_length': len(cleaned_text),
                'image_count': len(image_descriptions),
                'has_images': len(image_descriptions) > 0,
                'file_type': 'pdf' if file_path.suffix.lower() == '.pdf' else 'image'
            })
            
            total_text_length += len(cleaned_text)
            
            # Update processed files tracking
            processed_files[file_path.name] = {
                'mtime': file_path.stat().st_mtime,
                'hash': get_file_hash(file_path),
                'processed_at': datetime.now().isoformat(),
                'text_length': len(cleaned_text),
                'status': 'success'
            }
            
            print(f"   ✅ Extracted: {len(cleaned_text)} characters")
            
        except Exception as e:
            print(f"   ❌ ERROR processing {file_path.name}: {e}")
            # Mark as processed with error to avoid retrying
            processed_files[file_path.name] = {
                'mtime': file_path.stat().st_mtime,
                'hash': get_file_hash(file_path),
                'processed_at': datetime.now().isoformat(),
                'text_length': 0,
                'status': 'error',
                'error': str(e)
            }
            continue
    
    # Save tracking file
    save_processed_files(tracking_file, processed_files)
    
    # Only append to combined file if we have new content
    if exported_files:
        # Create combined text file for bulk upload (append mode)
        combined_file = output_path / "all_documents_combined.txt"
        
        with open(combined_file, 'a', encoding='utf-8') as f:
            # Add session header
            f.write("\n" + "="*80 + "\n")
            f.write("NEW EXTRACTION SESSION\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"New/Modified documents in this session: {len(exported_files)}\n")
            f.write("="*80 + "\n\n")
            
            for i, file_info in enumerate(exported_files, 1):
                f.write(f"\n{'='*60}\n")
                f.write(f"DOCUMENT {i}: {file_info['source']}\n")
                f.write(f"Length: {file_info['text_length']} characters\n")
                f.write(f"Extracted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")
                f.write(file_info['text_content'])
                f.write(f"\n\n{'='*60}\n")
                f.write(f"END OF DOCUMENT {i}\n")
                f.write(f"{'='*60}\n\n")
    
    # Print final summary
    print("\n" + "="*80)
    print("EXPORT COMPLETE!")
    print("="*80)
    print(f"📊 New files processed: {len(exported_files)}")
    print(f"📊 Total text length (new): {total_text_length:,} characters")
    print(f"📊 Total files tracked: {len(processed_files)}")
    print(f"📁 Output folder: {output_path.absolute()}")
    
    if exported_files:
        print(f"📄 Combined file: all_documents_combined.txt (NEW CONTENT APPENDED)")
    else:
        print(f"📄 Combined file: all_documents_combined.txt (NO NEW CONTENT)")
    
    print(f"🗂️  Tracking file: {tracking_file.name}")
    print()
    print("🚀 FILE READY FOR API UPLOAD:")
    print(f"   • all_documents_combined.txt (incremental updates)")
    print()
    print("💡 USAGE TIPS:")
    print("   • Only new/modified files are processed each run")
    print("   • Tracking file keeps record of processed files")
    print("   • Delete tracking file to reprocess all files")
    print("   • Each document is clearly separated with headers")
    print("="*80)

def clean_text_for_api(text: str) -> str:
    """Clean extracted text for better API processing"""
    
    # Remove excessive whitespace while preserving structure
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Clean each line but preserve meaningful spaces
        cleaned_line = ' '.join(line.split())
        if cleaned_line:  # Only add non-empty lines
            cleaned_lines.append(cleaned_line)
    
    # Join lines back together with single newlines
    cleaned_text = '\n'.join(cleaned_lines)
    
    # Remove excessive newlines (more than 2 consecutive)
    import re
    cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
    
    # Remove special markers that might confuse APIs
    cleaned_text = cleaned_text.replace('--- ', '')
    cleaned_text = cleaned_text.replace(' (Enhanced OCR) ---', '')
    cleaned_text = cleaned_text.replace(' (OCR) ---', '')
    
    return cleaned_text.strip()



if __name__ == "__main__":
    # Export text files with command line argument support
    import sys
    
    # Default values
    input_folder = "databrick"
    output_folder = "api_ready_texts"
    
    # Check for command line arguments
    if len(sys.argv) >= 2:
        input_folder = sys.argv[1]
    if len(sys.argv) >= 3:
        output_folder = sys.argv[2]
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print()
    
    # Export text files
    export_text_for_api(input_folder, output_folder) 