#!/usr/bin/env python3
"""
Debug script to identify and fix the image data serialization issue.
"""

import sys
import os
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multimodal_rag import MultimodalRAG
from vector_store import VectorStore
import tempfile
import shutil

def debug_image_processing():
    """Debug the image processing pipeline."""
    print("🔍 Debugging Image Processing Pipeline")
    print("=" * 60)
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize system
        rag = MultimodalRAG(persist_directory=temp_dir, collection_name="debug_test")
        
        # Check if there are any PDFs to test with
        pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
        
        if not pdf_files:
            print("❌ No PDF files found for testing")
            print("   Please add a PDF file with images to test the system")
            return
        
        pdf_path = pdf_files[0]
        print(f"📄 Testing with PDF: {pdf_path}")
        
        # Process PDF
        print("\n📊 Processing PDF...")
        result = rag.process_pdf(pdf_path)
        
        print(f"   Text chunks: {len(result.get('text_chunks', []))}")
        print(f"   Images: {len(result.get('images', []))}")
        
        # Check vector store contents
        print("\n🗄️  Checking Vector Store Contents...")
        vs = VectorStore(persist_directory=temp_dir, collection_name="debug_test")
        all_docs = vs.get_all_documents(limit=100)
        
        text_count = 0
        image_count = 0
        invalid_images = 0
        
        for doc in all_docs:
            doc_type = doc['metadata'].get('type', 'unknown')
            if doc_type == 'text':
                text_count += 1
            elif doc_type == 'image':
                image_count += 1
                img_data = doc['metadata'].get('image_data')
                if not img_data or not rag._validate_base64_image(img_data):
                    invalid_images += 1
                    print(f"   ⚠️  Invalid image found: page {doc['metadata'].get('page')}, length: {len(img_data) if img_data else 0}")
        
        print(f"   Stored text chunks: {text_count}")
        print(f"   Stored images: {image_count}")
        print(f"   Invalid images: {invalid_images}")
        
        # Test query
        print("\n🔍 Testing Query with Images...")
        try:
            response = rag.query("What can you see in the images?", include_images=True, n_results=3)
            print(f"   Query successful: {response['answer'][:100]}...")
            print(f"   Sources used: {response['sources']}")
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test query without images
        print("\n📝 Testing Query without Images...")
        try:
            response = rag.query("What is this document about?", include_images=False, n_results=3)
            print(f"   Query successful: {response['answer'][:100]}...")
            print(f"   Sources used: {response['sources']}")
        except Exception as e:
            print(f"   ❌ Query failed: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_base64_validation():
    """Test the base64 validation functions."""
    print("\n🧪 Testing Base64 Validation Functions")
    print("=" * 50)
    
    from multimodal_rag import MultimodalRAG
    from ollama_client import OllamaClient
    
    rag = MultimodalRAG()
    ollama = OllamaClient()
    
    # Test cases
    test_cases = [
        ("Valid PNG", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="),
        ("Empty string", ""),
        ("Invalid base64", "not_base64_data"),
        ("Too short", "abc"),
        ("None", None),
    ]
    
    for name, data in test_cases:
        try:
            rag_valid = rag._validate_base64_image(data)
            ollama_valid = ollama._validate_image_data(data)
            print(f"   {name}: RAG={rag_valid}, Ollama={ollama_valid}")
        except Exception as e:
            print(f"   {name}: Error - {e}")

if __name__ == "__main__":
    debug_image_processing()
    test_base64_validation()
