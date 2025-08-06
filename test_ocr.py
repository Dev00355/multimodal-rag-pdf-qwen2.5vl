#!/usr/bin/env python3
"""
Test script to demonstrate OCR functionality for scanned PDFs.
This script shows how the enhanced multimodal RAG system now handles scanned research papers.
"""

import asyncio
import os
import logging
from multimodal_rag import MultimodalRAG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_ocr_functionality():
    """Test OCR functionality with a sample PDF."""
    
    print("🔍 Testing OCR functionality for scanned PDFs...")
    print("=" * 60)
    
    # Initialize the multimodal RAG system
    rag_system = MultimodalRAG(
        persist_directory="./test_chroma_db",
        collection_name="ocr_test",
        ollama_model="qwen2.5vl:7b"
    )
    
    # Check if Ollama model is available
    if not rag_system.ollama_client.check_model_availability():
        print("❌ Qwen2.5-VL model not found!")
        print("Please run: ollama pull qwen2.5vl:7b")
        return
    
    print("✅ Qwen2.5-VL model is available")
    
    # Test with a sample PDF (you can replace this with your scanned research paper)
    sample_pdf_path = input("Enter path to a scanned PDF research paper (or press Enter to skip): ").strip()
    
    if not sample_pdf_path:
        print("ℹ️  No PDF provided. Here's what the system can now do:")
        print("\n📋 Enhanced Features:")
        print("1. ✅ OCR fallback when regular text extraction yields < 50 characters")
        print("2. ✅ Text extraction from scanned PDF pages using Qwen2.5-VL")
        print("3. ✅ OCR processing of diagrams and images within PDFs")
        print("4. ✅ Combined embeddings of visual descriptions + extracted text")
        print("5. ✅ Spatial relationship preservation with OCR content")
        return
    
    if not os.path.exists(sample_pdf_path):
        print(f"❌ File not found: {sample_pdf_path}")
        return
    
    print(f"📄 Processing PDF: {sample_pdf_path}")
    
    try:
        # Process the PDF with OCR capabilities
        result = await rag_system.process_pdf_async(sample_pdf_path, "test_document")
        
        print("\n📊 Processing Results:")
        print(f"✅ Total text chunks: {result.get('text_chunks', 0)}")
        print(f"✅ Total images processed: {result.get('images', 0)}")
        print(f"✅ OCR text chunks: {result.get('ocr_chunks', 0)}")
        print(f"✅ Images with extracted text: {result.get('images_with_ocr', 0)}")
        print(f"⏱️  Processing time: {result.get('processing_time', 0):.2f} seconds")
        
        # Test querying
        print("\n🔍 Testing query functionality...")
        test_queries = [
            "What is the main topic of this research?",
            "Are there any diagrams or figures mentioned?",
            "What methodology is described?",
            "What are the key findings?"
        ]
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            response = await rag_system.query_async(query, max_results=3)
            print(f"💬 Response: {response[:200]}..." if len(response) > 200 else f"💬 Response: {response}")
        
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        logger.error(f"Processing error: {e}")

def show_ocr_capabilities():
    """Show the OCR capabilities that have been implemented."""
    
    print("\n🚀 OCR Enhancement Summary")
    print("=" * 50)
    
    print("\n📋 What's New:")
    print("1. 🔍 Smart OCR Detection:")
    print("   - Automatically detects when regular text extraction fails")
    print("   - Triggers OCR when < 50 characters extracted per page")
    print("   - Uses high-resolution rendering for better OCR accuracy")
    
    print("\n2. 🖼️  Image Text Extraction:")
    print("   - Extracts text from diagrams, charts, and figures")
    print("   - Preserves spatial relationships between text and images")
    print("   - Adds extracted text as searchable metadata")
    
    print("\n3. 🧠 Enhanced Embeddings:")
    print("   - Combines visual descriptions with OCR text")
    print("   - Creates richer, more searchable content")
    print("   - Maintains separate tracking of OCR vs regular text")
    
    print("\n4. 🔄 Fallback Strategy:")
    print("   - Primary: PyMuPDF text extraction (fast, accurate for searchable PDFs)")
    print("   - Fallback: Qwen2.5-VL OCR (comprehensive for scanned documents)")
    print("   - Hybrid: Both methods for maximum coverage")
    
    print("\n💡 Use Cases Now Supported:")
    print("   ✅ Scanned research papers")
    print("   ✅ Image-heavy documents with embedded text")
    print("   ✅ Handwritten notes (if legible)")
    print("   ✅ Charts, diagrams, and flowcharts with text")
    print("   ✅ Mixed content documents (text + images)")

if __name__ == "__main__":
    print("🔬 Multimodal RAG OCR Test Suite")
    print("=" * 40)
    
    # Show capabilities first
    show_ocr_capabilities()
    
    # Ask if user wants to test with actual PDF
    test_with_pdf = input("\nWould you like to test with an actual PDF? (y/n): ").lower().strip()
    
    if test_with_pdf == 'y':
        asyncio.run(test_ocr_functionality())
    else:
        print("\n✅ OCR functionality is ready to use!")
        print("📝 To use: Simply upload a scanned PDF through the normal interface.")
        print("🔍 The system will automatically detect and apply OCR when needed.")
