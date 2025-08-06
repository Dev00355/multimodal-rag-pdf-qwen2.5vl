#!/usr/bin/env python3
"""
Test script for the Multimodal RAG system.
This script demonstrates how to use the system programmatically.
"""

import sys

# Check Python version
if sys.version_info < (3, 11):
    print("❌ Python 3.11 or higher is required")
    print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    sys.exit(1)

import asyncio
import logging
from multimodal_rag import MultimodalRAG
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_system():
    """Test the multimodal RAG system."""
    
    print("🚀 Initializing Multimodal RAG System...")
    rag = MultimodalRAG()
    
    # Check system status
    print("\n📊 System Status:")
    stats = rag.get_system_stats()
    print(f"  - Ollama Model Available: {stats.get('ollama_model_available', False)}")
    print(f"  - Model: {stats.get('ollama_model', 'Unknown')}")
    print(f"  - Documents Indexed: {stats.get('vector_store', {}).get('total_documents', 0)}")
    print(f"  - Text Chunks: {stats.get('vector_store', {}).get('text_chunks', 0)}")
    print(f"  - Image Chunks: {stats.get('vector_store', {}).get('image_chunks', 0)}")
    
    if not stats.get('ollama_model_available', False):
        print("\n⚠️  Qwen2.5-VL model not available!")
        print("Please run: ollama pull qwen2.5vl:7b")
        return
    
    # Test with sample PDF (if available)
    sample_pdf_path = "sample.pdf"  # Replace with actual PDF path
    
    if os.path.exists(sample_pdf_path):
        print(f"\n📄 Processing sample PDF: {sample_pdf_path}")
        try:
            result = await rag.process_pdf_async(sample_pdf_path, "sample_doc")
            print(f"  ✅ Processing completed:")
            print(f"     - Pages: {result['total_pages']}")
            print(f"     - Text chunks: {result['text_chunks_added']}")
            print(f"     - Image chunks: {result['image_chunks_added']}")
            print(f"     - Processing time: {result['processing_time']:.2f}s")
            if result['errors']:
                print(f"     - Errors: {len(result['errors'])}")
        except Exception as e:
            print(f"  ❌ Error processing PDF: {e}")
    else:
        print(f"\n📄 Sample PDF not found at {sample_pdf_path}")
        print("To test PDF processing, place a PDF file named 'sample.pdf' in this directory")
    
    # Test querying (if documents are available)
    updated_stats = rag.get_system_stats()
    if updated_stats.get('vector_store', {}).get('total_documents', 0) > 0:
        print("\n❓ Testing queries...")
        
        test_queries = [
            "What is the main topic of the document?",
            "Are there any diagrams or charts in the document?",
            "What technical concepts are discussed?",
            "Summarize the key points from the images"
        ]
        
        for query in test_queries:
            print(f"\n  Query: {query}")
            try:
                response = await rag.query_async(query, n_results=3)
                print(f"  Answer: {response['answer'][:200]}...")
                print(f"  Sources: {response['sources']['total_sources']} "
                      f"({response['sources']['text_chunks']} text, "
                      f"{response['sources']['images']} images)")
                print(f"  Time: {response['processing_time']:.2f}s")
            except Exception as e:
                print(f"  ❌ Error: {e}")
    else:
        print("\n❓ No documents available for querying")
        print("Upload some PDF documents first to test the query functionality")
    
    print("\n✅ System test completed!")
    print("\n💡 To use the API server:")
    print("   python main.py")
    print("   Then visit http://localhost:8000/docs for interactive API documentation")

def main():
    """Main function to run the test."""
    print("🧪 Multimodal RAG System Test")
    print("=" * 50)
    
    try:
        asyncio.run(test_system())
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.error(f"Test error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
