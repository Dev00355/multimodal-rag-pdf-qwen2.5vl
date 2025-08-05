#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced spatial features of the multimodal RAG system.
This script shows how the system now captures bounding boxes and spatial relationships.
"""

import sys

# Check Python version
if sys.version_info < (3, 11):
    print("❌ Python 3.11 or higher is required")
    print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    sys.exit(1)

import asyncio
import logging
from pdf_processor import PDFProcessor
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_pdf():
    """Create a simple sample PDF for testing (placeholder)."""
    print("📄 For testing, please provide a PDF file with both text and images.")
    print("   The system will extract:")
    print("   - Text chunks with bounding boxes")
    print("   - Images with bounding boxes") 
    print("   - Spatial relationships between text and images")
    return None

def demonstrate_spatial_extraction():
    """Demonstrate the enhanced spatial extraction capabilities."""
    print("🧪 Testing Enhanced PDF Processor with Spatial Features")
    print("=" * 60)
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Check for sample PDF
    sample_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    
    if not sample_files:
        print("❌ No PDF files found in current directory")
        print("   Please add a PDF file to test the spatial features")
        return
    
    # Use the first PDF found
    pdf_path = sample_files[0]
    print(f"📖 Processing: {pdf_path}")
    
    try:
        # Process PDF with enhanced features
        result = processor.process_pdf(pdf_path)
        
        print(f"\n📊 Processing Results:")
        print(f"   📄 Total pages: {result['metadata']['total_pages']}")
        print(f"   📝 Text chunks: {len(result['text_chunks'])}")
        print(f"   🖼️  Images: {len(result['images'])}")
        print(f"   🔗 Spatial relationships: {len(result['spatial_relationships'])}")
        
        # Show text chunks with spatial info
        print(f"\n📝 Text Chunks with Spatial Information:")
        for i, chunk in enumerate(result['text_chunks'][:3]):  # Show first 3
            bbox = chunk.get('bbox', 'No bbox')
            area = chunk.get('area', 0)
            print(f"   Chunk {i+1}:")
            print(f"     Page: {chunk['page']}")
            print(f"     Content: {chunk['content'][:100]}...")
            print(f"     Bounding box: {bbox}")
            print(f"     Area: {area:.1f} pixels²")
            print()
        
        # Show images with spatial info
        print(f"🖼️  Images with Spatial Information:")
        for i, image in enumerate(result['images'][:2]):  # Show first 2
            bbox = image.get('bbox', 'No bbox')
            dimensions = image.get('dimensions', [])
            area = image.get('area', 0)
            print(f"   Image {i+1}:")
            print(f"     Page: {image['page']}")
            print(f"     Bounding box: {bbox}")
            print(f"     Dimensions: {dimensions}")
            print(f"     Area: {area:.1f} pixels²")
            print()
        
        # Show spatial relationships
        print(f"🔗 Spatial Relationships:")
        for i, rel in enumerate(result['spatial_relationships'][:5]):  # Show first 5
            print(f"   Relationship {i+1}:")
            print(f"     Text: {rel['text_chunk_id']}")
            print(f"     Image: {rel['image_id']}")
            print(f"     Type: {rel['relationship_type']}")
            print(f"     Distance: {rel['distance']:.1f} pixels")
            print(f"     Confidence: {rel['confidence']:.2f}")
            print(f"     Page: {rel['page']}")
            print()
        
        # Demonstrate spatial analysis
        print(f"🔍 Spatial Analysis Summary:")
        relationship_types = {}
        for rel in result['spatial_relationships']:
            rel_type = rel['relationship_type']
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        for rel_type, count in relationship_types.items():
            print(f"   {rel_type}: {count} relationships")
        
        # Show high-confidence relationships
        high_confidence = [r for r in result['spatial_relationships'] if r['confidence'] > 0.8]
        print(f"   High confidence relationships (>0.8): {len(high_confidence)}")
        
        print(f"\n✅ Enhanced spatial processing complete!")
        print(f"   The system now understands document layout and can:")
        print(f"   - Link text to nearby images")
        print(f"   - Identify captions and labels") 
        print(f"   - Understand spatial document structure")
        print(f"   - Provide context-aware retrieval")
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        print(f"❌ Error: {e}")

def demonstrate_bbox_calculations():
    """Demonstrate bounding box calculations."""
    print(f"\n📐 Bounding Box Coordinate System:")
    print(f"   PDF coordinates: [x0, y0, x1, y1]")
    print(f"   - (x0, y0): Top-left corner")
    print(f"   - (x1, y1): Bottom-right corner")
    print(f"   - Width: x1 - x0")
    print(f"   - Height: y1 - y0")
    print(f"   - Area: (x1 - x0) * (y1 - y0)")
    
    # Example calculations
    example_bbox = [100, 200, 400, 250]
    width = example_bbox[2] - example_bbox[0]
    height = example_bbox[3] - example_bbox[1]
    area = width * height
    
    print(f"\n   Example bbox {example_bbox}:")
    print(f"   - Width: {width} pixels")
    print(f"   - Height: {height} pixels") 
    print(f"   - Area: {area} pixels²")

def main():
    """Main test function."""
    print("🚀 Enhanced Multimodal RAG - Spatial Features Test")
    print("=" * 60)
    
    # Demonstrate spatial extraction
    demonstrate_spatial_extraction()
    
    # Show coordinate system
    demonstrate_bbox_calculations()
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Upload a PDF with both text and images")
    print(f"   2. The system will extract spatial coordinates")
    print(f"   3. Query the system - it will use spatial relationships")
    print(f"   4. Get more accurate, context-aware responses")

if __name__ == "__main__":
    main()
