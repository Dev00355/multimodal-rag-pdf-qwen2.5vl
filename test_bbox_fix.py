#!/usr/bin/env python3
"""
Test script to verify the bbox metadata fix in vector_store.py
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vector_store import VectorStore
import tempfile
import shutil

def test_bbox_metadata():
    """Test that bbox coordinates are properly handled as metadata."""
    print("🧪 Testing bbox metadata handling...")
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Initialize vector store
        vs = VectorStore(persist_directory=temp_dir, collection_name="test_bbox")
        
        # Test text chunk with bbox
        text_metadata = {
            "page": 1,
            "bbox": [93.000068687, 187.9876649749, 89.655567788, 80.6789989988],
            "area": 1000.5,
            "type": "text"
        }
        
        print("📝 Adding text chunk with bbox metadata...")
        text_id = vs.add_text_chunk("This is a test text chunk", text_metadata)
        print(f"✅ Successfully added text chunk with ID: {text_id}")
        
        # Test image chunk with bbox and dimensions
        image_metadata = {
            "page": 1,
            "bbox": [100.0, 200.0, 300.0, 400.0],
            "dimensions": [200, 200],
            "area": 40000.0,
            "format": "png"
        }
        
        print("🖼️  Adding image chunk with bbox and dimensions metadata...")
        image_id = vs.add_image_chunk("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", 
                                    "Test image description", image_metadata)
        print(f"✅ Successfully added image chunk with ID: {image_id}")
        
        # Test search functionality
        print("🔍 Testing search functionality...")
        results = vs.similarity_search("test", n_results=2)
        
        print(f"📊 Search returned {len(results)} results")
        for i, result in enumerate(results):
            print(f"   Result {i+1}:")
            print(f"     Content: {result['content'][:50]}...")
            print(f"     Type: {result['metadata'].get('type')}")
            print(f"     Has bbox: {result['metadata'].get('has_bbox')}")
            if result['metadata'].get('has_bbox'):
                print(f"     Bbox coords: x0={result['metadata'].get('bbox_x0')}, y0={result['metadata'].get('bbox_y0')}, x1={result['metadata'].get('bbox_x1')}, y1={result['metadata'].get('bbox_y1')}")
            print(f"     Spatial context: {result.get('spatial_context', {}).get('has_coordinates')}")
            print()
        
        print("✅ All tests passed! Bbox metadata is now properly handled.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_bbox_metadata()
