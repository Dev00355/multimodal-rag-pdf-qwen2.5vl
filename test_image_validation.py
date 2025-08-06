#!/usr/bin/env python3
"""
Test script to verify the image data validation fixes.
"""

import sys
import os
import base64

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multimodal_rag import MultimodalRAG
from ollama_client import OllamaClient

def test_image_validation():
    """Test image data validation functions."""
    print("🧪 Testing Image Data Validation...")
    print("=" * 50)
    
    # Initialize components
    rag = MultimodalRAG()
    ollama_client = OllamaClient()
    
    # Test cases
    test_cases = [
        ("Valid base64", "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==", True),
        ("Empty string", "", False),
        ("None value", None, False),
        ("Invalid base64", "invalid_base64_data", False),
        ("Too short", "abc", False),
        ("Non-string", 123, False),
    ]
    
    print("📝 Testing MultimodalRAG validation:")
    for name, data, expected in test_cases:
        try:
            result = rag._validate_base64_image(data)
            status = "✅" if result == expected else "❌"
            print(f"   {status} {name}: {result} (expected: {expected})")
        except Exception as e:
            print(f"   ❌ {name}: Error - {e}")
    
    print("\n🔧 Testing OllamaClient validation:")
    for name, data, expected in test_cases:
        try:
            result = ollama_client._validate_image_data(data)
            status = "✅" if result == expected else "❌"
            print(f"   {status} {name}: {result} (expected: {expected})")
        except Exception as e:
            print(f"   ❌ {name}: Error - {e}")
    
    print("\n🔍 Testing with actual image data:")
    # Create a simple test image (1x1 pixel PNG)
    test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    print(f"   Valid image data length: {len(test_image_b64)}")
    print(f"   RAG validation: {rag._validate_base64_image(test_image_b64)}")
    print(f"   Ollama validation: {ollama_client._validate_image_data(test_image_b64)}")
    
    # Test decoding
    try:
        decoded = base64.b64decode(test_image_b64, validate=True)
        print(f"   Decoded size: {len(decoded)} bytes")
        print("   ✅ Base64 decoding successful")
    except Exception as e:
        print(f"   ❌ Base64 decoding failed: {e}")
    
    print("\n✅ Image validation tests completed!")

if __name__ == "__main__":
    test_image_validation()
