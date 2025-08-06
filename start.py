#!/usr/bin/env python3
"""
Startup script for the Multimodal RAG system.
Checks all requirements and starts the server.
"""

import sys
import subprocess
import requests
import time
import os

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 11):
        print("❌ Python 3.11 or higher is required")
        print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'ollama', 'chromadb', 
        'langchain', 'pymupdf', 'sentence_transformers'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Run: python install.py")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is running and required models are available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model['name'] for model in models.get('models', [])]
            
            # Check for Qwen2.5-VL model
            qwen_models = [name for name in model_names if 'qwen2.5-vl' in name.lower()]
            nomic_models = [name for name in model_names if 'nomic-embed-text' in name.lower()]
            
            missing_models = []
            
            if qwen_models:
                print(f"✅ Qwen2.5-VL model: {qwen_models[0]}")
            else:
                print("❌ Qwen2.5-VL model not found")
                missing_models.append("qwen2.5vl:7b")
            
            if nomic_models:
                print(f"✅ Nomic Embed Text model: {nomic_models[0]}")
            else:
                print("❌ Nomic Embed Text model not found")
                missing_models.append("nomic-embed-text")
            
            if missing_models:
                print("\nMissing models. Run:")
                for model in missing_models:
                    print(f"  ollama pull {model}")
                return False
            
            print("✅ All required models available")
            return True
        else:
            print("❌ Ollama not responding correctly")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Ollama not running")
        print("Please start Ollama first")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def create_directories():
    """Create required directories"""
    dirs = ["chroma_db", "uploads", "logs"]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)

def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting Multimodal RAG API server...")
    print("📚 Upload PDFs at: http://localhost:8000/upload")
    print("❓ Query system at: http://localhost:8000/query")
    print("📖 API docs at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")

def main():
    """Main startup function"""
    print("🚀 Multimodal RAG System Startup Check")
    print("=" * 50)
    
    # Check Python version
    print("\n🐍 Checking Python version...")
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    print("✅ Directories ready")
    
    # Check Ollama
    print("\n🤖 Checking Ollama...")
    if not check_ollama():
        print("\n⚠️  Ollama setup required:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Start Ollama service")
        print("3. Pull required models:")
        print("   ollama pull qwen2.5vl:7b")
        print("   ollama pull nomic-embed-text")
        sys.exit(1)
    
    print("\n✅ All checks passed!")
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
