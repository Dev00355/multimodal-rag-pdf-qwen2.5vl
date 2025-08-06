#!/usr/bin/env python3
"""
Simple installation script for the Multimodal RAG system.
Optimized for Python 3.11+
"""

import subprocess
import sys
import os
import platform

def check_python_version():
    """Check if Python version is 3.11+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print(f"❌ Python 3.11+ required, found {version.major}.{version.minor}.{version.micro}")
        print("Please upgrade Python to 3.11 or higher")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Core dependencies first
    core_deps = [
        "fastapi>=0.104.1",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "requests>=2.31.0",
        "aiofiles>=23.2.0",
        "python-multipart>=0.0.6"
    ]
    
    # Install core dependencies
    for dep in core_deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
            print(f"✅ Installed {dep.split('>=')[0]}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    # Basic ML dependencies
    ml_deps = [
        "numpy>=1.24.3",
        "pillow>=10.1.0",
    ]
    
    for dep in ml_deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
            print(f"✅ Installed {dep.split('>=')[0]}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {dep}: {e}")
            return False
    
    # Specialized dependencies
    special_deps = [
        "chromadb>=0.4.18",
        "ollama>=0.1.7",
        "pypdf2>=3.0.1",
        "pymupdf>=1.23.8",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
    ]
    
    for dep in special_deps:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
            print(f"✅ Installed {dep.split('>=')[0]}")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Failed to install {dep}: {e}")
            print("   You may need to install this manually")
    
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ["chroma_db", "uploads", "logs"]
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main installation function"""
    print("🚀 Multimodal RAG System Installation")
    print("=" * 50)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Some dependencies failed to install")
        print("You may need to install them manually or check your internet connection")
    
    print("\n✅ Installation completed!")
    print("\n📋 Next steps:")
    print("1. Install Ollama: https://ollama.ai")
    print("2. Pull Qwen2.5-VL model: ollama pull qwen2.5vl:7b")
    print("3. Run the system: python main.py")
    print("4. Visit: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
