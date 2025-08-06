import sys

# Check Python version
if sys.version_info < (3, 11):
    print("❌ Python 3.11 or higher is required")
    print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    sys.exit(1)

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os
import tempfile
import logging
from typing import Optional, List
import aiofiles
from datetime import datetime

from multimodal_rag import MultimodalRAG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal RAG API",
    description="A multimodal RAG system for PDFs with text and images using Qwen2.5-VL",
    version="1.0.0"
)

# Initialize RAG system
rag_system = MultimodalRAG()

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    n_results: Optional[int] = 5
    include_images: Optional[bool] = True
    document_filter: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: dict
    text_sources: List[dict]
    has_images: bool
    image_count: int
    processing_time: float

class UploadResponse(BaseModel):
    message: str
    document_id: str
    filename: str
    total_pages: int
    text_chunks_added: int
    image_chunks_added: int
    spatial_relationships_added: int
    processing_time: float
    errors: List[str]

class SystemStats(BaseModel):
    vector_store: dict
    ollama_model_available: bool
    ollama_model: str
    system_ready: bool

# Global variable to track upload status
upload_status = {}

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multimodal RAG API with Qwen2.5-VL",
        "version": "1.0.0",
        "endpoints": {
            "upload": "POST /upload - Upload PDF documents",
            "query": "POST /query - Query the RAG system",
            "health": "GET /health - Health check",
            "stats": "GET /stats - System statistics",
            "docs": "GET /docs - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = rag_system.get_system_stats()
        return {
            "status": "healthy" if stats.get('system_ready', False) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "ollama_available": stats.get('ollama_model_available', False),
            "documents_indexed": stats.get('vector_store', {}).get('total_documents', 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics."""
    try:
        stats = rag_system.get_system_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_pdf_background(file_path: str, document_id: str, original_filename: str):
    """Background task to process PDF."""
    try:
        upload_status[document_id] = {
            "status": "processing",
            "filename": original_filename,
            "started_at": datetime.now().isoformat()
        }
        
        # Process the PDF
        result = await rag_system.process_pdf_async(file_path, document_id)
        
        upload_status[document_id] = {
            "status": "completed",
            "filename": original_filename,
            "started_at": upload_status[document_id]["started_at"],
            "completed_at": datetime.now().isoformat(),
            "result": result
        }
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
            
    except Exception as e:
        logger.error(f"Error processing PDF in background: {e}")
        upload_status[document_id] = {
            "status": "failed",
            "filename": original_filename,
            "started_at": upload_status[document_id]["started_at"],
            "failed_at": datetime.now().isoformat(),
            "error": str(e)
        }
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_id: Optional[str] = None,
    process_async: Optional[bool] = True
):
    """
    Upload and process a PDF document.
    
    Args:
        file: PDF file to upload
        document_id: Optional custom document ID
        process_async: Whether to process asynchronously (default: True)
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate document ID if not provided
        if document_id is None:
            document_id = file.filename.replace('.pdf', '').replace(' ', '_')
        
        # Save uploaded file temporarily
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{document_id}_{file.filename}")
        
        async with aiofiles.open(temp_file_path, 'wb') as temp_file:
            content = await file.read()
            await temp_file.write(content)
        
        if process_async:
            # Process asynchronously
            background_tasks.add_task(
                process_pdf_background, 
                temp_file_path, 
                document_id, 
                file.filename
            )
            
            return UploadResponse(
                message="PDF upload successful. Processing started in background.",
                document_id=document_id,
                filename=file.filename,
                total_pages=0,  # Will be updated when processing completes
                text_chunks_added=0,
                image_chunks_added=0,
                spatial_relationships_added=0,
                processing_time=0.0,
                errors=[]
            )
        else:
            # Process synchronously
            result = await rag_system.process_pdf_async(temp_file_path, document_id)
            
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            return UploadResponse(
                message="PDF processed successfully",
                document_id=result['document_id'],
                filename=result['filename'],
                total_pages=result['total_pages'],
                text_chunks_added=result['text_chunks_added'],
                image_chunks_added=result['image_chunks_added'],
                spatial_relationships_added=result.get('spatial_relationships_added', 0),
                processing_time=result['processing_time'],
                errors=result['errors']
            )
            
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        # Clean up temporary file if it exists
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/upload/status/{document_id}")
async def get_upload_status(document_id: str):
    """Get the status of a PDF upload/processing."""
    if document_id not in upload_status:
        raise HTTPException(status_code=404, detail="Document ID not found")
    
    return upload_status[document_id]

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Query the multimodal RAG system.
    
    Args:
        request: Query request with question and parameters
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if system is ready
        stats = rag_system.get_system_stats()
        if not stats.get('ollama_model_available', False):
            raise HTTPException(
                status_code=503, 
                detail="Ollama model not available. Please ensure Qwen2.5-VL is installed."
            )
        
        if stats.get('vector_store', {}).get('total_documents', 0) == 0:
            raise HTTPException(
                status_code=404, 
                detail="No documents indexed. Please upload PDF documents first."
            )
        
        # Process query
        result = await rag_system.query_async(
            query=request.query,
            n_results=request.n_results,
            include_images=request.include_images,
            document_filter=request.document_filter
        )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents")
async def clear_documents():
    """Clear all documents from the vector database."""
    try:
        success = rag_system.clear_database()
        if success:
            return {"message": "All documents cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear documents")
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    try:
        stats = rag_system.get_system_stats()
        return {
            "total_documents": stats.get('vector_store', {}).get('total_documents', 0),
            "text_chunks": stats.get('vector_store', {}).get('text_chunks', 0),
            "image_chunks": stats.get('vector_store', {}).get('image_chunks', 0)
        }
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Check if Ollama is available
    try:
        stats = rag_system.get_system_stats()
        if not stats.get('ollama_model_available', False):
            logger.warning("⚠️  Qwen2.5-VL model not found!")
            logger.warning("Please run: ollama pull qwen2.5vl:7b")
            logger.warning("The API will start but queries will fail until the model is available.")
    except Exception as e:
        logger.error(f"Error checking system status: {e}")
    
    logger.info("🚀 Starting Multimodal RAG API server...")
    logger.info("📚 Upload PDFs at: http://localhost:8000/upload")
    logger.info("❓ Query system at: http://localhost:8000/query")
    logger.info("📖 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
