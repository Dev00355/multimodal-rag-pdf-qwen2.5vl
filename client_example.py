#!/usr/bin/env python3
"""
Example client for interacting with the Multimodal RAG API.
This script shows how to upload PDFs and query the system via HTTP API.
"""

import requests
import json
import time
import os
from typing import Optional

class MultimodalRAGClient:
    """Client for interacting with the Multimodal RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> dict:
        """Check the health status of the API."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        try:
            response = requests.get(f"{self.base_url}/stats")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def upload_pdf(self, 
                   pdf_path: str, 
                   document_id: Optional[str] = None,
                   process_async: bool = True) -> dict:
        """
        Upload a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Optional custom document ID
            process_async: Whether to process asynchronously
            
        Returns:
            Upload response
        """
        try:
            if not os.path.exists(pdf_path):
                return {"error": f"File not found: {pdf_path}"}
            
            with open(pdf_path, 'rb') as file:
                files = {'file': (os.path.basename(pdf_path), file, 'application/pdf')}
                data = {}
                
                if document_id:
                    data['document_id'] = document_id
                if not process_async:
                    data['process_async'] = 'false'
                
                response = requests.post(
                    f"{self.base_url}/upload",
                    files=files,
                    data=data
                )
                
                return response.json()
                
        except Exception as e:
            return {"error": str(e)}
    
    def get_upload_status(self, document_id: str) -> dict:
        """Get the status of a document upload/processing."""
        try:
            response = requests.get(f"{self.base_url}/upload/status/{document_id}")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def query(self, 
              query: str,
              n_results: int = 5,
              include_images: bool = True,
              document_filter: Optional[str] = None) -> dict:
        """
        Query the RAG system.
        
        Args:
            query: Question to ask
            n_results: Number of results to retrieve
            include_images: Whether to include images in the response
            document_filter: Optional document ID filter
            
        Returns:
            Query response
        """
        try:
            payload = {
                "query": query,
                "n_results": n_results,
                "include_images": include_images
            }
            
            if document_filter:
                payload["document_filter"] = document_filter
            
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            return response.json()
            
        except Exception as e:
            return {"error": str(e)}
    
    def list_documents(self) -> dict:
        """List all indexed documents."""
        try:
            response = requests.get(f"{self.base_url}/documents")
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def clear_documents(self) -> dict:
        """Clear all documents from the database."""
        try:
            response = requests.delete(f"{self.base_url}/documents")
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def main():
    """Example usage of the client."""
    print("🔌 Multimodal RAG API Client Example")
    print("=" * 50)
    
    # Initialize client
    client = MultimodalRAGClient()
    
    # Check health
    print("\n🏥 Health Check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    
    if health.get("status") != "healthy" and health.get("status") != "degraded":
        print("❌ API is not healthy. Make sure the server is running.")
        return
    
    # Get system stats
    print("\n📊 System Statistics:")
    stats = client.get_stats()
    print(json.dumps(stats, indent=2))
    
    # Example PDF upload (replace with actual PDF path)
    pdf_path = "sample.pdf"
    if os.path.exists(pdf_path):
        print(f"\n📤 Uploading PDF: {pdf_path}")
        upload_result = client.upload_pdf(pdf_path, process_async=True)
        print(json.dumps(upload_result, indent=2))
        
        if "document_id" in upload_result:
            document_id = upload_result["document_id"]
            
            # Check upload status
            print(f"\n⏳ Checking upload status for: {document_id}")
            for i in range(10):  # Check for up to 10 times
                status = client.get_upload_status(document_id)
                print(f"Status: {status.get('status', 'unknown')}")
                
                if status.get('status') == 'completed':
                    print("✅ Processing completed!")
                    break
                elif status.get('status') == 'failed':
                    print("❌ Processing failed!")
                    print(f"Error: {status.get('error', 'Unknown error')}")
                    break
                
                time.sleep(5)  # Wait 5 seconds before checking again
    else:
        print(f"\n📄 Sample PDF not found at {pdf_path}")
        print("To test upload, place a PDF file named 'sample.pdf' in this directory")
    
    # List documents
    print("\n📚 Listing Documents:")
    docs = client.list_documents()
    print(json.dumps(docs, indent=2))
    
    # Example queries
    if docs.get("total_documents", 0) > 0:
        print("\n❓ Example Queries:")
        
        example_queries = [
            "What is this document about?",
            "Are there any diagrams or images?",
            "What are the main topics discussed?",
            "Summarize the key findings"
        ]
        
        for query in example_queries:
            print(f"\n  Query: {query}")
            result = client.query(query, n_results=3)
            
            if "error" in result:
                print(f"  ❌ Error: {result['error']}")
            else:
                print(f"  📝 Answer: {result.get('answer', 'No answer')[:200]}...")
                print(f"  📊 Sources: {result.get('sources', {}).get('total_sources', 0)}")
                print(f"  ⏱️  Time: {result.get('processing_time', 0):.2f}s")
    else:
        print("\n❓ No documents available for querying")
    
    print("\n✅ Client example completed!")
    print("\n💡 API Documentation available at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
