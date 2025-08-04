import os
import logging
from typing import List, Dict, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

from pdf_processor import PDFProcessor
from ollama_client import OllamaClient
from vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalRAG:
    """Main multimodal RAG system orchestrating PDF processing, vector storage, and querying."""
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "multimodal_rag",
                 ollama_model: str = "qwen2.5-vl:7b",
                 ollama_host: str = "http://localhost:11434"):
        
        self.pdf_processor = PDFProcessor()
        self.ollama_client = OllamaClient(model_name=ollama_model, host=ollama_host)
        self.vector_store = VectorStore(persist_directory, collection_name, ollama_host)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Check if Ollama model is available
        if not self.ollama_client.check_model_availability():
            logger.warning(f"Model {ollama_model} not found. Please run: ollama pull {ollama_model}")
    
    async def process_pdf_async(self, pdf_path: str, document_id: str = None) -> Dict:
        """
        Asynchronously process a PDF document and add to vector store.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Optional document identifier
            
        Returns:
            Processing results summary
        """
        try:
            start_time = time.time()
            
            if document_id is None:
                document_id = os.path.basename(pdf_path).replace('.pdf', '')
            
            logger.info(f"Starting processing of {pdf_path}")
            
            # Extract content from PDF
            loop = asyncio.get_event_loop()
            extracted_data = await loop.run_in_executor(
                self.executor, 
                self.pdf_processor.process_pdf, 
                pdf_path
            )
            
            results = {
                'document_id': document_id,
                'filename': extracted_data['metadata']['filename'],
                'total_pages': extracted_data['metadata']['total_pages'],
                'text_chunks_added': 0,
                'image_chunks_added': 0,
                'processing_time': 0,
                'errors': []
            }
            
            # Process text chunks
            for text_chunk in extracted_data['text_chunks']:
                try:
                    # Generate summary for better retrieval
                    summary = await loop.run_in_executor(
                        self.executor,
                        self.ollama_client.generate_text_summary,
                        text_chunk['content']
                    )
                    
                    # Combine original text with summary for embedding
                    enhanced_content = f"{text_chunk['content']}\n\nSummary: {summary}"
                    
                    metadata = {
                        'document_id': document_id,
                        'page': text_chunk['page'],
                        'chunk_type': 'text',
                        'filename': extracted_data['metadata']['filename'],
                        'summary': summary
                    }
                    
                    self.vector_store.add_text_chunk(enhanced_content, metadata)
                    results['text_chunks_added'] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing text chunk on page {text_chunk['page']}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            # Process image chunks
            for image_chunk in extracted_data['images']:
                try:
                    # Analyze image with Qwen2.5-VL
                    image_analysis = await loop.run_in_executor(
                        self.executor,
                        self.ollama_client.analyze_image,
                        image_chunk['content']
                    )
                    
                    # Generate embedding-friendly description
                    embedding_description = await loop.run_in_executor(
                        self.executor,
                        self.ollama_client.generate_image_embeddings_description,
                        image_chunk['content']
                    )
                    
                    # Extract text from image if any
                    extracted_text = await loop.run_in_executor(
                        self.executor,
                        self.ollama_client.extract_image_text,
                        image_chunk['content']
                    )
                    
                    # Combine all descriptions
                    full_description = f"""
Image Analysis: {image_analysis}

Embedding Description: {embedding_description}

Extracted Text: {extracted_text}
"""
                    
                    metadata = {
                        'document_id': document_id,
                        'page': image_chunk['page'],
                        'image_index': image_chunk['image_index'],
                        'chunk_type': 'image',
                        'filename': extracted_data['metadata']['filename'],
                        'analysis': image_analysis,
                        'extracted_text': extracted_text
                    }
                    
                    self.vector_store.add_image_chunk(
                        full_description, 
                        image_chunk['content'], 
                        metadata
                    )
                    results['image_chunks_added'] += 1
                    
                except Exception as e:
                    error_msg = f"Error processing image {image_chunk['image_index']} on page {image_chunk['page']}: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            results['processing_time'] = time.time() - start_time
            
            logger.info(f"Completed processing {pdf_path}: "
                       f"{results['text_chunks_added']} text chunks, "
                       f"{results['image_chunks_added']} image chunks, "
                       f"{results['processing_time']:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def process_pdf(self, pdf_path: str, document_id: str = None) -> Dict:
        """
        Synchronous wrapper for PDF processing.
        
        Args:
            pdf_path: Path to the PDF file
            document_id: Optional document identifier
            
        Returns:
            Processing results summary
        """
        return asyncio.run(self.process_pdf_async(pdf_path, document_id))
    
    async def query_async(self, 
                         query: str, 
                         n_results: int = 5,
                         include_images: bool = True,
                         document_filter: str = None) -> Dict:
        """
        Asynchronously query the multimodal RAG system.
        
        Args:
            query: User's question
            n_results: Number of results to retrieve
            include_images: Whether to include images in the response
            document_filter: Optional document ID filter
            
        Returns:
            Query response with answer and sources
        """
        try:
            start_time = time.time()
            
            # Prepare metadata filter
            metadata_filter = {}
            if document_filter:
                metadata_filter['document_id'] = document_filter
            
            # Perform hybrid search
            if include_images:
                text_results, image_base64_list = self.vector_store.hybrid_search(
                    query, n_results
                )
            else:
                text_results = self.vector_store.similarity_search(
                    query, n_results, metadata_filter
                )
                image_base64_list = []
            
            # Generate answer using Ollama
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                self.executor,
                self.ollama_client.answer_question,
                query,
                text_results,
                image_base64_list if include_images else None
            )
            
            # Prepare response
            response = {
                'query': query,
                'answer': answer,
                'sources': {
                    'text_chunks': len(text_results),
                    'images': len(image_base64_list),
                    'total_sources': len(text_results) + len(image_base64_list)
                },
                'text_sources': [
                    {
                        'content': result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                        'page': result['metadata'].get('page'),
                        'document': result['metadata'].get('filename'),
                        'distance': result['distance']
                    }
                    for result in text_results
                ],
                'processing_time': time.time() - start_time
            }
            
            if include_images and image_base64_list:
                response['has_images'] = True
                response['image_count'] = len(image_base64_list)
            else:
                response['has_images'] = False
                response['image_count'] = 0
            
            logger.info(f"Query processed in {response['processing_time']:.2f}s: "
                       f"{response['sources']['total_sources']} sources used")
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            raise
    
    def query(self, 
              query: str, 
              n_results: int = 5,
              include_images: bool = True,
              document_filter: str = None) -> Dict:
        """
        Synchronous wrapper for querying.
        
        Args:
            query: User's question
            n_results: Number of results to retrieve
            include_images: Whether to include images in the response
            document_filter: Optional document ID filter
            
        Returns:
            Query response with answer and sources
        """
        return asyncio.run(self.query_async(query, n_results, include_images, document_filter))
    
    def get_system_stats(self) -> Dict:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            vector_stats = self.vector_store.get_collection_stats()
            model_available = self.ollama_client.check_model_availability()
            
            stats = {
                'vector_store': vector_stats,
                'ollama_model_available': model_available,
                'ollama_model': self.ollama_client.model_name,
                'system_ready': model_available and vector_stats.get('total_documents', 0) > 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}
    
    def clear_database(self) -> bool:
        """
        Clear the vector database.
        
        Returns:
            True if successful
        """
        try:
            return self.vector_store.clear_collection()
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False
