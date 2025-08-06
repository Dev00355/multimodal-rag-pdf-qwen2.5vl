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
                 ollama_model: str = "qwen2.5vl:7b",
                 ollama_host: str = "http://localhost:11434"):
        
        self.ollama_client = OllamaClient(model_name=ollama_model, host=ollama_host)
        self.pdf_processor = PDFProcessor(ollama_client=self.ollama_client)
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
                        'summary': summary,
                        'bbox': text_chunk.get('bbox'),
                        'block_index': text_chunk.get('block_index'),
                        'area': text_chunk.get('area', 0)
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
                        'extracted_text': extracted_text,
                        'bbox': image_chunk.get('bbox'),
                        'dimensions': image_chunk.get('dimensions', []),
                        'area': image_chunk.get('area', 0)
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
            
            # Process spatial relationships
            if 'spatial_relationships' in extracted_data:
                try:
                    self.vector_store.add_spatial_relationships(extracted_data['spatial_relationships'])
                    results['spatial_relationships_added'] = len(extracted_data['spatial_relationships'])
                    logger.info(f"Added {len(extracted_data['spatial_relationships'])} spatial relationships")
                except Exception as e:
                    error_msg = f"Error processing spatial relationships: {e}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
                    results['spatial_relationships_added'] = 0
            else:
                results['spatial_relationships_added'] = 0
            
            results['processing_time'] = time.time() - start_time
            
            logger.info(f"Completed processing {pdf_path}: "
                       f"{results['text_chunks_added']} text chunks, "
                       f"{results['image_chunks_added']} image chunks, "
                       f"{results['spatial_relationships_added']} spatial relationships, "
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
            
            # Perform enhanced spatial-aware search
            if include_images:
                # Get mixed results with spatial awareness
                all_results = self.vector_store.similarity_search(
                    query, n_results * 2, use_spatial_boost=True
                )
                
                # Separate text and image results
                text_results = [r for r in all_results if r['metadata'].get('type') == 'text']
                image_results = [r for r in all_results if r['metadata'].get('type') == 'image']
                
                # Balance results and extract images
                text_results = text_results[:max(1, n_results // 2)]
                image_results = image_results[:max(1, n_results // 2)]
                
                # Extract base64 images for vision model
                image_base64_list = []
                logger.debug(f"Processing {len(image_results)} image results for query")
                
                for i, img_result in enumerate(image_results):
                    img_data = img_result['metadata'].get('image_data')
                    page = img_result['metadata'].get('page', 'unknown')
                    
                    if not img_data:
                        logger.warning(f"No image data found for image {i} from page {page}")
                        continue
                        
                    if self._validate_base64_image(img_data):
                        image_base64_list.append(img_data)
                        logger.debug(f"Added valid image {i} from page {page} (length: {len(img_data)})")
                    else:
                        logger.warning(f"Invalid image data found, skipping image {i} from page {page} (length: {len(img_data) if img_data else 0})")
                
                logger.info(f"Using {len(image_base64_list)} valid images out of {len(image_results)} found")
                
                # Combine results for context
                combined_results = text_results + image_results
            else:
                # Text-only search with spatial awareness
                combined_results = self.vector_store.similarity_search(
                    query, n_results, content_type='text', use_spatial_boost=True
                )
                text_results = combined_results
                image_results = []
                image_base64_list = []
            
            # Generate answer using Ollama with enhanced context
            loop = asyncio.get_event_loop()
            answer = await loop.run_in_executor(
                self.executor,
                self.ollama_client.answer_question,
                query,
                combined_results if include_images else text_results,
                image_base64_list if include_images else None
            )
            
            # Prepare enhanced response with spatial information
            all_sources = combined_results if include_images else text_results
            
            response = {
                'query': query,
                'answer': answer,
                'sources': {
                    'text_chunks': len(text_results),
                    'images': len(image_results) if include_images else 0,
                    'total_sources': len(all_sources),
                    'spatial_aware': True
                },
                'text_sources': [
                    {
                        'content': result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                        'page': result['metadata'].get('page'),
                        'document': result['metadata'].get('filename'),
                        'similarity_score': result.get('similarity_score', 0),
                        'spatial_boost': result.get('spatial_boost', 0),
                        'final_score': result.get('final_score', 0),
                        'has_spatial_info': result.get('has_spatial_info', False),
                        'spatial_context': result.get('spatial_context', {})
                    }
                    for result in text_results
                ],
                'processing_time': time.time() - start_time
            }
            
            # Add image sources if included
            if include_images and image_results:
                response['image_sources'] = [
                    {
                        'description': result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                        'page': result['metadata'].get('page'),
                        'document': result['metadata'].get('filename'),
                        'similarity_score': result.get('similarity_score', 0),
                        'spatial_boost': result.get('spatial_boost', 0),
                        'final_score': result.get('final_score', 0),
                        'spatial_context': result.get('spatial_context', {}),
                        'analysis': result['metadata'].get('analysis', ''),
                        'extracted_text': result['metadata'].get('extracted_text', '')
                    }
                    for result in image_results
                ]
            
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
    
    def _validate_base64_image(self, image_data: str) -> bool:
        """
        Validate that the image data is a proper base64 encoded string.
        
        Args:
            image_data: Base64 encoded image string
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not image_data or not isinstance(image_data, str):
                return False
                
            # Check if it's a valid base64 string
            import base64
            base64.b64decode(image_data, validate=True)
            
            # Check minimum length (a valid image should be reasonably sized)
            if len(image_data) < 100:  # Very small base64 strings are likely invalid
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Invalid base64 image data: {e}")
            return False
