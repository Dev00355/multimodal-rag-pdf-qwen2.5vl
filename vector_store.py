import chromadb
from chromadb.config import Settings
import ollama
import uuid
import logging
from typing import List, Dict, Optional, Tuple
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Vector store using ChromaDB for multimodal RAG system."""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "multimodal_rag", ollama_host: str = "http://localhost:11434"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.ollama_host = ollama_host
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize Ollama client for embeddings
        self.ollama_client = ollama.Client(host=ollama_host)
        self.embedding_model = "nomic-embed-text"
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Multimodal RAG collection for PDFs with text and images"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding using Ollama Nomic Embed Text model."""
        try:
            response = self.ollama_client.embeddings(
                model=self.embedding_model,
                prompt=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Fallback to a zero vector if embedding fails
            return [0.0] * 768  # Nomic embed text dimension
    
    def add_text_chunk(self, content: str, metadata: Dict, chunk_id: str = None) -> str:
        """
        Add a text chunk to the vector store with spatial coordinates.
        
        Args:
            content: Text content to add
            metadata: Metadata associated with the chunk (should include bbox, page, etc.)
            chunk_id: Optional custom ID for the chunk
            
        Returns:
            ID of the added chunk
        """
        try:
            if chunk_id is None:
                chunk_id = str(uuid.uuid4())
            
            # Generate embedding using Ollama
            embedding = self._generate_embedding(content)
            
            # Enhance metadata with spatial information
            enhanced_metadata = {
                **{k: v for k, v in metadata.items() if k != "bbox"},  # Exclude bbox list
                "type": "text",
                "has_bbox": "bbox" in metadata and metadata["bbox"] is not None,
                "area": metadata.get("area", 0)
            }
            
            # Convert bbox to individual scalar values if present
            if "bbox" in metadata and metadata["bbox"] is not None:
                bbox = metadata["bbox"]
                enhanced_metadata.update({
                    "bbox_x0": float(bbox[0]),
                    "bbox_y0": float(bbox[1]),
                    "bbox_x1": float(bbox[2]),
                    "bbox_y1": float(bbox[3]),
                    "bbox_width": float(bbox[2] - bbox[0]),
                    "bbox_height": float(bbox[3] - bbox[1])
                })
            
            # Add to collection
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[enhanced_metadata],
                ids=[chunk_id]
            )
            
            logger.debug(f"Added text chunk with ID: {chunk_id}, bbox: {metadata.get('bbox', 'None')}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error adding text chunk: {e}")
            raise
    
    def add_image_chunk(self, image_base64: str, image_description: str, metadata: Dict, chunk_id: str = None) -> str:
        """
        Add an image chunk to the vector store with spatial coordinates.
        
        Args:
            image_base64: Base64 encoded image data
            image_description: Description of the image for embedding
            metadata: Metadata associated with the image (should include bbox, page, etc.)
            chunk_id: Optional custom ID for the chunk
            
        Returns:
            ID of the added chunk
        """
        try:
            if chunk_id is None:
                chunk_id = str(uuid.uuid4())
            
            # Check if image has OCR-extracted text
            extracted_text = metadata.get('extracted_text', '')
            
            # Create combined content for embedding (description + OCR text)
            embedding_content = image_description
            if extracted_text and extracted_text.strip():
                embedding_content = f"{image_description}\n\nExtracted text: {extracted_text.strip()}"
                logger.info(f"Image chunk includes OCR text: {len(extracted_text)} characters")
            
            # Generate embedding from combined content
            embedding = self._generate_embedding(embedding_content)
            
            # Store image data in metadata with spatial information
            image_metadata = {
                **{k: v for k, v in metadata.items() if k not in ["bbox", "dimensions", "extracted_text"]},  # Exclude list values
                "type": "image",
                "image_data": image_base64,
                "description": image_description,
                "has_bbox": "bbox" in metadata and metadata["bbox"] is not None,
                "area": metadata.get("area", 0),
                "has_ocr_text": bool(extracted_text and extracted_text.strip()),
                "ocr_text_length": len(extracted_text) if extracted_text else 0
            }
            
            # Add OCR text to metadata if available
            if extracted_text and extracted_text.strip():
                image_metadata["extracted_text"] = extracted_text.strip()
            
            # Convert bbox to individual scalar values if present
            if "bbox" in metadata and metadata["bbox"] is not None:
                bbox = metadata["bbox"]
                image_metadata.update({
                    "bbox_x0": float(bbox[0]),
                    "bbox_y0": float(bbox[1]),
                    "bbox_x1": float(bbox[2]),
                    "bbox_y1": float(bbox[3]),
                    "bbox_width": float(bbox[2] - bbox[0]),
                    "bbox_height": float(bbox[3] - bbox[1])
                })
            
            # Convert dimensions to individual scalar values if present
            if "dimensions" in metadata and metadata["dimensions"]:
                dims = metadata["dimensions"]
                if len(dims) >= 2:
                    image_metadata.update({
                        "image_width": int(dims[0]),
                        "image_height": int(dims[1])
                    })
            
            # Add to collection (store description as document for search)
            self.collection.add(
                documents=[image_description],
                embeddings=[embedding],
                metadatas=[image_metadata],
                ids=[chunk_id]
            )
            
            logger.debug(f"Added image chunk with ID: {chunk_id}, bbox: {metadata.get('bbox', 'None')}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error adding image chunk: {e}")
            raise
    
    def add_spatial_relationships(self, relationships: List[Dict]) -> None:
        """
        Store spatial relationships in metadata for future reference.
        
        Args:
            relationships: List of spatial relationship dictionaries
        """
        try:
            # Store relationships in a separate collection or as metadata
            # For now, we'll log them and they can be used during retrieval
            for relationship in relationships:
                logger.debug(f"Spatial relationship: {relationship['text_chunk_id']} -> "
                           f"{relationship['image_id']} ({relationship['relationship_type']}, "
                           f"confidence: {relationship['confidence']:.2f})")
            
            # Store relationships count in collection metadata
            if hasattr(self.collection, 'modify'):
                try:
                    current_metadata = self.collection.metadata or {}
                    current_metadata['spatial_relationships_count'] = len(relationships)
                    self.collection.modify(metadata=current_metadata)
                except Exception as e:
                    logger.warning(f"Could not update collection metadata: {e}")
                    
        except Exception as e:
            logger.error(f"Error storing spatial relationships: {e}")
    
    def similarity_search(self, query: str, n_results: int = 5, content_type: str = None, use_spatial_boost: bool = True) -> List[Dict]:
        """
        Perform enhanced similarity search with spatial relationship awareness.
        
        Args:
            query: Search query
            n_results: Number of results to return
            content_type: Optional content type filter (text or image)
            use_spatial_boost: Whether to boost spatially related content
            
        Returns:
            List of matching chunks with metadata and spatial context
        """
        try:
            # Generate query embedding using Ollama
            query_embedding = self._generate_embedding(query)
            
            # Search in collection with expanded results for spatial processing
            search_multiplier = 3 if use_spatial_boost else 2
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results * search_multiplier,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process initial results
            processed_results = []
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0], 
                results['distances'][0]
            )):
                base_score = 1 - distance  # Convert distance to similarity
                spatial_boost = 0
                
                # Apply spatial boost if enabled
                if use_spatial_boost and metadata.get('has_bbox'):
                    spatial_boost = self._calculate_spatial_boost(metadata, processed_results)
                
                processed_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity_score': base_score,
                    'spatial_boost': spatial_boost,
                    'final_score': base_score + spatial_boost,
                    'rank': i + 1,
                    'has_spatial_info': metadata.get('has_bbox', False)
                })
            
            # Filter by content type if specified
            if content_type:
                processed_results = [
                    r for r in processed_results 
                    if r['metadata'].get('type') == content_type
                ]
            
            # Sort by final score (similarity + spatial boost)
            processed_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            # Add spatial context to results
            enhanced_results = self._add_spatial_context(processed_results[:n_results])
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def _calculate_spatial_boost(self, metadata: Dict, existing_results: List[Dict]) -> float:
        """
        Calculate spatial boost score based on proximity to other relevant content.
        
        Args:
            metadata: Metadata of current result
            existing_results: Previously processed results
            
        Returns:
            Spatial boost score (0.0 to 0.2)
        """
        boost = 0.0
        current_page = metadata.get('page')
        
        if not current_page:
            return boost
        
        # Boost for same-page content
        same_page_count = sum(1 for r in existing_results 
                             if r['metadata'].get('page') == current_page)
        if same_page_count > 0:
            boost += 0.05  # Small boost for same-page content
        
        # Boost for high-area content (likely important)
        area = metadata.get('area', 0)
        if area > 10000:  # Large content area
            boost += 0.03
        
        # Boost for content with spatial relationships
        if metadata.get('type') == 'image' and area > 5000:
            boost += 0.02  # Images are often important
        
        return min(boost, 0.2)  # Cap boost at 0.2
    
    def _add_spatial_context(self, results: List[Dict]) -> List[Dict]:
        """
        Add spatial context information to search results.
        
        Args:
            results: List of search results
            
        Returns:
            Enhanced results with spatial context
        """
        for result in results:
            metadata = result['metadata']
            spatial_context = {
                'has_coordinates': metadata.get('has_bbox', False),
                'page': metadata.get('page'),
                'area': metadata.get('area', 0),
                'content_type': metadata.get('type'),
            }
            
            # Add bounding box info if available
            if metadata.get('has_bbox') and all(k in metadata for k in ['bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1']):
                spatial_context.update({
                    'coordinates': {
                        'top_left': [metadata['bbox_x0'], metadata['bbox_y0']],
                        'bottom_right': [metadata['bbox_x1'], metadata['bbox_y1']],
                        'width': metadata['bbox_width'],
                        'height': metadata['bbox_height']
                    }
                })
            
            # Add image dimensions if available
            if metadata.get('image_width') and metadata.get('image_height'):
                spatial_context['image_dimensions'] = [metadata['image_width'], metadata['image_height']]
            
            result['spatial_context'] = spatial_context
        
        return results
    
    def get_all_documents(self, limit: int = None) -> List[Dict]:
        """
        Get all documents from the collection.
        
        Args:
            limit: Optional limit on number of documents
            
        Returns:
            List of all documents
        """
        try:
            results = self.collection.get(
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            for i in range(len(results['documents'])):
                result = {
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the collection.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            True if successful
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document with ID: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection.
        
        Returns:
            True if successful
        """
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Multimodal RAG collection for PDFs with text and images"}
            )
            logger.info("Cleared collection successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            all_docs = self.get_all_documents()
            
            text_count = sum(1 for doc in all_docs if doc['metadata'].get('type') == 'text')
            image_count = sum(1 for doc in all_docs if doc['metadata'].get('type') == 'image')
            
            stats = {
                'total_documents': len(all_docs),
                'text_chunks': text_count,
                'image_chunks': image_count,
                'collection_name': self.collection_name
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def hybrid_search(self, query: str, n_results: int = 10) -> Tuple[List[Dict], List[str]]:
        """
        Perform hybrid search returning both text and image results.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Tuple of (text_results, image_base64_list)
        """
        try:
            # Get all results
            all_results = self.similarity_search(query, n_results * 2)
            
            text_results = []
            image_results = []
            image_base64_list = []
            
            for result in all_results:
                if result['metadata'].get('type') == 'text':
                    text_results.append(result)
                elif result['metadata'].get('type') == 'image':
                    image_results.append(result)
                    if 'image_data' in result['metadata']:
                        image_base64_list.append(result['metadata']['image_data'])
            
            # Limit results
            text_results = text_results[:n_results//2] if len(text_results) > n_results//2 else text_results
            image_base64_list = image_base64_list[:n_results//2] if len(image_base64_list) > n_results//2 else image_base64_list
            
            return text_results, image_base64_list
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return [], []
