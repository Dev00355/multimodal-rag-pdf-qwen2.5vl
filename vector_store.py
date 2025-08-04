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
        Add a text chunk to the vector store.
        
        Args:
            content: Text content
            metadata: Metadata dictionary
            chunk_id: Optional custom ID
            
        Returns:
            ID of the added chunk
        """
        try:
            if chunk_id is None:
                chunk_id = str(uuid.uuid4())
            
            # Generate embedding using Ollama
            embedding = self._generate_embedding(content)
            
            # Add to collection
            self.collection.add(
                documents=[content],
                embeddings=[embedding],
                metadatas=[{**metadata, "type": "text"}],
                ids=[chunk_id]
            )
            
            logger.debug(f"Added text chunk with ID: {chunk_id}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error adding text chunk: {e}")
            raise
    
    def add_image_chunk(self, image_description: str, image_base64: str, metadata: Dict, chunk_id: str = None) -> str:
        """
        Add an image chunk to the vector store.
        
        Args:
            image_description: Text description of the image
            image_base64: Base64 encoded image
            metadata: Metadata dictionary
            chunk_id: Optional custom ID
            
        Returns:
            ID of the added chunk
        """
        try:
            if chunk_id is None:
                chunk_id = str(uuid.uuid4())
            
            # Generate embedding from description using Ollama
            embedding = self._generate_embedding(image_description)
            
            # Store image data in metadata
            image_metadata = {
                **metadata,
                "type": "image",
                "image_data": image_base64,
                "description": image_description
            }
            
            # Add to collection
            self.collection.add(
                documents=[image_description],
                embeddings=[embedding],
                metadatas=[image_metadata],
                ids=[chunk_id]
            )
            
            logger.debug(f"Added image chunk with ID: {chunk_id}")
            return chunk_id
            
        except Exception as e:
            logger.error(f"Error adding image chunk: {e}")
            raise
    
    def similarity_search(self, query: str, n_results: int = 5, filter_metadata: Dict = None) -> List[Dict]:
        """
        Perform similarity search in the vector store.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of matching chunks with metadata
        """
        try:
            # Generate query embedding using Ollama
            query_embedding = self._generate_embedding(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['documents'][0])):
                result = {
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                }
                formatted_results.append(result)
            
            logger.debug(f"Found {len(formatted_results)} results for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise
    
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
