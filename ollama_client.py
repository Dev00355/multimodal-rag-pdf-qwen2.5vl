import ollama
import base64
import logging
from typing import List, Dict, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama and Qwen2.5-VL model."""
    
    def __init__(self, model_name: str = "qwen2.5-vl:7b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.host = host
        self.client = ollama.Client(host=host)
        
    def check_model_availability(self) -> bool:
        """Check if the Qwen2.5-VL model is available."""
        try:
            models = self.client.list()
            available_models = [model['name'] for model in models['models']]
            return self.model_name in available_models
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            return False
    
    def analyze_image(self, image_base64: str, prompt: str = None) -> str:
        """
        Analyze an image using Qwen2.5-VL model.
        
        Args:
            image_base64: Base64 encoded image
            prompt: Optional prompt for analysis
            
        Returns:
            Analysis result as string
        """
        try:
            if prompt is None:
                prompt = """Analyze this image in detail. Describe:
1. What type of content this is (diagram, chart, photo, text, etc.)
2. Key elements and components visible
3. Any text or labels present
4. The main purpose or message conveyed
5. Technical details if it's a diagram or workflow

Be comprehensive and specific in your description."""

            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_base64],
                stream=False
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return f"Error analyzing image: {str(e)}"
    
    def generate_text_summary(self, text: str, context: str = None) -> str:
        """
        Generate a summary or analysis of text content.
        
        Args:
            text: Text to analyze
            context: Optional context for the analysis
            
        Returns:
            Summary or analysis result
        """
        try:
            prompt = f"""Analyze and summarize the following text content:

{text}

Provide:
1. A concise summary of the main points
2. Key concepts and terms mentioned
3. Any technical information or procedures described
4. Context and significance of the content

{f"Additional context: {context}" if context else ""}

Summary:"""

            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                stream=False
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"Error generating text summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def answer_question(self, question: str, context_chunks: List[Dict], images: List[str] = None) -> str:
        """
        Answer a question based on provided context and images.
        
        Args:
            question: User's question
            context_chunks: Relevant text chunks from the knowledge base
            images: Optional list of base64 encoded images
            
        Returns:
            Answer to the question
        """
        try:
            # Prepare context
            text_context = "\n\n".join([
                f"Page {chunk.get('page', 'N/A')}: {chunk.get('content', '')}"
                for chunk in context_chunks
            ])
            
            prompt = f"""Based on the following context and any provided images, answer the user's question comprehensively.

Context:
{text_context}

Question: {question}

Instructions:
1. Use the provided context to answer the question
2. If images are provided, analyze them and incorporate visual information
3. Be specific and cite relevant parts of the context
4. If the answer isn't fully available in the context, clearly state what information is missing
5. Provide a clear, well-structured response

Answer:"""

            # Validate and filter images if provided
            valid_images = []
            if images:
                for i, img in enumerate(images):
                    if self._validate_image_data(img):
                        valid_images.append(img)
                    else:
                        logger.warning(f"Skipping invalid image data at index {i}")
                
                if images and not valid_images:
                    logger.warning(f"All {len(images)} provided images were invalid, proceeding with text-only response")
            
            # Include images if provided and valid
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=valid_images,
                stream=False
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error generating answer: {str(e)}"
    
    def extract_image_text(self, image_base64: str) -> str:
        """
        Extract text from an image using OCR capabilities of Qwen2.5-VL.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Extracted text
        """
        try:
            prompt = """Extract all visible text from this image. 
Include:
1. All readable text, labels, titles, and captions
2. Text in diagrams, charts, or flowcharts
3. Any handwritten text if visible
4. Maintain the structure and formatting where possible

Extracted text:"""

            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_base64],
                stream=False
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return f"Error extracting text: {str(e)}"
    
    def generate_image_embeddings_description(self, image_base64: str) -> str:
        """
        Generate a description suitable for creating embeddings.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Description for embedding generation
        """
        try:
            prompt = """Create a detailed description of this image that would be useful for semantic search and retrieval. Include:

1. Content type (diagram, chart, photo, screenshot, etc.)
2. Main subjects and objects
3. Key visual elements and their relationships
4. Any text or labels visible
5. Technical concepts or processes shown
6. Colors, shapes, and layout if relevant
7. Context and purpose

Focus on creating a description that captures the semantic meaning and would help match relevant user queries.

Description:"""

            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_base64],
                stream=False
            )
            
            return response['response']
            
        except Exception as e:
            logger.error(f"Error generating image description: {e}")
            return f"Error generating description: {str(e)}"
    
    def _validate_image_data(self, image_data: str) -> bool:
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
            base64.b64decode(image_data, validate=True)
            
            # Check minimum length (a valid image should be reasonably sized)
            if len(image_data) < 100:  # Very small base64 strings are likely invalid
                return False
                
            return True
            
        except Exception as e:
            logger.debug(f"Invalid base64 image data: {e}")
            return False
